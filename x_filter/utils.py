from pathlib import Path
from numba import njit, prange
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from itertools import chain
from typing import List, Dict, Tuple, Union, Any, Optional
import logging
import time
import numpy as np
import pandas as pd
import psutil
import gc
import argparse
import sys
import os
from x_filter import __version__
from x_filter.common import detect_input_type, validate_mmap_folder, get_open_func
from x_filter.logging_setup import get_logger

# Logger setup
log = get_logger()
log.setLevel(logging.INFO)
timestr = time.strftime("%Y%m%d-%H%M%S")

# Constants
UNITS = ["K", "M", "G"]
DEFAULT_FILTERS = [{"filter_name": "depthEvenness", "value": 1.0}]

# User-friendly filter mappings
USER_FRIENDLY_FILTER_MAPPING = {
    "avgAlnLength": ("avg_alnLength", ">="),
    "nAlns": ("n_alns", ">="),
    "avgReadLength": ("avg_read_length", ">="),
    "avgIdentity": ("avg_identity", ">="),
    "breadth": ("breadth", ">="),
    "covMean": ("depth_mean", ">="),
    "depthEvenness": ("depth_evenness", "<="),
}

# Default values and help messages
DEFAULTS = {
    "bitscore": 60,
    "evalue": 1e-10,
    "percent_identity": 0.0,
    "depth_evenness": 1.0,
    "prefix": None,
    "sort_memory": "1G",
    "mapping_file": None,
    "iters": 25,
    "scale": 0.9,
    "filter": "depthEvenness",
    "annotation_source": "unknown",
    "evalue_perc": None,
    "evalue_perc_step": 0.1,
    "tmp_dir": None,
    "min_assignment_confidence": 0.01,
    "min_confidence_margin": 0.001,
    "handle_ties": "keep_all",
    "selection_mode": "primary",
    "reference_bias": 0.0,
    "confidence_decay": "none",
    "adaptive_convergence": False,
    "reassign_mode": "em",
    "acceleration_method": "hybrid",
    "anderson_memory": 10,
    "lbfgs_memory": 10,
    "lambda_scale": 3.0,  # Balanced lambda scale for stable convergence
    "min_improvement": 1e-4,
    "max_consecutive_failures": 3,
    "convergence_threshold": 1e-4,
    "n_iters": 20,
    "assignment_threshold": 0.5,
    "confidence_threshold": 0.9,
    "random_seed": 42,
}

HELP_MESSAGES = {
    "input": "Input file/directory containing the data. Can be:\n"
    "  - A blastx m8 formatted TSV file containing aligned reads to references\n"
    "  - A directory containing multiple TSV files (will be processed together)\n"
    "  - A Parquet file or directory containing Parquet files\n"
    "  - A DuckDB database with 'filtered_blast' table\n"
    "The input must contain query and subject lengths",
    "threads": "Number of threads to use",
    "prefix": "Prefix used for the output files",
    "bitscore": "Bitscore where to filter the results",
    "evalue": "Evalue where to filter the results",
    "percent_identity": "Minimum percent identity to filter the results (0.0-1.0)",  # Updated help message
    "filter": "Which filter to use. Possible values are: avgAlnLength, nAlns, avgReadLength, avgIdentity, breadth, avgDepth, covMean, covStd, depthEvenness, breadthExpectedRatio",
    "scale": "Scale threshold for selecting alignments (0-1). Lower values keep more alignments. 0=keep all alignments, 0.9=keep alignments ≥90%% of maximum probability, 1.0=keep only the best alignments.",
    "mapping_file": "File with mappings to genes for aggregation",
    "iters": "Number of iterations for the FAMLI-like filtering",
    "annotation_source": "Source of the annotation",
    "debug": "Print debug messages",
    "version": "Print program version",
    "anvio": "Create output compatible with anvi'o",
    "trim": "Deactivate the trimming for the coverage calculations",
    "max_memory": "Maximum memory to use. If not provided will use 80%% of the available memory",
    "tmp_dir": "Temporary directory to store intermediate files",
    "duplicates": "Keep duplicated reads in the output",
    "keep_db": "Save the exported Parquet file in the working directory instead of temp directory",
    "enable_initial_filtering": "Enable initial filtering before reassignment",
    "mmap_folder_dir": "Path to folder containing memory-mapped arrays. If provided, skips mmap export",
    "min_assignment_confidence": "Minimum confidence required for an alignment to be kept (0.0-1.0). For highly multi-mapping datasets, use very low values (0.001-0.01). Higher values keep fewer, more confident alignments.",
    "min_confidence_margin": "Minimum margin between primary and secondary alignment probabilities. For multi-mapping data, use very low values (0.0001-0.001). Higher values require larger gaps between best hits.",
    "handle_ties": "How to handle tied alignments with equal probabilities: 'keep_all' keeps all tied alignments, 'keep_one' keeps one random alignment, 'discard' removes reads with ties.",
    "selection_mode": "Strategy for selecting alignments from multimapping reads: 'primary' uses confidence thresholds, 'weighted' keeps single best alignment, 'proportional' assigns based on relative confidence, 'threshold' uses hard cutoffs, 'all' keeps all alignments.",
    "reference_bias": "Bias factor for reference selection (-1.0 to 1.0). Positive values favor longer references, negative values favor shorter ones.",
    "confidence_decay": "How confidence scores decay: 'linear' applies linear scaling, 'exponential' applies exponential decay to low confidence alignments, 'sigmoid' uses a sigmoid function for smoother transition, 'none' uses raw scores.",
    "adaptive_convergence": "Use adaptive convergence criteria instead of fixed iterations for the EM algorithm",
    "reassign_mode": "Mode for handling multi-mapped reads: 'em' (use EM algorithm), 'best_score' (assign to highest scoring reference), 'all' (keep all mappings), or 'discard' (remove multi-mapped reads)",
    "acceleration_method": "Acceleration method for EM algorithm: 'anderson' (Anderson acceleration), 'lbfgs' (L-BFGS), or 'hybrid' (combines both methods)",
    "anderson_memory": "Memory depth for Anderson acceleration (number of previous iterations to store)",
    "lbfgs_memory": "Memory depth for L-BFGS acceleration (number of previous iterations to store)",
    "lambda_scale": "Scale parameter for exponential transformation of bit scores in EM algorithm. Higher values make the algorithm more selective (default: 1.0)",
    "min_improvement": "Minimum improvement threshold for EM convergence. Smaller values allow more iterations before convergence (default: 1e-4)",
    "max_consecutive_failures": "Maximum number of consecutive acceleration failures before falling back to basic EM",
    "convergence_threshold": "Threshold for likelihood changes to determine EM convergence (default: 1e-4)",
    "n_iters": "Maximum number of EM iterations to perform (default: 20)",
}


@njit(parallel=True, fastmath=True)
def safe_normalize_probabilities(
    prob: np.ndarray,
    query_indices: np.ndarray,
    mask: np.ndarray,
    max_query: int,
) -> np.ndarray:
    """
    Safely normalize probabilities to sum to 1 per query.
    Handles edge cases like zero sums.
    """
    result = np.zeros_like(prob)
    tiny = np.finfo(np.float64).tiny
    
    # Compute sums per query
    query_sums = np.zeros(max_query + 1, dtype=np.float64)
    for i in prange(len(prob)):
        if mask[i]:
            query_sums[query_indices[i]] += prob[i]
    
    # Ensure no zero sums
    for i in prange(len(query_sums)):
        if query_sums[i] < tiny:
            query_sums[i] = tiny
    
    # Normalize
    for i in prange(len(prob)):
        if mask[i]:
            result[i] = prob[i] / query_sums[query_indices[i]]
    
    return result


@njit(parallel=True, fastmath=True)
def compute_effective_sample_size(
    prob: np.ndarray,
    query_indices: np.ndarray,
    mask: np.ndarray,
    max_query: int,
) -> float:
    """
    Compute effective sample size based on probability distribution.
    Higher values indicate more uniform distributions.
    """
    total_ess = 0.0
    query_count = 0
    
    for query_idx in prange(max_query + 1):
        sum_prob = 0.0
        sum_prob_sq = 0.0
        count = 0
        
        for i in range(len(prob)):
            if mask[i] and query_indices[i] == query_idx:
                p = prob[i]
                sum_prob += p
                sum_prob_sq += p * p
                count += 1
        
        if count > 0 and sum_prob > 0:
            ess = (sum_prob * sum_prob) / sum_prob_sq
            total_ess += ess
            query_count += 1
    
    return total_ess / max(query_count, 1)


@njit(fastmath=True)
def check_probability_validity(prob: np.ndarray, mask: np.ndarray) -> Tuple[bool, str]:
    """
    Check if probability array is valid (non-negative, finite).
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    for i in range(len(prob)):
        if mask[i]:
            if not np.isfinite(prob[i]):
                return (False, "Non-finite probabilities detected")
            if prob[i] < 0:
                return (False, "Negative probabilities detected")
    
    return (True, "")


def create_memory_mapped_copy(
    array: np.ndarray,
    mmap_folder: str,
    prefix: str,
) -> np.memmap:
    """Create a memory-mapped copy of an array."""
    filename = os.path.join(mmap_folder, f"{prefix}_copy.dat")
    mmap_array = np.memmap(
        filename, dtype=array.dtype, mode="w+", shape=array.shape
    )
    mmap_array[:] = array[:]
    mmap_array.flush()
    return mmap_array


def estimate_memory_usage(arrays: Dict[str, np.ndarray]) -> float:
    """Estimate total memory usage of arrays in GB."""
    total_bytes = 0
    for array in arrays.values():
        total_bytes += array.nbytes
    return total_bytes / (1024 ** 3)


def get_available_memory() -> float:
    """Get available system memory in GB."""
    memory = psutil.virtual_memory()
    return memory.available / (1024 ** 3)


def force_garbage_collection():
    """Force garbage collection and return collected objects count."""
    collected = gc.collect()
    return collected


class MemoryMonitor:
    """Monitor memory usage during computation."""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.iteration_count = 0
        self.peak_memory_gb = 0.0
        
    def check_memory(self, iteration: Optional[int] = None):
        """Check current memory usage and log if needed."""
        if iteration is not None:
            self.iteration_count = iteration
        else:
            self.iteration_count += 1
            
        current_memory = self._get_memory_usage_gb()
        self.peak_memory_gb = max(self.peak_memory_gb, current_memory)
        
        if self.iteration_count % self.log_interval == 0:
            log.info(f"Memory usage at iteration {self.iteration_count}: {current_memory:.2f}GB (peak: {self.peak_memory_gb:.2f}GB)")
            
        return current_memory
    
    def _get_memory_usage_gb(self) -> float:
        """Get current process memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def get_summary(self) -> Dict[str, float]:
        """Get memory usage summary."""
        return {
            "current_memory_gb": self._get_memory_usage_gb(),
            "peak_memory_gb": self.peak_memory_gb,
            "available_memory_gb": get_available_memory(),
        }


class TemporaryFileManager:
    """Manage temporary files for EM algorithm."""
    
    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        self.temp_files = []
        
    def create_temp_file(self, prefix: str, suffix: str = ".dat") -> str:
        """Create a temporary file and track it for cleanup."""
        filename = os.path.join(self.base_folder, f"{prefix}_{len(self.temp_files)}{suffix}")
        self.temp_files.append(filename)
        return filename
    
    def cleanup(self):
        """Clean up all temporary files."""
        for filename in self.temp_files:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except OSError:
                pass
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def validate_inputs(
    prob: np.ndarray,
    mask: np.ndarray,
    slen: np.ndarray,
    query_indices: np.ndarray,
) -> None:
    """Validate input arrays for EM algorithm."""
    # Check shapes
    if not all(arr.shape == prob.shape for arr in [mask, slen, query_indices]):
        raise ValueError("All input arrays must have the same shape")
    
    # Check data types
    if prob.dtype != np.float64:
        raise ValueError("Probability array must be float64")
    
    if mask.dtype != np.bool_:
        raise ValueError("Mask array must be boolean")
    
    # Check for valid ranges
    if np.any(prob[mask] < 0):
        raise ValueError("Probabilities must be non-negative")
    
    if np.any(slen[mask] <= 0):
        raise ValueError("Subject lengths must be positive")
    
    if np.any(query_indices[mask] < 0):
        raise ValueError("Query indices must be non-negative")
    
    log.debug("Input validation passed")


def is_debug() -> bool:
    """Check if debug logging is enabled."""
    return log.getEffectiveLevel() <= logging.DEBUG


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate units."""
    if isinstance(num, float):
        if num < 1000:
            return f"{num:.1f}"
        elif num < 1_000_000:
            return f"{num/1000:.1f}K"
        elif num < 1_000_000_000:
            return f"{num/1_000_000:.1f}M"
        else:
            return f"{num/1_000_000_000:.1f}G"
    else:
        if num < 1000:
            return f"{num:,}"
        elif num < 1_000_000:
            return f"{num//1000}K"
        elif num < 1_000_000_000:
            return f"{num//1_000_000}M"
        else:
            return f"{num//1_000_000_000}G"


def get_default_max_memory():
    """
    Calculate the default max memory as 80% of available memory.
    Returns the value as a string with units (e.g., "32GB").
    """
    available_memory_gb = get_available_memory()
    default_memory_gb = max(1, int(available_memory_gb * 0.8))
    return f"{default_memory_gb}GB"


def is_integer(n: Any) -> bool:
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def convert_list_to_str(lst: List[str]) -> str:
    if not lst:
        return ""
    if len(lst) == 1:
        return lst[0]
    return ", ".join(lst[:-1]) + f" or {lst[-1]}"


def check_suffix(
    val: str, parser: argparse.ArgumentParser, var: str
) -> str:
    """Return memory limit as string with units instead of converting to bytes."""
    unit = val[-1]
    value = val[:-1]

    # check if its None
    if value is None:
        return None

    # Support both single-letter (K, M, G) and multi-letter (KB, MB, GB) units
    valid_single_units = ["K", "M", "G"]
    valid_multi_units = ["KB", "MB", "GB", "KIB", "MIB", "GIB"]
    
    # Check for multi-letter units first
    is_valid_unit = False
    if len(val) >= 3 and val[-2:].upper() in valid_multi_units:
        unit = val[-2:].upper()
        value = val[:-2]
        is_valid_unit = True
    elif len(val) >= 4 and val[-3:].upper() in valid_multi_units:
        unit = val[-3:].upper()
        value = val[:-3]
        is_valid_unit = True
    elif unit.upper() in valid_single_units:
        is_valid_unit = True
        unit = unit.upper()

    if not (is_integer(value) and is_valid_unit and int(value) > 0):
        parser.error(
            f"argument {var}: Invalid value {val}. Has to be an integer larger than 0 with the following suffix: K, M, G, KB, MB, GB, KIB, MIB, GIB"
        )

    # For memory limits, return the original string format (but ensure consistent case)
    if var == "--max-memory":
        return f"{value}{unit}"
    
    # For other uses (like scale), convert to appropriate format
    value = int(value)
    if var == "--scale":
        return str(value * 1000 if unit in ["K", "KB", "KIB"] else value * 1000000)
    else:
        # Convert to bytes
        if unit in ["K", "KB", "KIB"]:
            multiplier = 1024
        elif unit in ["M", "MB", "MIB"]:
            multiplier = 1024 ** 2
        elif unit in ["G", "GB", "GIB"]:
            multiplier = 1024 ** 3
        else:
            multiplier = 1
        return value * multiplier


def validate_filters(
    filters: List[Dict[str, Union[str, float]]],
    filter_mapping: Dict[str, Tuple[str, str]],
) -> None:
    for f in filters:
        filter_name = f["filter_name"]
        value = f["value"]

        if filter_name not in filter_mapping:
            raise ValueError(f"Unknown filter: {filter_name}")

        try:
            float(value)
        except ValueError:
            raise ValueError(f"Invalid value for filter {filter_name}: {value}")


def check_values(
    val: Union[int, float],
    minval: Union[int, float],
    maxval: Union[int, float],
    parser: argparse.ArgumentParser,
    var: str,
) -> Union[int, float]:
    value = float(val)
    if not minval <= value <= maxval:
        parser.error(
            f"argument {var}: Invalid value. Range has to be between {minval} and {maxval}!"
        )
    return value


def is_valid_file(parser: argparse.ArgumentParser, arg: str, var: str) -> str:
    if not os.path.exists(arg):
        parser.error(f"argument {var}: The file/directory {arg} does not exist!")
    return arg


def get_arguments(
    argv: Optional[List[str]] = None,
) -> Tuple[argparse.Namespace, List[Dict[str, Union[str, float]]]]:
    parser = argparse.ArgumentParser(
        description="A simple tool to filter BLASTx results using the FAMLI algorithm. "
        "Supports TSV, Parquet, and DuckDB input formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=lambda x: is_valid_file(parser, x, "--input"),
        required=True,
        help=HELP_MESSAGES["input"],
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=lambda x: int(
            check_values(x, minval=1, maxval=1000, parser=parser, var="--threads")
        ),
        default=1,
        help=HELP_MESSAGES["threads"],
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=DEFAULTS["prefix"],
        help=HELP_MESSAGES["prefix"],
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1, parser=parser, var="--scale")
        ),
        default=DEFAULTS["scale"],
        help=HELP_MESSAGES["scale"],
    )
    parser.add_argument(
        "--filters",
        type=str,
        required=False,
        help="Specify filters in the format: filterName=value,filterName=value (e.g., 'avgAlnLength=150,breadth=0.8')",
    )
    parser.add_argument(
        "-b",
        "--bitscore",
        type=lambda x: int(
            check_values(x, minval=0, maxval=1e6, parser=parser, var="--bitscore")
        ),
        default=DEFAULTS["bitscore"],
        help=HELP_MESSAGES["bitscore"],
    )
    parser.add_argument(
        "-e",
        "--evalue",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1e6, parser=parser, var="--evalue")
        ),
        default=DEFAULTS["evalue"],
        help=HELP_MESSAGES["evalue"],
    )
    parser.add_argument(
        "--percent-identity",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1, parser=parser, var="--percent-identity")
        ),
        default=DEFAULTS["percent_identity"],
        dest="percent_identity",
        help=HELP_MESSAGES["percent_identity"],
    )
    parser.add_argument(
        "-m",
        "--mapping-file",
        type=lambda x: is_valid_file(parser, x, "mapping_file"),
        default=DEFAULTS["mapping_file"],
        help=HELP_MESSAGES["mapping_file"],
    )
    parser.add_argument(
        "--no-trim", dest="trim", action="store_false", help=HELP_MESSAGES["trim"]
    )
    parser.add_argument(
        "--skip-reassign",
        dest="skip_reassign",
        action="store_true",
        help="Skip the reassignment step",
    )

    # EM algorithm arguments (moved to appear first)
    em_group = parser.add_argument_group("EM algorithm arguments")
    em_group.add_argument(
        "--adaptive-convergence",
        action="store_true",
        default=DEFAULTS["adaptive_convergence"],
        dest="adaptive_convergence",
        help=HELP_MESSAGES["adaptive_convergence"],
    )
    em_group.add_argument(
        "-n",
        "--n-iters",
        type=lambda x: int(
            check_values(x, minval=1, maxval=1000, parser=parser, var="--n-iters")
        ),
        default=DEFAULTS["n_iters"],
        dest="n_iters",
        help=HELP_MESSAGES["n_iters"],
    )
    em_group.add_argument(
        "--anderson-memory",
        type=lambda x: int(
            check_values(x, minval=1, maxval=50, parser=parser, var="--anderson-memory")
        ),
        default=DEFAULTS["anderson_memory"],
        dest="anderson_memory",
        help=HELP_MESSAGES["anderson_memory"],
    )
    em_group.add_argument(
        "--lbfgs-memory",
        type=lambda x: int(
            check_values(x, minval=1, maxval=50, parser=parser, var="--lbfgs-memory")
        ),
        default=DEFAULTS["lbfgs_memory"],
        dest="lbfgs_memory", 
        help=HELP_MESSAGES["lbfgs_memory"],
    )
    em_group.add_argument(
        "--lambda-scale",
        type=lambda x: float(
            check_values(x, minval=0, maxval=10, parser=parser, var="--lambda-scale")
        ),
        default=DEFAULTS["lambda_scale"],
        dest="lambda_scale",
        help=HELP_MESSAGES["lambda_scale"],
    )
    em_group.add_argument(
        "--min-improvement",
        type=lambda x: float(
            check_values(x, minval=1e-6, maxval=1e-2, parser=parser, var="--min-improvement")
        ),
        default=DEFAULTS["min_improvement"],
        dest="min_improvement",
        help=HELP_MESSAGES["min_improvement"],
    )
    em_group.add_argument(
        "--max-consecutive-failures",
        type=lambda x: int(
            check_values(x, minval=0, maxval=100, parser=parser, var="--max-consecutive-failures")
        ),
        default=DEFAULTS["max_consecutive_failures"],
        dest="max_consecutive_failures",
        help=HELP_MESSAGES["max_consecutive_failures"],
    )
    em_group.add_argument(
        "--convergence-threshold",
        type=lambda x: float(
            check_values(x, minval=1e-6, maxval=1e-2, parser=parser, var="--convergence-threshold")
        ),
        default=DEFAULTS["convergence_threshold"],
        dest="convergence_threshold",
        help=HELP_MESSAGES["convergence_threshold"],
    )

    # Selection arguments (updated title)
    selection_group = parser.add_argument_group("Selection arguments")
    selection_group.add_argument(
        "--min-assignment-confidence",
        type=lambda x: float(
            check_values(x, minval=0.0, maxval=1.0, parser=parser, var="--min-assignment-confidence")
        ),
        default=DEFAULTS["min_assignment_confidence"],
        dest="min_assignment_confidence",
        help=HELP_MESSAGES["min_assignment_confidence"],
    )
    selection_group.add_argument(
        "--min-confidence-margin",
        type=lambda x: float(
            check_values(x, minval=0.0, maxval=1.0, parser=parser, var="--min-confidence-margin")
        ),
        default=DEFAULTS["min_confidence_margin"],
        dest="min_confidence_margin",
        help=HELP_MESSAGES["min_confidence_margin"],
    )
    selection_group.add_argument(
        "--handle-ties",
        type=str,
        choices=["keep_all", "keep_one", "discard"],
        default=DEFAULTS["handle_ties"],
        dest="handle_ties",
        help=HELP_MESSAGES["handle_ties"],
    )
    selection_group.add_argument(
        "--selection-mode",
        type=str,
        choices=["primary", "weighted", "proportional", "threshold", "all"],
        default=DEFAULTS["selection_mode"],
        dest="selection_mode",
        help=HELP_MESSAGES["selection_mode"],
    )
    selection_group.add_argument(
        "--assignment-threshold",
        type=lambda x: float(
            check_values(x, minval=0.0, maxval=1.0, parser=parser, var="--assignment-threshold")
        ),
        default=DEFAULTS["assignment_threshold"],
        dest="assignment_threshold",
        help="Minimum probability threshold for assignments.",
    )
    selection_group.add_argument(
        "--confidence-threshold",
        type=lambda x: float(
            check_values(x, minval=0.0, maxval=1.0, parser=parser, var="--confidence-threshold")
        ),
        default=DEFAULTS["confidence_threshold"],
        dest="confidence_threshold",
        help="Confidence threshold for high-quality assignments.",
    )
    selection_group.add_argument(
        "--tie-breaking-method",
        type=str,
        choices=["random", "deterministic"],
        default="random",
        dest="tie_breaking_method",
        help="Method for breaking ties between equal assignments.",
    )
    selection_group.add_argument(
        "--reference-bias",
        type=lambda x: float(
            check_values(x, minval=-1.0, maxval=1.0, parser=parser, var="--reference-bias")
        ),
        default=DEFAULTS["reference_bias"],
        dest="reference_bias",
        help=HELP_MESSAGES["reference_bias"],
    )
    
    parser.add_argument(
        "--anvio", action="store_true", help=HELP_MESSAGES["anvio"]
    )
    parser.add_argument(
        "--annotation-source",
        type=str,
        default=DEFAULTS["annotation_source"],
        help=HELP_MESSAGES["annotation_source"],
    )
    parser.add_argument(
        "--max-memory",
        type=lambda x: check_suffix(x, parser=parser, var="--max-memory"),
        default=None,
        metavar="STR",
        help=HELP_MESSAGES["max_memory"],
    )
    parser.add_argument(
        "--seed",
        type=lambda x: int(
            check_values(x, minval=0, maxval=2**31 - 1, parser=parser, var="--seed")
        ),
        default=DEFAULTS["random_seed"],
        help="Deterministic seed for random number generation",
    )
        
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=DEFAULTS["tmp_dir"],
        metavar="DIR",
        help=HELP_MESSAGES["tmp_dir"],
    )
    parser.add_argument(
        "--mmap-folder-dir",
        type=str,
        default=None,
        help=HELP_MESSAGES["mmap_folder_dir"],
    )
    parser.add_argument(
        "--keep-duplicates", action="store_false", help=HELP_MESSAGES["duplicates"]
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help=HELP_MESSAGES["keep_db"],
    )
    parser.add_argument(
        "--enable-initial-filtering",
        action="store_true",
        dest="enable_initial_filtering",
        help=HELP_MESSAGES["enable_initial_filtering"],
    )
    parser.add_argument("--debug", action="store_true", help=HELP_MESSAGES["debug"])
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help=HELP_MESSAGES["version"],
    )

    args = parser.parse_args(None if sys.argv[1:] else ["-h"])

    if args.max_memory is None:
        args.max_memory = get_default_max_memory()

    filters = []
    if args.filters:
        for f in args.filters.split(","):
            filter_name, value = f.split("=")
            filters.append({"filter_name": filter_name.strip(), "value": float(value.strip())})

    validate_filters(filters, USER_FRIENDLY_FILTER_MAPPING)

    # Validate mmap folder if provided
    if args.mmap_folder_dir:
        # First detect input type
        try:
            input_type = detect_input_type(args.input)
        except ValueError as e:
            parser.error(f"Error detecting input type: {e}")

        if input_type not in ["parquet", "duckdb", "tsv"]:
            parser.error(f"Invalid input type for mmap folder: {input_type}")

        if not os.path.isdir(args.mmap_folder_dir):
            parser.error(f"mmap folder directory does not exist: {args.mmap_folder_dir}")

        try:
            validate_mmap_folder(args.mmap_folder_dir)
        except ValueError as e:
            parser.error(f"Invalid mmap folder: {e}")

    return args, filters


# Global variable to store current filters
_current_filters = []

def set_current_filters(filters):
    """Store the current filters for access by other modules."""
    global _current_filters
    _current_filters = filters

def get_filters():
    """Get the currently applied filters."""
    return _current_filters


@contextmanager
def suppress_stdout() -> Tuple[Any, Any]:
    with open(devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield fnull, fnull


def fast_flatten(input_list: List[List[Any]]) -> List[Any]:
    return list(chain.from_iterable(input_list))


def concat_df(frames: List[pd.DataFrame]) -> pd.DataFrame:
    column_names = frames[0].columns
    df_dict = {
        col: fast_flatten(frame[col] for frame in frames) for col in column_names
    }
    return pd.DataFrame.from_dict(df_dict)[column_names]


def create_output_files(prefix: Optional[str], input_file: str) -> Dict[str, str]:
    if prefix is None:
        # Handle both parquet folder and regular file cases
        input_path = Path(input_file).resolve()
        if input_path.is_dir() and input_path.name.endswith(".parquet"):
            prefix = input_path.stem  # Remove .parquet extension
        else:
            prefix = input_path.stem

    # get timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    return {
        "multimap": f"{prefix}_no-multimap.tsv.gz",
        "coverage": f"{prefix}_cov-stats.tsv.gz",
        "kegg_coverage": f"{prefix}_kegg-cov-stats.tsv.gz",
        "group_abundances": f"{prefix}_group-abundances.tsv.gz",
        "group_abundances_anvio": f"{prefix}_group-abundances-anvio.tsv.gz",
        "group_abundances_agg": f"{prefix}_group-abundances-agg.tsv.gz",
        "parquet": f"{prefix}_filtered_blast-{timestamp}.parquet",
        "mmap": f"{prefix}_filtered_blast-{timestamp}-mmap",
    }


def apply_filters(
    df: pd.DataFrame, filters: List[Dict[str, Union[str, float]]]
) -> pd.DataFrame:
    query_str = [
        f"{USER_FRIENDLY_FILTER_MAPPING[f['filter_name']][0]} {USER_FRIENDLY_FILTER_MAPPING[f['filter_name']][1]} {f['value']}"
        for f in filters
    ]
    full_query = " & ".join(query_str)
    return df.query(full_query)


def create_mmap_arrays(
    df: pd.DataFrame, temp_dir: str
) -> Tuple[Dict[str, np.memmap], str]:
    mmap_arrays = {}

    for column in [
        "subject_numeric_id",
        "subjectStart",
        "subjectEnd",
        "alnLength",
        "qlen",
        "percIdentity",
        "slen",
        "rowid",  # Changed from row_hash to rowid
    ]:
        file_path = os.path.join(temp_dir, f"{column}.dat")
        arr = df[column].to_numpy()

        # Save the array to a file
        arr.tofile(file_path)

        # Create a memory-mapped array
        mmap_arr = np.memmap(file_path, dtype=arr.dtype, mode="r+", shape=arr.shape)

        mmap_arrays[column] = mmap_arr

    return mmap_arrays, temp_dir
