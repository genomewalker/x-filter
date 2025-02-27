# x_filter/cli.py

import argparse
import sys
import os
from typing import Tuple, List, Dict, Any, Optional
import textwrap
from pathlib import Path

from x_filter._version import get_versions
from x_filter.config import Config, DEFAULT_CONFIG

# Get version from versioneer
__version__ = get_versions()["version"]


def parse_arguments() -> Tuple[argparse.Namespace, List[Dict[str, Any]]]:
    """
    Parse command-line arguments for xFilter.

    Returns:
        Tuple of (parsed_args, filters)
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="xFilter: A BLASTx filtering tool with FAMLI algorithm for ancient DNA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input and output options
    input_group = parser.add_argument_group("Input/Output Options")
    input_group.add_argument(
        "-i",
        "--input",
        required=True,
        help="A BLASTx m8 formatted file with query and subject lengths",
    )
    input_group.add_argument(
        "-p", "--prefix", default=None, help="Prefix for output files"
    )
    input_group.add_argument(
        "-m",
        "--mapping-file",
        default=None,
        help="File with mappings to genes for aggregation",
    )
    input_group.add_argument(
        "--mmap-folder-dir",
        default=None,
        help="Use existing memory-mapped files from this directory",
    )
    input_group.add_argument(
        "--tmp-dir", default=None, help="Directory for temporary files"
    )
    input_group.add_argument(
        "--keep-db", action="store_true", help="Keep DuckDB files after processing"
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "-t",
        "--threads",
        type=int,
        default=DEFAULT_CONFIG["threads"],
        help="Number of threads to use",
    )
    proc_group.add_argument(
        "--max-memory",
        default=DEFAULT_CONFIG["max_memory"],
        help="Maximum memory to use (e.g., '4G', '8G')",
    )
    proc_group.add_argument(
        "--skip-reassign", action="store_true", help="Skip multi-mapping resolution"
    )
    proc_group.add_argument(
        "--disable-initial-filtering",
        action="store_true",
        help="Disable initial filtering step",
    )

    # Filtering options
    filter_group = parser.add_argument_group("Filtering Options")
    filter_group.add_argument(
        "-e",
        "--evalue",
        type=float,
        default=DEFAULT_CONFIG["evalue"],
        help="E-value threshold for filtering",
    )
    filter_group.add_argument(
        "-b",
        "--bitscore",
        type=float,
        default=DEFAULT_CONFIG["bitscore"],
        help="Bit score threshold for filtering",
    )
    filter_group.add_argument(
        "-f",
        "--filter",
        default="breadth_expected_ratio",
        choices=["breadth", "depth", "depth_evenness", "breadth_expected_ratio"],
        help="Filter type to use",
    )
    filter_group.add_argument(
        "--breadth",
        type=float,
        default=DEFAULT_CONFIG["filters"]["breadth"],
        help="Breadth of coverage threshold",
    )
    filter_group.add_argument(
        "--depth",
        type=float,
        default=DEFAULT_CONFIG["filters"]["depth"],
        help="Depth threshold",
    )
    filter_group.add_argument(
        "--depth-evenness",
        type=float,
        default=DEFAULT_CONFIG["filters"]["depth_evenness"],
        help="Depth evenness threshold (lower is more even)",
    )
    filter_group.add_argument(
        "--breadth-expected-ratio",
        type=float,
        default=DEFAULT_CONFIG["filters"]["breadth_expected_ratio"],
        help="Expected breadth to observed breadth ratio threshold",
    )

    # FAMLI algorithm options
    famli_group = parser.add_argument_group("FAMLI Algorithm Options")
    famli_group.add_argument(
        "-n",
        "--n-iters",
        type=int,
        default=DEFAULT_CONFIG["reassignment"]["iters"],
        help="Number of iterations for FAMLI algorithm",
    )
    famli_group.add_argument(
        "-s",
        "--scale",
        type=float,
        default=DEFAULT_CONFIG["reassignment"]["scale"],
        help="Scale to select best weighting alignments",
    )
    famli_group.add_argument(
        "--evalue-perc",
        type=float,
        default=None,
        help="Percentage of -log(E-value) to filter results",
    )
    famli_group.add_argument(
        "--evalue-perc-step",
        type=float,
        default=0.1,
        help="Step size for E-value percentage filter",
    )

    # Coverage calculation options
    coverage_group = parser.add_argument_group("Coverage Options")
    coverage_group.add_argument(
        "--no-trim",
        action="store_false",
        dest="trim",
        help="Disable trimming for coverage calculations",
    )

    # Output format options
    output_group = parser.add_argument_group("Output Format Options")
    output_group.add_argument(
        "--anvio", action="store_true", help="Create output compatible with anvi'o"
    )
    output_group.add_argument(
        "--annotation-source",
        default="unknown",
        help="Source of annotation for anvi'o output",
    )

    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    other_group.add_argument(
        "--version",
        action="version",
        version=f"xFilter v{__version__}",
        help="Show version and exit",
    )
    other_group.add_argument("--config", help="Configuration file (YAML or JSON)")

    # Parse arguments
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = Config(args.config)

        # Override with command-line arguments
        for key, value in vars(args).items():
            if value is not None and key != "config":
                if key in config.config:
                    config.config[key] = value

        # Update args with config values
        for key, value in config.config.items():
            if key not in vars(args) or getattr(args, key) is None:
                setattr(args, key, value)

    # Determine filters to apply
    filters = []

    if args.filter == "breadth":
        filters.append({"name": "breadth", "threshold": args.breadth})
    elif args.filter == "depth":
        filters.append({"name": "depth", "threshold": args.depth})
    elif args.filter == "depth_evenness":
        filters.append({"name": "depth_evenness", "threshold": args.depth_evenness})
    elif args.filter == "breadth_expected_ratio":
        filters.append(
            {"name": "breadth_expected_ratio", "threshold": args.breadth_expected_ratio}
        )

    # Generate output file prefix if not provided
    if args.prefix is None:
        args.prefix = os.path.splitext(os.path.basename(args.input))[0]

    return args, filters


def create_output_files(
    prefix: str, input_file: Optional[str] = None
) -> Dict[str, str]:
    """
    Create paths for output files.

    Args:
        prefix: Prefix for output files
        input_file: Input file name (for default prefix)

    Returns:
        Dictionary of output file paths
    """
    # Generate default prefix from input file if not provided
    if prefix is None and input_file is not None:
        prefix = os.path.splitext(os.path.basename(input_file))[0]
    elif prefix is None:
        prefix = "xfilter_output"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define output files
    return {
        "multimap": f"{prefix}_no-multimap.tsv.gz",
        "coverage": f"{prefix}_cov-stats.tsv.gz",
        "group_abundances": f"{prefix}_group-abundances.tsv.gz",
        "group_abundances_agg": f"{prefix}_group-abundances-agg.tsv.gz",
        "group_abundances_anvio": f"{prefix}_group-abundances-anvio.tsv.gz",
        "mmap": f"{prefix}_mmap",
        "parquet": f"{prefix}.parquet",
    }
