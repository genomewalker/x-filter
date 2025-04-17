import argparse
import sys
import gzip
import os
import logging
import time
from pathlib import Path
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from itertools import chain
from typing import List, Dict, Tuple, Union, Any, Optional
import numpy as np
import pandas as pd
import psutil

from x_filter import __version__
from x_filter.common import detect_input_type, validate_mmap_folder, get_open_func

# Logger setup
log = logging.getLogger("my_logger")
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
}

HELP_MESSAGES = {
    "input": "Input file/directory containing the data. Can be:\n"
    "  - A blastx m8 formatted TSV file containing aligned reads to references\n"
    "  - A Parquet file or directory containing Parquet files\n"
    "  - A DuckDB database with 'filtered_blast' table\n"
    "The input must contain query and subject lengths",
    "threads": "Number of threads to use",
    "prefix": "Prefix used for the output files",
    "bitscore": "Bitscore where to filter the results",
    "evalue": "Evalue where to filter the results",
    "filter": "Which filter to use. Possible values are: avgAlnLength, nAlns, avgReadLength, avgIdentity, breadth, avgDepth, covMean, covStd, depthEvenness, breadthExpectedRatio",
    "scale": "Scale threshold for selecting alignments (0-1). Lower values keep more alignments. 0=keep all alignments, 0.9=keep alignments ≥90% of maximum probability, 1.0=keep only the best alignments.",
    "evalue_perc": "Percentage of the -log(Evalue) to filter out results",
    "evalue_perc_step": "Step size to find the percentage of the -log(Evalue) to filter out results",
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
    "disable_initial_filtering": "Disable initial filtering before reassignment",
    "mmap_folder_dir": "Path to folder containing memory-mapped arrays. If provided, skips mmap export",
}


def is_debug() -> bool:
    return log.getEffectiveLevel() == logging.DEBUG


def get_available_memory():
    """
    Get the available system memory in bytes.
    """
    return psutil.virtual_memory().available


def get_default_max_memory():
    """
    Calculate the default max memory as 80% of available memory.
    Returns the value in bytes.
    """
    available_memory = get_available_memory()
    return int(available_memory * 0.8)


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
) -> Union[str, int]:
    unit = val[-1]
    value = val[:-1]

    # check if its None
    if value is None:
        return None

    if not (is_integer(value) and unit in UNITS and int(value) > 0):
        parser.error(
            f"argument {var}: Invalid value {val}. Has to be an integer larger than 0 with the following suffix K, M or G"
        )

    value = int(value)
    if var == "--scale":
        return str(value * 1000 if unit == "K" else value * 1000000)
    else:
        multiplier = 1024 ** (UNITS.index(unit) + 1)
        return value * multiplier


def validate_filters(
    filters: List[Dict[str, Union[str, float]]],
    filter_mapping: Dict[str, Tuple[str, str]],
) -> None:
    for f in filters:
        filter_name = f["filter_name"]
        value = f["value"]

        if filter_name not in filter_mapping:
            raise ValueError(
                f"Invalid filter name: {filter_name}. Please choose from {list(filter_mapping.keys())}."
            )

        try:
            float_value = float(value)
            if float_value < 0:
                raise ValueError(
                    f"Invalid value '{value}' for filter '{filter_name}'. Negative numbers are not allowed."
                )
        except ValueError:
            raise ValueError(
                f"Invalid value '{value}' for filter '{filter_name}'. Must be a numeric value."
            )


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
        "-n",
        "--n-iters",
        type=lambda x: int(
            check_values(x, minval=0, maxval=100000, parser=parser, var="--n-iters")
        ),
        default=DEFAULTS["iters"],
        help=HELP_MESSAGES["iters"],
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

    parser.add_argument("--anvio", action="store_true", help=HELP_MESSAGES["anvio"])
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
        help="Path to folder containing memory-mapped arrays. If provided, skips mmap export.",
    )
    parser.add_argument(
        "--keep-duplicates", action="store_false", help=HELP_MESSAGES["duplicates"]
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Save the exported Parquet file in the working directory instead of temp directory",
    )
    parser.add_argument(
        "--disable-initial-filtering",
        action="store_true",
        help=HELP_MESSAGES["disable_initial_filtering"],
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
            filter_name, filter_value = f.split("=")
            filters.append(
                {
                    "filter_name": filter_name.strip(),
                    "value": float(filter_value.strip()),
                }
            )

    validate_filters(filters, USER_FRIENDLY_FILTER_MAPPING)

    # Validate mmap folder if provided
    if args.mmap_folder_dir:
        # First detect input type
        try:
            input_type = detect_input_type(args.input)
        except ValueError as e:
            parser.error(str(e))

        if input_type not in ["parquet", "duckdb"]:
            parser.error("--mmap-folder can only be used with Parquet or DuckDB inputs")

        if not os.path.isdir(args.mmap_folder_dir):
            parser.error(
                f"mmap-folder {args.mmap_folder_dir} does not exist or is not a directory"
            )

        try:
            validate_mmap_folder(args.mmap_folder_dir)
        except ValueError as e:
            parser.error(str(e))

    return args, filters


@contextmanager
def suppress_stdout() -> Tuple[Any, Any]:
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


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
            # For parquet folder, use the folder name without .parquet extension
            # But ensure we're using just the name, not the full path
            prefix = Path(input_path.name).stem
        else:
            # For regular files, use the stem (filename without extension)
            prefix = input_path.stem.split(".")[0]

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
    ]:
        file_path = os.path.join(temp_dir, f"{column}.dat")
        arr = df[column].to_numpy()

        # Save the array to a file
        arr.tofile(file_path)

        # Create a memory-mapped array
        mmap_arr = np.memmap(file_path, dtype=arr.dtype, mode="r+", shape=arr.shape)

        mmap_arrays[column] = mmap_arr

    return mmap_arrays, temp_dir
