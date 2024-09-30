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
    "covMean": ("cov_mean", ">="),
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
    "input": "A blastx m8 formatted file containing aligned reads to references. It has to contain query and subject lengths",
    "threads": "Number of threads to use",
    "prefix": "Prefix used for the output files",
    "bitscore": "Bitscore where to filter the results",
    "evalue": "Evalue where to filter the results",
    "filter": "Which filter to use. Possible values are: avgAlnLength, nAlns, avgReadLength, avgIdentity, breadth, avgDepth, covMean, covStd, depthEvenness, breadthExpectedRatio",
    "scale": "Scale to select the best weighting alignments",
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


def get_compression_type(filename: str) -> str:
    magic_dict = {
        "gz": (b"\x1f", b"\x8b", b"\x08"),
        "bz2": (b"\x42", b"\x5a", b"\x68"),
        "zip": (b"\x50", b"\x4b", b"\x03", b"\x04"),
    }
    max_len = max(len(x) for x in magic_dict)

    with open(filename, "rb") as unknown_file:
        file_start = unknown_file.read(max_len)

    for file_type, magic_bytes in magic_dict.items():
        if file_start.startswith(magic_bytes):
            if file_type in ["bz2", "zip"]:
                sys.exit(f"Error: cannot use {file_type} format - use gzip instead")
            return file_type
    return "plain"


def get_open_func(filename: str) -> Union[gzip.open, open]:
    return gzip.open if get_compression_type(filename) == "gz" else open


def is_valid_file(parser: argparse.ArgumentParser, arg: str, var: str) -> str:
    if not os.path.exists(arg):
        parser.error(f"argument {var}: The file {arg} does not exist!")
    return arg


def get_arguments(
    argv: Optional[List[str]] = None,
) -> Tuple[argparse.Namespace, List[Dict[str, Union[str, float]]]]:
    parser = argparse.ArgumentParser(
        description="A simple tool to filter BLASTx m8 files using the FAMLI algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=lambda x: is_valid_file(parser, x, "--input"),
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
            check_values(x, minval=1, maxval=100000, parser=parser, var="--n-iters")
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
        "--keep-duplicates", action="store_true", help=HELP_MESSAGES["duplicates"]
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
        prefix = Path(input_file).resolve().stem.split(".")[0]

    return {
        "multimap": f"{prefix}_no-multimap.tsv.gz",
        "coverage": f"{prefix}_cov-stats.tsv.gz",
        "kegg_coverage": f"{prefix}_kegg-cov-stats.tsv.gz",
        "group_abundances": f"{prefix}_group-abundances.tsv.gz",
        "group_abundances_anvio": f"{prefix}_group-abundances-anvio.tsv.gz",
        "group_abundances_agg": f"{prefix}_group-abundances-agg.tsv.gz",
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
