import argparse
import sys
import gzip
import os
import logging
import pandas as pd
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from x_filter import __version__
import time
from itertools import chain
from pathlib import Path
import datatable as dt

log = logging.getLogger("my_logger")
log.setLevel(logging.INFO)
timestr = time.strftime("%Y%m%d-%H%M%S")


def is_debug():
    return logging.getLogger("my_logger").getEffectiveLevel() == logging.DEBUG


filters = ["breadth", "depth", "depth_evenness", "breadth_expected_ratio"]

# From https://stackoverflow.com/a/59617044/15704171
def convert_list_to_str(lst):
    n = len(lst)
    if not n:
        return ""
    if n == 1:
        return lst[0]
    return ", ".join(lst[:-1]) + f" or {lst[-1]}"


def check_filter_values(val, parser, var):
    value = str(val)
    if value in filters:
        return value
    else:
        parser.error(
            f"argument {var}: Invalid value {value}. Filter has to be one of {convert_list_to_str(filters)}"
        )


def check_values(val, minval, maxval, parser, var):
    value = float(val)
    if value < minval or value > maxval:
        parser.error(
            f"argument {var}: Invalid value value. Range has to be between {minval} and {maxval}!"
        )
    return value


# From: https://note.nkmk.me/en/python-check-int-float/
def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


# function to check if the input value has K, M or G suffix in it
def check_suffix(val, parser, var):
    units = ["K", "M", "G"]
    unit = val[-1]
    value = int(val[:-1])

    if is_integer(value) & (unit in units) & (value > 0):
        return val
    else:
        parser.error(
            "argument %s: Invalid value %s. Memory has to be an integer larger than 0 with the following suffix K, M or G"
            % (var, val)
        )


def get_compression_type(filename):
    """
    Attempts to guess the compression (if any) on a file using the first few bytes.
    http://stackoverflow.com/questions/13044562
    """
    magic_dict = {
        "gz": (b"\x1f", b"\x8b", b"\x08"),
        "bz2": (b"\x42", b"\x5a", b"\x68"),
        "zip": (b"\x50", b"\x4b", b"\x03", b"\x04"),
    }
    max_len = max(len(x) for x in magic_dict)

    unknown_file = open(filename, "rb")
    file_start = unknown_file.read(max_len)
    unknown_file.close()
    compression_type = "plain"
    for file_type, magic_bytes in magic_dict.items():
        if file_start.startswith(magic_bytes):
            compression_type = file_type
    if compression_type == "bz2":
        sys.exit("Error: cannot use bzip2 format - use gzip instead")
        sys.exit("Error: cannot use zip format - use gzip instead")
    return compression_type


def get_open_func(filename):
    if get_compression_type(filename) == "gz":
        return gzip.open
    else:  # plain text
        return open


# From: https://stackoverflow.com/a/11541450
def is_valid_file(parser, arg, var):
    if not os.path.exists(arg):
        parser.error("argument %s: The file %s does not exist!" % (var, arg))
    else:
        return arg


defaults = {
    "bitscore": 60,
    "evalue": 1e-10,
    "breadth": 0.5,
    "breadth_expected_ratio": 0.5,
    "depth": 0.1,
    "depth_evenness": 1.0,
    "prefix": None,
    "sort_memory": "1G",
    "mapping_file": None,
    "iters": 25,
    "scale": 0.9,
    "filter": "breadth_expected_ratio",
    "annotation_source": "unknown",
    "evalue_perc": None,
    "evalue_perc_step": 0.1,
}

help_msg = {
    "input": "A blastx m8 formatted file containing aligned reads to references. It has to contain query and subject lengths",
    "threads": "Number of threads to use",
    "prefix": "Prefix used for the output files",
    "bitscore": "Bitscore where to filter the results",
    "evalue": "Evalue where to filter the results",
    "filter": "Which filter to use. Possible values are: breadth, depth, depth_evenness, breadth_expected_ratio",
    "breadth": "Breadth of the coverage",
    "breadth_expected_ratio": "Expected breath to observed breadth ratio (scaled)",
    "depth": "Depth to filter out",
    "depth_evenness": "Reference with higher evenness will be removed",
    "mapping_file": "File with mappings to genes for aggregation",
    "iters": "Number of iterations for the FAMLI-like filtering",
    "scale": "Scale to select the best weithing alignments",
    "evalue_perc": "Percentage of the -log(Evalue) to filter out results",
    "evalue_perc_step": "Step size to find the percentage of the -log(Evalue) to filter out results",
    "anvio": "Create output compatible with anvi'o",
    "annotation_source": "Source of the annotation",
    "help": "Help message",
    "debug": f"Print debug messages",
    "version": f"Print program version",
    "trim": f"Deactivate the trimming for the coverage calculations",
}


def get_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="A simple tool to filter BLASTx m8 files using the FAMLI algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=lambda x: is_valid_file(parser, x, "--input"),
        help=help_msg["input"],
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=lambda x: int(
            check_values(x, minval=1, maxval=1000, parser=parser, var="--threads")
        ),
        dest="threads",
        default=1,
        help=help_msg["threads"],
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=defaults["prefix"],
        dest="prefix",
        help=help_msg["prefix"],
    )
    parser.add_argument(
        "-n",
        "--n-iters",
        type=lambda x: int(
            check_values(x, minval=1, maxval=100000, parser=parser, var="--n-iters")
        ),
        default=defaults["iters"],
        dest="iters",
        help=help_msg["iters"],
    )
    parser.add_argument(
        "-e",
        "--evalue",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1e6, parser=parser, var="--evalue")
        ),
        default=defaults["evalue"],
        dest="evalue",
        help=help_msg["evalue"],
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1, parser=parser, var="--scale")
        ),
        default=defaults["scale"],
        dest="scale",
        help=help_msg["scale"],
    )
    parser.add_argument(
        "-b",
        "--bitscore",
        type=lambda x: int(
            check_values(x, minval=0, maxval=1e6, parser=parser, var="--bitscore")
        ),
        default=defaults["bitscore"],
        dest="bitscore",
        help=help_msg["bitscore"],
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=lambda x: str(check_filter_values(x, parser=parser, var="--filter")),
        default=defaults["filter"],
        dest="filter",
        help=help_msg["filter"],
    )
    parser.add_argument(
        "--breadth",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1, parser=parser, var="--breadth")
        ),
        default=defaults["breadth"],
        dest="breadth",
        help=help_msg["breadth"],
    )
    parser.add_argument(
        "--breadth-expected-ratio",
        type=lambda x: float(
            check_values(
                x, minval=0, maxval=1, parser=parser, var="--breadth-expected-ratio"
            )
        ),
        default=defaults["breadth_expected_ratio"],
        dest="breadth_expected_ratio",
        help=help_msg["breadth_expected_ratio"],
    )
    parser.add_argument(
        "--depth",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1e6, parser=parser, var="--depth")
        ),
        default=defaults["depth"],
        dest="depth",
        help=help_msg["depth"],
    )
    parser.add_argument(
        "--depth-evenness",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1e6, parser=parser, var="--depth-evenness")
        ),
        default=defaults["depth_evenness"],
        dest="depth_evenness",
        help=help_msg["depth_evenness"],
    )
    parser.add_argument(
        "--evalue-perc",
        type=lambda x: float(
            check_values(x, minval=0, maxval=1.0, parser=parser, var="--evalue-perc")
        ),
        default=defaults["evalue_perc"],
        dest="evalue_perc",
        help=help_msg["evalue_perc"],
    )
    parser.add_argument(
        "--evalue-perc-step",
        type=lambda x: float(
            check_values(
                x, minval=0, maxval=1.0, parser=parser, var="--evalue-perc-step"
            )
        ),
        default=defaults["evalue_perc_step"],
        dest="evalue_perc_step",
        help=help_msg["evalue_perc_step"],
    )
    # reference_lengths
    parser.add_argument(
        "-m",
        "--mapping-file",
        type=lambda x: is_valid_file(parser, x, "mapping_file"),
        default=defaults["mapping_file"],
        dest="mapping_file",
        help=help_msg["mapping_file"],
    )
    parser.add_argument(
        "--no-trim", dest="trim", action="store_false", help=help_msg["trim"]
    )
    parser.add_argument(
        "--anvio", dest="anvio", action="store_true", help=help_msg["anvio"]
    )
    parser.add_argument(
        "--annotation-source",
        type=str,
        default=defaults["annotation_source"],
        dest="annotation_source",
        help=help_msg["annotation_source"],
    )
    parser.add_argument(
        "--debug", dest="debug", action="store_true", help=help_msg["debug"]
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
        help=help_msg["version"],
    )
    args = parser.parse_args(None if sys.argv[1:] else ["-h"])
    return args


@contextmanager
def suppress_stdout():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def fast_flatten(input_list):
    return list(chain.from_iterable(input_list))


def concat_df(frames):
    COLUMN_NAMES = frames[0].columns
    df_dict = dict.fromkeys(COLUMN_NAMES, [])
    for col in COLUMN_NAMES:
        extracted = (frame[col] for frame in frames)
        # Flatten and save to df_dict
        df_dict[col] = fast_flatten(extracted)
    df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
    return df


def create_output_files(prefix, input):
    if prefix is None:
        prefix = Path(input).resolve().stem.split(".")[0]
    # create output files
    out_files = {
        "multimap": f"{prefix}_no-multimap.tsv.gz",
        "coverage": f"{prefix}_cov-stats.tsv.gz",
        "kegg_coverage": f"{prefix}_kegg-cov-stats.tsv.gz",
        "group_abundances": f"{prefix}_group-abundances.tsv.gz",
        "group_abundances_anvio": f"{prefix}_group-abundances-anvio.tsv.gz",
        "group_abundances_agg": f"{prefix}_group-abundances-agg.tsv.gz",
    }
    return out_files


def create_filter_conditions(filter_type, filter_conditions):
    if filter_type in ["breadth", "depth", "breadth_expected_ratio"]:
        dt_filter = dt.f[filter_type] >= filter_conditions[filter_type]
    else:
        dt_filter = dt.f[filter_type] <= filter_conditions[filter_type]
    return dt_filter
