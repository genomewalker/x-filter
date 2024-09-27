import numpy as np
import sys
import tqdm
import logging
import datatable as dt
import gc
import os
from functools import reduce
import pandas as pd
import gzip
import duckdb
import pyarrow.parquet as pq
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import duckdb


os.environ["POLARS_MAX_THREADS"] = "16"
import polars as pl
import zlib
import struct

log = logging.getLogger("my_logger")

sys.setrecursionlimit(10**6)

# estimate file size of a gzip compressed file



def calculate_coverage():
    pass


def aggregate_gene_abundances(mapping_file, gene_abundances, threads=1):
    """
    Aggregates gene abundances based on mapping file and gene abundance data.

    Args:
        mapping_file (str): Path to mapping file.
        gene_abundances (pandas.DataFrame): Gene abundance data.
        threads (int): Number of threads to use.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: A tuple containing two pandas DataFrames.
            The first DataFrame contains the merged mapping data and gene abundance data.
            The second DataFrame contains the aggregated gene abundances.
    """
    dt.options.progress.clear_on_success = True
    dt.options.nthreads = threads
    mappings = dt.fread(
        mapping_file, sep="\t", columns=["reference", "group"]
    ).to_pandas()
    mappings = mappings.merge(gene_abundances.to_pandas(), how="inner", on="reference")

    # exit if mappings has no rows
    if mappings.shape[0] == 0:
        return None, None
    else:
        mappings_agg_dm = mappings.group_by("group").agg(
            {
                "depth_mean": ["mean", "std", "median", "sum", "count"],
            }
        )
        mappings_agg_dm = mappings_agg_dm.xs(
            "depth_mean",
            axis=1,
            drop_level=True,
        )
        mappings_agg_dm = mappings_agg_dm.reset_index("group")
        mappings_agg_dm.rename({"count": "n_genes"}, axis=1, inplace=True)

        mappings_agg_al = mappings.group_by("group").agg(
            {
                "avg_read_length": ["mean"],
            }
        )
        mappings_agg_al = mappings_agg_al.xs(
            "avg_read_length",
            axis=1,
            drop_level=True,
        )
        mappings_agg_al = mappings_agg_al.reset_index("group")
        mappings_agg_al.rename({"mean": "avg_read_length"}, axis=1, inplace=True)

        mappings_agg_sl = mappings.group_by("group").agg(
            {
                "stdev_read_length": ["mean"],
            }
        )
        mappings_agg_sl = mappings_agg_sl.xs(
            "stdev_read_length",
            axis=1,
            drop_level=True,
        )
        mappings_agg_sl = mappings_agg_sl.reset_index("group")
        mappings_agg_sl.rename({"mean": "stdev_read_length"}, axis=1, inplace=True)

        # Get average identities
        mappings_agg_ai = mappings.group_by("group").agg(
            {
                "avg_identity": ["mean"],
            }
        )
        mappings_agg_ai = mappings_agg_ai.xs(
            "avg_identity",
            axis=1,
            drop_level=True,
        )
        mappings_agg_ai = mappings_agg_ai.reset_index("group")
        mappings_agg_ai.rename({"mean": "avg_identity"}, axis=1, inplace=True)

        mappings_agg_si = mappings.group_by("group").agg(
            {
                "stdev_identity": ["mean"],
            }
        )
        mappings_agg_si = mappings_agg_si.xs(
            "stdev_identity",
            axis=1,
            drop_level=True,
        )
        mappings_agg_si = mappings_agg_si.reset_index("group")
        mappings_agg_si.rename({"mean": "stdev_identity"}, axis=1, inplace=True)

        mappings_agg = reduce(
            lambda left, right: pd.merge(left, right, on=["group"], how="inner"),
            [
                mappings_agg_dm,
                mappings_agg_al,
                mappings_agg_sl,
                mappings_agg_ai,
                mappings_agg_si,
            ],
        )
        mappings_agg.columns = [
            "group",
            "coverage_mean",
            "coverage_stdev",
            "coverage_median",
            "coverage_sum",
            "n_genes",
            "avg_read_length",
            "stdev_read_length",
            "avg_identity",
            "stdev_identity",
        ]
        return mappings, mappings_agg


def convert_to_anvio(df, annotation_source):
    """
    Converts a pandas dataframe to an Anvio-compatible format.

    Args:
        df (pandas.DataFrame): The input dataframe to convert.
        annotation_source (str): The source of the annotation.

    Returns:
        pandas.DataFrame: The converted dataframe.
    """
    # gene_id	enzyme_accession	source	coverage	detection
    # Select and rename columns from pandas dataframe
    df = df.copy()
    df["group"] = df["group"].str.replace("ko:", "")
    df["source"] = annotation_source
    df = df[["reference", "group", "source", "depth_mean", "breadth"]]
    df.rename(
        columns={
            "reference": "gene_id",
            "group": "enzyme_accession",
            "depth_mean": "coverage",
            "breadth": "detection",
        },
        inplace=True,
    )
    return df
