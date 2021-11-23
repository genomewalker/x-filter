import numpy as np
import re, os, sys
import pandas as pd
from multiprocessing import Pool
import functools
from scipy import stats
import tqdm
import logging
import warnings
from x_filter.utils import (
    is_debug,
    calc_chunksize,
    fast_flatten,
    initializer,
    apply_parallel,
)
import datatable as dt
import pyranges as pr

log = logging.getLogger("my_logger")

sys.setrecursionlimit(10 ** 6)


def read_and_filter_alns(
    aln,
    bitscore=60,
    evalue=1e-10,
    col_names=[
        "queryId",
        "subjectId",
        "percIdentity",
        "alnLength",
        "mismatchCount",
        "gapOpenCount",
        "queryStart",
        "queryEnd",
        "subjectStart",
        "subjectEnd",
        "eVal",
        "bitScore",
        "qlen",
        "slen",
    ],
    threads=1,
):
    """Read and filter alignments.

    Args:
        aln (str): A path to the file to be read
        bitscore (int): Bitscore value to filter the alignments
        evalue (float): E-value to filter the alignments
        col_names (list): List with the column names to use

    Returns:
        [dataFrame]: A dataframe with the filtered alignments
    """
    dt.options.progress.clear_on_success = True
    dt.options.nthreads = threads
    aln = dt.fread(
        aln,
        sep="\t",
        header=False,
        nthreads=32,
        columns=col_names[0],
    )
    aln[dt.bool8] = dt.int32

    dt.options.nthreads = 2
    nqueries = aln["queryId"].nunique1()
    nrefs = aln["subjectId"].nunique1()
    nalns = aln.shape[0]

    logging.info(
        f"Pre-filtering: {nalns:,} alignments found with {nqueries:,} queries and {nrefs:,} references"
    )

    dt.options.nthreads = threads
    aln = aln[(dt.f.eVal < evalue) & (dt.f.bitScore > bitscore), :]

    dt.options.nthreads = 2
    nqueries = aln["queryId"].nunique1()
    nrefs = aln["subjectId"].nunique1()
    nalns = aln.shape[0]

    logging.info(
        f"Post-filtering: {nalns:,} alignments found with {nqueries:,} queries and {nrefs:,} references"
    )
    return aln


def resolve_multimaps(
    df,
    scale=0.9,
    iters=10,
    threads=1,
):

    """Resolve multimaps by iteratively removing the lowest scoring reads.

    Args:
        df ([dataFrame]): [description]
        scale (float, optional): [description]. Defaults to 0.9.
        iters (int, optional): [description]. Defaults to 10.
        threa
    """
    dt.options.progress.clear_on_success = True
    dt.options.nthreads = threads

    if "iter" in list(df.names):
        iter = df["iter"].to_list()
        iter = iter[0][0]
    else:
        iter = 0

    #       col_names <- c("label", "query", "theader", "pident", "alnlen", "mismatch", "gapopen", "qstart", "qend", "tstart", "tend", "evalue", "bits", "qlen", "tlen")

    logging.info(f"\tIter: {iter + 1} - Calculating query bitscores")

    # Calculate bit scores per queryID
    df[
        :,
        dt.update(W=dt.f.bitScore / dt.sum(dt.f.bitScore), n_aln=dt.count()),
        dt.by("queryId"),
    ]

    logging.info(f"\tIter: {iter + 1} - Calculating reference bitscores")

    tot_bits = df[
        :,
        {"tot_bits": dt.sum(dt.f.bitScore), "n_aln_prot": dt.count()},
        dt.by("subjectId"),
    ]

    logging.info(f"\tIter: {iter + 1} - Calculating alignment likelihoods")
    tot_bits.key = "subjectId"
    df = df[dt.f.n_aln > 1, :]
    df = df[:, :, dt.join(tot_bits)]
    df[:, dt.update(L=dt.f.W * dt.f.tot_bits)]
    df[
        :,
        dt.update(
            max_L=scale * dt.max(dt.f.L),
        ),
        dt.by("queryId"),
    ]
    df = df[dt.f.L >= dt.f.max_L, :]

    keep_processing_n = df[:, {"refined_n": dt.count()}, dt.by("queryId")]
    logging.info(
        f"\tIter: {iter + 1} - Queries with multi-mappings left: {str(keep_processing_n[dt.f.refined_n > 1, :].shape[0])}"
    )

    iter = iter + 1
    df[:, dt.update(iter=iter)]
    left_alns = keep_processing_n[dt.f.refined_n <= 1, :].shape[0]
    keep_processing = left_alns == 0
    logging.info(f"\tIter: {iter} - done!")
    if (not keep_processing) & (iter < iters):
        df1 = df
        df1 = resolve_multimaps(df1, iters=iters)
    else:
        df1 = df
        return df1

    return df1


def cov_stats(alns, refs):

    rle = alns.to_rle(nb_cpu=1)

    g3 = rle[refs]

    df = g3.df

    df.insert(df.shape[1], "Sum", df.Value * df.Run)
    g = df.groupby(["Start", "End"])
    mean = g.Sum.sum() / g.Run.sum()
    length = g.Sum.count()

    means = np.repeat(mean, length).values
    df.insert(df.shape[1], "depth_mean", means)
    df.insert(df.shape[1], "Intermediate", (((df.Value - df.depth_mean) ** 2) * df.Run))

    g = df.groupby(["Start", "End"])

    sqrt = np.sqrt(g.Intermediate.sum() / (g.Run.sum() - 1))

    df = df.drop_duplicates("Start End".split())
    df.insert(df.shape[1], "depth_sd", sqrt.values)

    df.insert(df.shape[1], "depth_evenness", df["depth_sd"] / df["depth_mean"])

    return df.drop("Run Value Sum Intermediate".split(), axis=1)


def get_stats(df):
    refs = df.head(1)[["Chromosome", "len"]]
    refs["Start"] = 1
    refs = refs.rename(
        columns={
            "len": "End",
        }
    )

    refs = pr.PyRanges(refs)
    alns = pr.PyRanges(df)
    mean_cov = cov_stats(alns, refs)
    b_cov = refs.coverage(
        alns, overlap_col="n_reads", fraction_col="breadth", nb_cpu=1
    ).df

    b_cov = b_cov.merge(
        mean_cov[["Chromosome", "depth_mean", "depth_sd", "depth_evenness"]],
        how="inner",
        on="Chromosome",
    )
    b_cov.insert(b_cov.shape[1], "exp_breadth", 1 - np.exp(-b_cov["depth_mean"]))
    b_cov.insert(
        b_cov.shape[1], "breadth_exp_ratio", b_cov["breadth"] / b_cov["exp_breadth"]
    )
    b_cov.drop(["Start"], axis=1, inplace=True)

    return b_cov.rename(columns={"End": "length", "Chromosome": "reference"})
