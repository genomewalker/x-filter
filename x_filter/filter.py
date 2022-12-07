import numpy as np
import sys
import tqdm
import logging
import datatable as dt
import gc
import psutil
import tempfile
from collections import defaultdict
import os
from functools import reduce, partial
import pandas as pd
import gzip
from mimetypes import guess_type
from io import SEEK_END
import os

import zlib
import struct

log = logging.getLogger("my_logger")

sys.setrecursionlimit(10**6)

# estimate file size of a gzip compressed file

# From https://stackoverflow.com/a/68939759
def estimate_uncompressed_gz_size(filename):
    # From the input file, get some data:
    # - the 32 LSB from the gzip stream
    # - 1MB sample of compressed data
    # - compressed file size
    with open(filename, "rb") as gz_in:
        sample = gz_in.read(1000000)
        gz_in.seek(-4, SEEK_END)
        lsb = struct.unpack("I", gz_in.read(4))[0]
        file_size = os.fstat(gz_in.fileno()).st_size
    # Estimate the total size by decompressing the sample to get the
    # compression ratio so we can extrapolate the uncompressed size
    # using the compression ratio and the real file size
    dobj = zlib.decompressobj(31)
    d_sample = dobj.decompress(sample)

    compressed_len = len(sample) - len(dobj.unconsumed_tail)
    decompressed_len = len(d_sample)

    estimate = file_size * decompressed_len // compressed_len
    # 32 LSB to zero
    mask = ~0xFFFFFFFF

    # Kill the 32 LSB to be substituted by the data read from the file
    adjusted_estimate = (estimate & mask) | lsb

    return adjusted_estimate


# from https://softwareengineering.stackexchange.com/a/419985
def line_estimation(filename, first_size=1 << 24):
    encoding = guess_type(filename)[1]
    _open = partial(gzip.open, mode="rb") if encoding == "gzip" else open
    if encoding == "gzip":
        with _open(filename) as file:
            buf = file.read(first_size)
            if buf.count(b"\n") == 0:
                log.info("No lines found in file to process. Exiting...")
                exit(0)
            else:
                return len(buf) // buf.count(b"\n")
    else:
        with _open(filename) as file:
            buf = file.read(first_size)
            if buf.count("\n") == 0:
                log.info("No lines found in file to process. Exiting...")
                exit(0)
            else:
                return len(buf) // buf.count("\n")


# def get_multi_keys(aln_rows, key_col):
#     keys = aln_rows[0]
#     aln_rows.pop(0)
#     if len(aln_rows) > 0:
#         for k in aln_rows:
#             keys.rbind(k)
#     keys = keys[
#         :,
#         {"n": dt.count()},
#         dt.by(
#             key_col,
#         ),
#     ]
#     # keys_multi = keys[dt.f.n > 1, key_col]
#     # keys_nomulti = keys[dt.f.n == 1, key_col]
#     # keys_nomulti.key = key_col
#     keys[:, dt.update(is_multi=dt.ifelse(dt.f.n > 1, "multi", "nomulti"))]
#     keys.key = key_col
#     return keys


def filter_eval_perc(alns, evalues, perc=0.1):
    logging.info(f"Removing alignments within {int(perc * 100)}% of the best e-value")
    evalues[:, "maxlogEvalue"] = -1 * perc * (np.log(evalues[:, "eVal"]) / np.log(10))
    evalues.key = "subjectId"
    aln_sizes = []
    for i in np.arange(len(alns)):
        alns[i] = alns[i][:, :, dt.join(evalues)]
        alns[i][:, "logEvalue"] = -1 * ((np.log(alns[i][:, "eVal"])) / np.log(10))
        aln_sizes.append(alns[i][(dt.f.logEvalue > dt.f.maxlogEvalue), :].shape[0])
        del alns[i]["maxlogEvalue"]
        del alns[i]["logEvalue"]
    return aln_sizes


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
    evalue_perc=None,
    evalue_perc_step=0.1,
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

    # We do a raw estimation of the number of alignments in a file. This is
    # because we don't know how many alignments are in the file and they can be in
    # in the order of billions.

    logging.info("Getting a raw estimate of the number of alignments...")

    if guess_type(aln)[1] == "gzip":
        fsize = estimate_uncompressed_gz_size(aln)
    else:
        fsize = os.path.getsize(aln)

    nalns = fsize // line_estimation(
        filename=aln, first_size=int(os.path.getsize(aln) * 0.01)
    )
    logging.info(f"Approximately {nalns:,} alignments found.")

    max_rows = int((2**31) - 1)
    # max_rows = int(5e6)
    # if the number of alignmentsis larger than the number of rows
    # that can be read by fread, we need to read it by chunks instead

    if nalns > max_rows:
        gc.collect()
        logging.info(
            f"Pre-filtering: {nalns:,} alignments found. This is more than {max_rows:,} alignments."
        )
        logging.info(
            f"::: This is not supported at the moment. Trying to read and filter using chunks instead."
        )

        k, m = divmod(nalns, max_rows)
        evalues = []
        alns = []
        for i in range(k + 1):
            logging.info(f"::: Reading chunk #{i+1:,}.")
            up = max_rows * (i + 1)
            if up > nalns:
                up = nalns
                aln_chunk = dt.fread(
                    aln,
                    sep="\t",
                    header=False,
                    nthreads=threads,
                    columns=col_names[0],
                    skip_to_line=int(max_rows * i),
                )
            else:
                up = int(up - max_rows * i)
                aln_chunk = dt.fread(
                    aln,
                    sep="\t",
                    header=False,
                    nthreads=threads,
                    columns=col_names[0],
                    skip_to_line=int(max_rows * i),
                    max_nrows=up,
                )
            logging.info(
                f"::: Filtering alignments in chunk #{i+1:,} with bitscore >= {bitscore} and evalue <= {evalue}"
            )
            aln_chunk = aln_chunk[(dt.f.eVal < evalue) & (dt.f.bitScore > bitscore), :]
            logging.info(f"Getting evalues")
            evals = aln_chunk[:1, ["eVal"], dt.by(dt.f.subjectId), dt.sort(dt.f.eVal)]
            # evals[:, "maxlogEvalue"] = (
            #     -1 * evalue_perc * (np.log(evals[:, "eVal"]) / np.log(10))
            # )
            logging.info(f"Deep copying resulting frames")
            evalues.append(evals.copy(deep=True))
            alns.append(aln_chunk.copy(deep=True))
            del aln_chunk
            del evals
            gc.collect(generation=2)

        evalues = dt.rbind(*evalues)
        evalues = evalues[:1, ["eVal"], dt.by(dt.f.subjectId), dt.sort(dt.f.eVal)]
        # evalues.key = "subjectId"
        # TODO: do a binary search to speed up the process
        if evalue_perc is None:
            logging.info("Trying to find best filtering threshold to fit data")
            for perc in np.arange(0, 1, evalue_perc_step)[1:]:
                # logging.info(
                #     f"Removing alignments within {int(perc * 100)}% of the best e-value"
                # )
                # evalues[:, "maxlogEvalue"] = (
                #     -1 * perc * (np.log(evalues[:, "eVal"]) / np.log(10))
                # )
                # evalues.key = "subjectId"
                # aln_sizes = []
                # for i in np.arange(len(alns)):
                #     alns[i] = alns[i][:, :, dt.join(evalues)]
                #     alns[i][:, "logEvalue"] = -1 * (
                #         (np.log(alns[i][:, "eVal"])) / np.log(10)
                #     )
                #     aln_sizes.append(
                #         alns[i][(dt.f.logEvalue > dt.f.maxlogEvalue), :].shape[0]
                #     )
                #     del alns[i]["maxlogEvalue"]
                #     del alns[i]["logEvalue"]
                aln_sizes = filter_eval_perc(alns=alns, evalues=evalues, perc=perc)
                if np.sum(aln_sizes) < max_rows:
                    for i in np.arange(len(alns)):
                        alns[i] = alns[i][:, :, dt.join(evalues)]
                        alns[i][:, "logEvalue"] = -1 * (
                            (np.log(alns[i][:, "eVal"])) / np.log(10)
                        )

                        alns[i] = alns[i][(dt.f.logEvalue > dt.f.maxlogEvalue), :]

                        del alns[i]["maxlogEvalue"]
                        del alns[i]["logEvalue"]
                        logging.info(f"Filter {perc} produced a manageable table.")
                    break
                else:
                    logging.info(
                        f"{np.sum(aln_sizes):,} alignments found. Increasing filtering threshold"
                    )
        else:
            aln_sizes = filter_eval_perc(alns=alns, evalues=evalues, perc=evalue_perc)
            if np.sum(aln_sizes) < max_rows:
                for i in np.arange(len(alns)):
                    alns[i] = alns[i][:, :, dt.join(evalues)]
                    alns[i][:, "logEvalue"] = -1 * (
                        (np.log(alns[i][:, "eVal"])) / np.log(10)
                    )

                    alns[i] = alns[i][(dt.f.logEvalue > dt.f.maxlogEvalue), :]

                    del alns[i]["maxlogEvalue"]
                    del alns[i]["logEvalue"]
                logging.info(f"Filter {evalue_perc} produced a manageable table.")

        if np.sum(aln_sizes) > max_rows:
            logging.error(
                f"The resulting table has more than {max_rows:,} alignments and it is not supported at the moment."
            )
            exit(1)
        logging.info(f"::: Concatenating chunks.")
        aln = alns[0]
        alns.pop(0)

        if len(alns) > 0:
            for i in alns:
                aln.rbind(i)
    else:
        logging.info(f"Pre-filtering: {nalns:,} alignments estimated")
        # logging.info(f"Read {nalns:,} alignments. Getting basic statistics.")

        logging.info(
            f"Filtering alignments with bitscore >= {bitscore} and evalue <= {evalue}"
        )
        aln = dt.fread(
            aln,
            sep="\t",
            header=False,
            nthreads=threads,
            columns=col_names[0],
        )
        aln = aln[(dt.f.eVal < evalue) & (dt.f.bitScore > bitscore), :]

    nalns = aln.shape[0]
    logging.info(f"Post-filtering: {nalns:,} alignments found")

    logging.info(f"Removing reads mapping multiple times to the same subject")
    aln = aln[:1, :, dt.by(dt.f.queryId, dt.f.subjectId), dt.sort(-dt.f.bitScore)]
    nalns = aln.shape[0]
    logging.info(f"::: Kept best bitScore alignments: {nalns:,}")
    return aln


def initialize_subject_weights(df):
    # df[:, dt.update(n_aln=dt.count()), dt.by("queryId")]
    # df_unique = df[dt.f.n_aln <= 1, :]
    # df = df[dt.f.n_aln > 1, :]
    if df.shape[0] > 0:
        # df[:, dt.update(weight=dt.f.n_aln / df.n_aln.sum()), dt.by("queryId")]
        df[:, dt.update(s_W=1 / dt.f.slen)]
        df[
            :,
            dt.update(prob=dt.f.bitScore / dt.sum(dt.f.bitScore)),
            dt.by("queryId"),
        ]
        # df = dt.rbind(df, df_unique, force=True)
        # del df["n_aln"]
        return df
    else:
        return None


def resolve_multimaps(
    df,
    scale=0.9,
    iters=10,
    threads=1,
):

    """Resolve multimaps by iteratively removing the lowest scoring reads.

    Args:
        df ([Frame]): A Frame containing the alignments
        scale (float, optional): Scale where to filter. Defaults to 0.9.
        iters (int, optional): Number of iterations. Defaults to 10.
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

    logging.info(f"::: Iter: {iter + 1} - Getting scores")

    n_alns = df.shape[0]
    logging.info(f"::: Iter: {iter + 1} - Total alignment: {n_alns:,}")

    # Calculate the weights for each subject
    df[:, dt.update(s_W=dt.sum(dt.f.prob) / dt.f.slen), dt.by("subjectId")]

    # Calculate the alignment probabilities
    df[:, dt.update(new_prob=dt.f.prob * dt.f.s_W), dt.by("queryId")]
    df[
        :,
        dt.update(prob_sum=dt.sum(dt.f.prob * dt.f.s_W)),
        dt.by("queryId"),
    ]
    df[
        :,
        dt.update(prob=dt.f.new_prob / dt.f.prob_sum),
    ]
    # print(
    #     df[
    #         (dt.f.subjectId == "nca:Noca_4326")
    #         & (dt.f.queryId == "11348bc767_000000480528__1"),
    #         :,
    #     ]
    # )
    del df["new_prob"]
    del df["prob_sum"]

    # Calculate how many alignments are in each query
    df[:, dt.update(n_aln=dt.count()), dt.by("queryId")]
    df_unique = df[dt.f.n_aln <= 1, :]
    df = df[(dt.f.n_aln > 1) & (dt.f.prob > 0), :]

    # Keep the ones that that have a probability higher than the maximum scaled probability

    # Get maximum scaled probability for each query
    df[
        :,
        dt.update(
            max_prob=dt.max(dt.f.prob),
        ),
        dt.by("queryId"),
    ]

    df[
        :,
        dt.update(
            max_prob_scaled=dt.max(dt.f.prob) * scale,
        ),
        dt.by("queryId"),
    ]

    to_remove = df[dt.f.prob < (dt.f.max_prob_scaled), :]

    df = df[dt.f.prob >= dt.f.max_prob_scaled, :]

    del df["max_prob"]
    del df["max_prob_scaled"]

    n_unique = df[:, {"n_aln": dt.count()}, dt.by("queryId")]
    n_unique = n_unique[dt.f.n_aln <= 1, :].shape[0]

    iter = iter + 1

    # Add the iteration
    df[:, dt.update(iter=iter)]
    df_unique[:, dt.update(iter=iter)]

    # Combine unique mapping and the ones that need to be re-evaluated
    df = dt.rbind(df, df_unique)
    total_n_unique = df[:, {"n_aln": dt.count()}, dt.by("queryId")]
    total_n_unique = total_n_unique[dt.f.n_aln <= 1, :].shape[0]
    keep_processing = to_remove.shape[0] != 0
    logging.info(f"::: Iter: {iter} - Removed {to_remove.shape[0]:,} alignments")
    logging.info(f"::: Iter: {iter} - New unique mapping queries: {n_unique:,}")
    logging.info(f"::: Iter: {iter} - Total unique mapping queries: {total_n_unique:,}")
    logging.info(f"::: Iter: {iter} - Alns left: {df.shape[0]:,}")

    logging.info(f"::: Iter: {iter} - done!")
    if iter < iters and keep_processing:
        df1 = df[
            :,
            ["queryId", "subjectId", "bitScore", "slen", "s_W", "prob", "iter"],
        ]
        df1 = resolve_multimaps(df1, iters=iters, scale=scale)
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


def get_stats_coverage(x, gen_data, rl, trim=False, strim_5=18, strim_3=18):
    """Get coverage statistics

    Args:
        x (list): A list with the coverage per position
        strim_5 (int, optional): How much to trim on the 5'. Defaults to 18.
        strim_3 (int, optional): How much to trim on the 3'. Defaults to 18.

    Returns:
        [type]: [description]
    """
    cov = gen_data[x]["cov"]
    length = cov.shape[0]
    # If we need to trim
    if trim:
        long_enough = cov.shape[0] >= strim_5 + strim_3 + 10
        if long_enough:
            cov = cov[strim_5:-strim_3]

    if gen_data[x]["n_alns"] > 1:
        rl_mean = np.mean(rl)
        rl_std = np.std(rl, ddof=1)
    else:
        rl_mean = rl[0]
        rl_std = 0

    cov_mean = cov.mean()
    if cov_mean > 0:
        cov_std = cov.std()
        cov_evenness = cov_std / cov_mean
        breadth = ((cov != 0).sum()) / length
        breadth_exp = 1 - np.exp(-cov_mean)
        if breadth >= breadth_exp:
            breadth_exp_ratio = 1.0 * breadth
        else:
            breadth_exp_ratio = (breadth / breadth_exp) * breadth
        return (
            x,
            cov_mean,
            cov_std,
            cov_evenness,
            breadth,
            breadth_exp,
            breadth_exp_ratio,
            gen_data[x]["n_alns"],
            rl_mean,
            rl_std,
        )


def get_coverage_stats(df, trim=True):

    queries = defaultdict(int)
    gen_data = defaultdict(dict)
    al = defaultdict(list)
    rl = defaultdict(list)
    # for a, q, b, c, d, e in tqdm.tqdm(
    #     zip(
    #         df["Chromosome"],
    #         df["Query"],
    #         df["Start"],
    #         df["End"],
    #         df["slen"],
    #         df["qlen"],
    #     ),
    #     total=df.shape[0],
    #     leave=False,
    #     ncols=100,
    #     desc=f"Alignments processed",
    # ):
    for i in tqdm.tqdm(
        np.arange(df.shape[0]),
        total=df.shape[0],
        leave=False,
        ncols=100,
        desc=f"Alignments processed",
    ):
        i = int(i)
        a = df[i, "subjectId"]
        q = df[i, "queryId"]
        b = df[i, "subjectStart"]
        c = df[i, "subjectEnd"]
        d = df[i, "slen"]
        e = df[i, "alnLength"]
        f = df[i, "qlen"]
        b = b - 1

        if a in gen_data:
            gen_data[a]["cov"][b:c] += 1
            gen_data[a]["n_alns"] += 1
            al[a] += [e / 3]
            rl[a] += [f]
        else:
            gen_data[a]["cov"] = np.zeros(d, dtype=int)
            gen_data[a]["cov"][b:c] += 1
            gen_data[a]["n_alns"] = 0
            gen_data[a]["n_alns"] += 1
            al[a] += [e / 3]
            rl[a] += [f]
    logging.info(
        f"References will be dynamically trimmed at 5'/3'-ends (half of the avg. aln length)"
    )
    stats = [
        get_stats_coverage(
            chrom,
            gen_data=gen_data,
            strim_5=int(np.mean(al[chrom]) / 2),
            strim_3=int(np.mean(al[chrom]) / 2),
            trim=trim,
            rl=rl[chrom],
        )
        for chrom in tqdm.tqdm(
            gen_data,
            total=len(gen_data.keys()),
            leave=False,
            ncols=100,
            desc=f"References processed",
        )
    ]
    stats = [x for x in stats if x is not None]
    stats = dt.Frame(
        stats,
    )

    stats.names = [
        "reference",
        "depth_mean",
        "depth_std",
        "depth_evenness",
        "breadth",
        "breadth_expected",
        "breadth_expected_ratio",
        "n_alns",
        "avg_read_length",
        "stdev_read_length",
    ]
    return stats


def aggregate_gene_abundances(mapping_file, gene_abundances, threads=1):
    dt.options.progress.clear_on_success = True
    dt.options.nthreads = threads
    """Aggregate gene abundances
    
        Args:
            mapping_file (str): Path to the mapping file
            gene_abundances (Frame): Path to the gene abundances file
    
        Returns:
            dt.Frame: The aggregated gene abundances
    """
    mappings = dt.fread(
        mapping_file, sep="\t", columns=["reference", "group"]
    ).to_pandas()
    mappings = mappings.merge(gene_abundances.to_pandas(), how="inner", on="reference")

    # exit if mappings has no rows
    if mappings.shape[0] == 0:
        return None, None
    else:
        mappings_agg_dm = mappings.groupby("group").agg(
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

        mappings_agg_al = mappings.groupby("group").agg(
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

        mappings_agg_sl = mappings.groupby("group").agg(
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

        mappings_agg = reduce(
            lambda left, right: pd.merge(left, right, on=["group"], how="inner"),
            [mappings_agg_dm, mappings_agg_al, mappings_agg_sl],
        )
        return mappings, mappings_agg


def convert_to_anvio(df, annotation_source):
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
