import numpy as np
import sys
from multiprocessing import Pool
from scipy import stats
import tqdm
import logging
from x_filter.utils import (
    is_debug,
    calc_chunksize,
)
import datatable as dt
import pyranges as pr
from multiprocessing import Pool
from functools import partial
from collections import defaultdict


log = logging.getLogger("my_logger")

sys.setrecursionlimit(10 ** 6)


def get_aln_stats_worker(what, parms):
    # return list(set(parms.df[what].to_list()[0]))
    return list(parms.df[what].to_pandas()[what].unique())


def get_aln_stats(ns, threads=1):
    lst = ["queryId", "subjectId"]

    func = partial(get_aln_stats_worker, parms=ns)

    p = Pool(threads, initializer=initializer, initargs=(ns,))
    ret_list = list(
        tqdm.tqdm(
            p.imap(func, lst),
            total=len(lst),
            leave=False,
            ncols=100,
            desc=f"Getting basic stats",
        )
    )
    p.close()
    p.join()

    nqueries = ret_list[0]
    nrefs = ret_list[1]
    return nqueries, nrefs


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
    nalns = aln.shape[0]
    # logging.info(f"Read {nalns:,} alignments. Getting basic statistics.")

    logging.info(f"Pre-filtering: {nalns:,} alignments found")
    logging.info(
        f"Filtering alignments with bitscore >= {bitscore} and evalue <= {evalue}"
    )
    aln = aln[(dt.f.eVal < evalue) & (dt.f.bitScore > bitscore), :]

    nalns = aln.shape[0]

    logging.info(f"Post-filtering: {nalns:,} alignments found")
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
    logging.info(f"::: Iter: {iter + 1} - Total alignment: {n_alns}")

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
    logging.info(f"::: Iter: {iter} - Removed {to_remove.shape[0]} alignments")
    logging.info(f"::: Iter: {iter} - New unique mapping queries: {n_unique}")
    logging.info(f"::: Iter: {iter} - Total unique mapping queries: {total_n_unique}")
    logging.info(f"::: Iter: {iter} - Alns left: {str(df.shape[0])}")

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


def get_stats_coverage(x, gen_data, trim=False, strim_5=18, strim_3=18):
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
        )


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


mgr = None


def initializer_1(x):
    global mgr
    mgr = x


def apply_parallel(data, func, p, threads):
    func = partial(func)
    if is_debug():
        ret_list = list(map(func, data))
    else:
        if len((data)) > 1000:
            c_size = calc_chunksize(threads, len((data)), factor=4)
        else:
            c_size = 1

        ret_list = list(
            tqdm.tqdm(
                p.imap_unordered(func, data, chunksize=c_size),
                total=len((data)),
                leave=False,
                ncols=100,
                desc=f"References processed",
            )
        )
        p.close()
        p.join()
    print(dt.rbind(ret_list))
    return dt.rbind(ret_list)


def get_stats(data):
    data = pr.PyRanges(data)
    refs = data.df.head(1)[["Chromosome", "len"]]
    refs["Start"] = 1
    refs = refs.rename(
        columns={
            "len": "End",
        }
    )
    # # refs["End"] = refs["End"] + 1
    refs = pr.PyRanges(refs)
    # alns = pr.PyRanges(data)
    mean_cov = cov_stats(data, refs)
    b_cov = refs.coverage(
        data, overlap_col="n_reads", fraction_col="breadth", nb_cpu=1
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
    b_cov["End"] = b_cov["End"]

    return dt.Frame(b_cov.rename(columns={"End": "length", "Chromosome": "reference"}))


def get_stats_coverage(x, gen_data, trim=False, strim_5=18, strim_3=18):
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
        )


def get_coverage_stats(df, trim=True):

    queries = defaultdict(int)
    gen_data = defaultdict(dict)
    rl = defaultdict(list)
    for a, q, b, c, d, e in tqdm.tqdm(
        zip(
            df["Chromosome"],
            df["Query"],
            df["Start"],
            df["End"],
            df["slen"],
            df["qlen"],
        ),
        total=df.shape[0],
        leave=False,
        ncols=100,
        desc=f"Alignments processed",
    ):
        b = b - 1

        if a in gen_data:
            gen_data[a]["cov"][b:c] += 1
            gen_data[a]["n_alns"] += 1
            rl[a] += [e / 3]
        else:
            gen_data[a]["cov"] = np.zeros(d, dtype=int)
            gen_data[a]["cov"][b:c] += 1
            gen_data[a]["n_alns"] = 0
            gen_data[a]["n_alns"] += 1
            rl[a] += [e / 3]
    logging.info(
        f"References will be dinamycally trimmed at 5'/3'-ends (half of the avg. read length)"
    )
    stats = [
        get_stats_coverage(
            chrom,
            gen_data=gen_data,
            strim_5=int(np.mean(rl[chrom]) / 2),
            strim_3=int(np.mean(rl[chrom]) / 2),
            trim=trim,
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
        return None
    else:
        mappings = mappings.groupby("group").agg(
            {"depth_mean": ["mean", "std", "median", "sum", "count"]}
        )
        mappings = mappings.xs("depth_mean", axis=1, drop_level=True)
        mappings = mappings.reset_index("group")
        mappings.rename({"count": "n_genes"}, axis=1, inplace=True)
        return mappings
