from typing import Dict, List, Tuple, Any
import logging
import duckdb
import pandas as pd
import numpy as np

from x_filter.ops import setup_temporary_directory, process_input_data
from x_filter.utils import get_arguments, apply_filters, create_output_files
from x_filter.stats import calculate_statistics
from x_filter.reassign import reassign
from x_filter.slice_mmap_arrays import parallel_slice_mmap
from x_filter.aggregate import aggregate_gene_abundances, convert_to_anvio

# Set up logging
log = logging.getLogger("my_logger")
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def setup_logging(debug_mode: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(levelname)s ::: %(asctime)s ::: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def process_data(
    args: Any, filters: List[Dict[str, Any]], tmp_dir: str, tmp_files: Dict[str, str]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray], str]:
    np_arrays, parquet_file = process_input_data(
        args.input,
        (tmp_dir, tmp_files),
        num_threads=args.threads,
        evalue_threshold=args.evalue,
        bitscore_threshold=args.bitscore,
        max_memory=args.max_memory,
        chunk_size=10_000_000,
    )

    log.info("Getting coverage statistics")
    final_stats, unique_subjects, inverse_indices, numpy_arrays = calculate_statistics(
        np_arrays, tmp_files, num_threads=args.threads
    )
    # sort by breadth descending
    final_stats = final_stats.sort_values("breadth", ascending=True)

    if filters:
        log.info("Applying filters")
        final_stats = apply_filters(final_stats, filters)
        # final_stats = final_stats[final_stats["breadth"] >= 0.25]
        final_stats = final_stats.sort_values(
            ["breadth", "subject_numeric_id"], ascending=True
        )

    return final_stats, unique_subjects, inverse_indices, numpy_arrays, parquet_file


def filter_arrays(
    final_stats: pd.DataFrame,
    unique_subjects: np.ndarray,
    inverse_indices: np.ndarray,
    numpy_arrays: Dict[str, np.ndarray],
    tmp_files: Dict[str, str],
    args: Any,
) -> pd.DataFrame:
    target_subjects = final_stats["subject_numeric_id"].to_numpy()

    if target_subjects.size > 0:
        target_indices = np.where(np.isin(unique_subjects, target_subjects))[0]
        if len(target_indices) == 0:
            raise ValueError(
                "None of the target subjects were found in unique_subjects."
            )
        filtered_arrays = parallel_slice_mmap(
            numpy_arrays,
            target_subjects,
            unique_subjects,
            inverse_indices,
            mmap_folder=tmp_files["mmap"],
            num_threads=args.threads,
        )

    log.info("Resolve multimappings...")
    return reassign(filtered_arrays, tmp_files)


def process_filtered_data(
    filtered_ids_df: pd.DataFrame,
    parquet_file: str,
    tmp_files: Dict[str, str],
    args: Any,
) -> pd.DataFrame:
    parquet_dir = tmp_files["parquet"]
    filtered_parquet = f"{parquet_dir}/filtered_query_subject_ids.parquet"
    filtered_parquet_results = f"{parquet_dir}/filtered_blast_results.parquet"
    filtered_ids_df.to_parquet(filtered_parquet, compression="snappy")

    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT blast_results.*
            FROM read_parquet('{parquet_file}') AS blast_results
            INNER JOIN read_parquet('{filtered_parquet}') AS filtered_ids
            ON blast_results.query_numeric_id = filtered_ids.query_numeric_id
            AND blast_results.row_hash = filtered_ids.row_hash
        ) TO '{filtered_parquet_results}'
        (FORMAT PARQUET, CODEC 'SNAPPY', ROW_GROUP_SIZE 1000000);
        """
    )

    return pd.read_parquet(filtered_parquet_results)


def analyze_alignments(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    query_counts = df["query_numeric_id"].value_counts()
    single_alignment_queries = query_counts[query_counts == 1].index
    multi_alignment_queries = query_counts[query_counts > 1].index

    log.info(f"Number of unique reads: {len(query_counts):,}")
    log.info(f"Number of single alignment: {len(single_alignment_queries):,}")
    log.info(f"Number of multi-alignment: {len(multi_alignment_queries):,}")

    return {
        "subject_numeric_id": df["subject_numeric_id"].to_numpy(),
        "subjectStart": df["subjectStart"].to_numpy(),
        "subjectEnd": df["subjectEnd"].to_numpy(),
        "alnLength": df["alnLength"].to_numpy(),
        "qlen": df["qlen"].to_numpy(),
        "percIdentity": df["percIdentity"].to_numpy(),
        "slen": df["slen"].to_numpy(),
    }


def save_results(
    final_stats: pd.DataFrame,
    df: pd.DataFrame,
    out_files: Dict[str, str],
    mapping_file: str,
    anvio: bool,
    annotation_source: str,
    tmp_files: Dict[str, str],
    threads: int = 1,
) -> None:
    unique_subjects = df[["subjectId", "subject_numeric_id"]].drop_duplicates()
    final_stats = final_stats.merge(
        unique_subjects, on="subject_numeric_id", how="left"
    ).drop(columns=["subject_numeric_id"])

    column_mapping = {
        "subjectId": "reference",
        "avg_depth": "depth_mean",
        "std_coverage": "depth_std",
        "depth_evenness": "depth_evenness",
        "breadth": "breadth",
        "num_alignments": "n_alns",
        "avg_read_length": "avg_read_length",
        "std_read_length": "stdev_read_length",
        "avg_identity": "avg_identity",
        "std_identity": "stdev_identity",
    }
    final_stats = final_stats.rename(columns=column_mapping)

    new_column_order = [
        "reference",
        "depth_mean",
        "depth_std",
        "depth_evenness",
        "breadth",
        "n_alns",
        "avg_read_length",
        "stdev_read_length",
        "avg_alignment_length",
        "avg_identity",
        "stdev_identity",
    ]

    missing_columns = set(new_column_order) - set(final_stats.columns)
    if missing_columns:
        log.error(f"Missing columns: {missing_columns}")
        exit(1)

    final_stats = final_stats[new_column_order]

    con = duckdb.connect(":memory:")
    con.register("final_stats", final_stats)
    con.execute(
        f"COPY (SELECT * FROM final_stats) TO '{out_files['coverage']}' (HEADER, DELIMITER '\t')"
    )
    con.close()

    column_mapping = {
        "percIdentity": "percIdentity",
        "alnLength": "alnLength",
        "mismatchCount": "mismatchCount",
        "gapOpenCount": "gapOpenCount",
        "queryStart": "queryStart",
        "queryEnd": "queryEnd",
        "subjectStart": "subjectStart",
        "subjectEnd": "subjectEnd",
        "eVal": "eVal",
        "bitScore": "bitScore",
        "qlen": "qlen",
        "slen": "slen",
        "cigar": "cigar",
        "qaln": "qaln",
        "saln": "saln",
    }

    new_column_order = [
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
    ]
    if "cigar" in df.columns:
        new_column_order.extend(["cigar", "qaln", "saln"])

    df = df.rename(columns=column_mapping)
    df[new_column_order].to_csv(out_files["multimap"], sep="\t", index=False)

    if mapping_file:
        log.info("Aggregating gene abundances")
        gene_abundances, gene_abundances_agg = aggregate_gene_abundances(
            mapping_file=mapping_file,
            gene_abundances=final_stats,
            num_threads=threads,
            temp_dir=tmp_files["db"],
        )

        if gene_abundances is None:
            log.info("Couldn't map anything to the references.")
            return

        log.info(f"Writing group abundances to {out_files['group_abundances']}")
        gene_abundances.to_csv(
            out_files["group_abundances"], sep="\t", index=False, compression="gzip"
        )

        if anvio:
            gene_abundances_agg_anvio = convert_to_anvio(
                df=gene_abundances, annotation_source=annotation_source
            )
            log.info(
                f"Writing abundances with anvi'o format to {out_files['group_abundances_anvio']}"
            )
            gene_abundances_agg_anvio.to_csv(
                out_files["group_abundances_anvio"],
                sep="\t",
                index=False,
                compression="gzip",
            )

        log.info(
            f"Writing aggregated group abundances to {out_files['group_abundances_agg']}"
        )
        gene_abundances_agg.to_csv(
            out_files["group_abundances_agg"], sep="\t", index=False, compression="gzip"
        )


def main() -> None:
    args, filters = get_arguments()
    setup_logging(args.debug)

    out_files = create_output_files(prefix=args.prefix, input_file=args.input)
    tmp_dir, tmp_files = setup_temporary_directory(base_dir=args.tmp_dir)

    final_stats, unique_subjects, inverse_indices, numpy_arrays, parquet_file = (
        process_data(args, filters, tmp_dir, tmp_files)
    )

    log.info("Filtering alignments")
    filtered_ids_df = filter_arrays(
        final_stats, unique_subjects, inverse_indices, numpy_arrays, tmp_files, args
    )

    df = process_filtered_data(filtered_ids_df, parquet_file, tmp_files, args)

    np_arrays = analyze_alignments(df)

    log.info("Getting coverage statistics")
    final_stats, unique_subjects, inverse_indices, numpy_arrays = calculate_statistics(
        np_arrays, tmp_files, num_threads=args.threads, rm_dups=False
    )

    if filters:
        log.info("Applying filters")
        final_stats = apply_filters(final_stats, filters)

    log.info(f"References kept: {final_stats.shape[0]:,}")
    df = df[df["subject_numeric_id"].isin(final_stats["subject_numeric_id"])]

    save_results(
        final_stats=final_stats,
        df=df,
        out_files=out_files,
        tmp_files=tmp_files,
        mapping_file=args.mapping_file,
        anvio=args.anvio,
        annotation_source=args.annotation_source,
    )
    log.info("ALL DONE.")


if __name__ == "__main__":
    main()
