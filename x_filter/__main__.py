from x_filter.logging_setup import setup_logging, get_logger
from x_filter.utils import get_arguments
from x_filter.ops import setup_temporary_directory, process_input_data, set_memory_limit
from x_filter.utils import apply_filters, create_output_files
from x_filter.stats import calculate_statistics
from x_filter.reassign import reassign
from x_filter.slice_mmap_arrays import parallel_slice_mmap
from x_filter.aggregate import aggregate_gene_abundances, convert_to_anvio
from typing import Dict, List, Tuple, Any
import logging
import duckdb
import pandas as pd
import numpy as np
import os
import glob
import time

# Get logger after configuration
log = get_logger()


def process_data(
    args: Any,
    filters: List[Dict[str, Any]],
    tmp_dir: str,
    tmp_files: Dict[str, str],
    output_files: Dict[str, str] = {},
    disable_initial_filtering: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray], str]:
    np_arrays, parquet_file = process_input_data(
        args.input,
        (tmp_dir, tmp_files),
        num_threads=args.threads,
        evalue_threshold=args.evalue,
        bitscore_threshold=args.bitscore,
        max_memory=args.max_memory,
        keep_db=args.keep_db,
        output_files=output_files,
        mmap_folder_dir=args.mmap_folder_dir,
        deduplicate=args.keep_duplicates,
    )

    if not disable_initial_filtering:
        log.info("Getting initial coverage statistics for filtering")
        final_stats, unique_subjects, inverse_indices, numpy_arrays = (
            calculate_statistics(
                np_arrays,
                tmp_files,
                num_threads=args.threads,
                max_memory=args.max_memory,
            )
        )
        # sort by breadth descending
        final_stats = final_stats.sort_values("breadth", ascending=True)

        if filters:
            log.info("Applying initial filters")
            final_stats = apply_filters(final_stats, filters)
            final_stats = final_stats.sort_values(
                ["breadth", "subject_numeric_id"], ascending=True
            )
    else:
        # Skip statistics calculation and filtering when disabled
        final_stats = pd.DataFrame()
        unique_subjects = np.array([])  # Empty array since we won't use it
        inverse_indices = np.array([])  # Empty array since we won't use it
        numpy_arrays = np_arrays

    return final_stats, unique_subjects, inverse_indices, numpy_arrays, parquet_file


def filter_arrays(
    final_stats: pd.DataFrame,
    unique_subjects: np.ndarray,
    inverse_indices: np.ndarray,
    numpy_arrays: Dict[str, np.ndarray],
    tmp_files: Dict[str, str],
    args: Any,
) -> pd.DataFrame:
    # If initial filtering was disabled or no filtering was done
    if final_stats.empty:
        log.info("Skipping array filtering - using all alignments")
        if args.skip_reassign:
            log.info("Skipping multimapping resolution...")
            return pd.DataFrame(
                {
                    "query_numeric_id": numpy_arrays["query_numeric_id"],
                    "subject_numeric_id": numpy_arrays["subject_numeric_id"],
                    "row_hash": numpy_arrays["row_hash"],
                }
            )

        log.info("Resolve multimappings...")
        return reassign(
            numpy_arrays, tmp_files, iters=args.n_iters, max_memory=args.max_memory
        )

    # Original filtering logic for when filtering is enabled
    target_subjects = final_stats["subject_numeric_id"].to_numpy()

    if target_subjects.size > 0:
        log.info("Filtering alignments")
        log.info(f"Number of target subjects: {len(target_subjects):,}")
        log.info(f"Number of unique subjects: {len(unique_subjects):,}")
        # target_indices = np.where(np.isin(unique_subjects, target_subjects))[0]
        target_set = set(target_subjects)
        target_indices = np.fromiter(
            (i for i, x in enumerate(unique_subjects) if x in target_set), dtype=int
        )
        if len(target_indices) == 0:
            raise ValueError(
                "None of the target subjects were found in unique_subjects."
            )
        log.info("Dumping filtered arrays to disk")
        filtered_arrays = parallel_slice_mmap(
            numpy_arrays,
            target_subjects,
            unique_subjects,
            inverse_indices,
            mmap_folder=tmp_files["mmap"],
            num_threads=args.threads,
            chunk_size=100_000_000,
        )

        if args.skip_reassign:
            log.info("Skipping multimapping resolution...")
            return pd.DataFrame(
                {
                    "query_numeric_id": filtered_arrays["query_numeric_id"],
                    "subject_numeric_id": filtered_arrays["subject_numeric_id"],
                    "row_hash": filtered_arrays["row_hash"],
                }
            )

        log.info("Resolve multimappings...")
        return reassign(
            filtered_arrays, tmp_files, iters=args.n_iters, max_memory=args.max_memory
        )


def process_filtered_data(
    filtered_ids_df: pd.DataFrame,
    db_file: str,
    tmp_files: Dict[str, str],
    args: Any,
) -> pd.DataFrame:
    """Process filtered data using DuckDB directly"""
    log.info("Retrieving filtered results from database")

    with duckdb.connect(database=db_file) as connection:
        # Set up the same configuration to avoid connection issues
        connection.execute(f"SET threads={args.threads}")
        connection.execute(f"SET temp_directory='{tmp_files['db']}'")
        connection.execute("SET preserve_insertion_order=false")
        if args.max_memory:
            formatted_memory = set_memory_limit(args.max_memory)
            connection.execute(f"SET memory_limit='{formatted_memory}'")
            connection.execute(f"SET max_memory='{formatted_memory}'")

        # Check if additional columns exist in the table
        table_info = connection.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'filtered_blast'
        """
        ).fetchall()
        table_columns = [col[0].lower() for col in table_info]

        # Add additional columns to selection if they exist
        additional_columns = ""
        if all(col in table_columns for col in ["cigar", "qaln", "taln"]):
            additional_columns = """
                ,blast.cigar,
                blast.qaln,
                blast.taln
            """

        # Register filtered IDs DataFrame directly
        connection.register("filtered_ids", filtered_ids_df)

        # Get results via join with explicit column references
        query = f"""
            SELECT DISTINCT 
                blast.queryId,
                blast.subjectId,
                blast.percIdentity,
                blast.alnLength,
                blast.mismatchCount,
                blast.gapOpenCount,
                blast.queryStart,
                blast.queryEnd,
                blast.subjectStart,
                blast.subjectEnd,
                blast.eVal,
                blast.bitScore,
                blast.qlen,
                blast.slen,
                blast.query_numeric_id,
                blast.subject_numeric_id,
                blast.row_hash
                {additional_columns}
            FROM filtered_blast AS blast
            INNER JOIN filtered_ids AS ids
            ON blast.query_numeric_id = ids.query_numeric_id
            AND blast.row_hash = ids.row_hash
        """
        results_df = connection.execute(query).df()

    connection.close()
    return results_df


def cleanup_mmap_files(mmap_folder: str) -> None:
    """Clean up memory mapped files in the given folder"""
    if os.path.exists(mmap_folder):
        for filename in os.listdir(mmap_folder):
            file_path = os.path.join(mmap_folder, filename)
            try:
                if os.path.isfile(file_path) and not file_path.startswith(".nfs"):
                    # Ensure any mmap is properly closed before deletion
                    try:
                        mmap = np.load(file_path, mmap_mode="r")
                        del mmap
                        # wait for the file to be closed
                        time.sleep(1)
                    except:
                        pass
                    os.unlink(file_path)
            except Exception as e:
                log.warning(f"Error deleting file {file_path}: {e}")


def cleanup_db_files(db_file: str) -> None:
    """Clean up database files"""
    if os.path.exists(db_file):
        try:
            os.unlink(db_file)
        except Exception as e:
            log.warning(f"Error deleting file {db_file}: {e}")


def cleanup_temp_files(tmp_files: Dict[str, str]) -> None:
    """Clean up all temporary files and directories"""
    for dir_type, dir_path in tmp_files.items():
        cleanup_mmap_files(dir_path)
        try:
            if os.path.exists(dir_path):
                os.rmdir(dir_path)
        except Exception as e:
            log.warning(f"Error deleting directory {dir_path}: {e}")


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
    # Create connection
    with duckdb.connect() as con:

        # Register DataFrames with DuckDB
        unique_subjects = df[["subjectId", "subject_numeric_id"]].drop_duplicates()
        con.register("final_stats", final_stats)
        con.register("unique_subjects", unique_subjects)

        # Perform the merge and column renaming in DuckDB
        coverage_query = """
            WITH merged AS (
                SELECT 
                    us.subjectId as reference,
                    fs.depth_mean,
                    fs.depth_std, 
                    fs.depth_evenness,
                    fs.breadth,
                    fs.num_alignments as n_alns,
                    fs.avg_read_length,
                    fs.std_read_length as stdev_read_length, 
                    fs.avg_alignment_length,
                    fs.avg_identity,
                    fs.std_identity as stdev_identity
                FROM final_stats fs
                LEFT JOIN unique_subjects us ON fs.subject_numeric_id = us.subject_numeric_id
            )
            SELECT 
                reference,
                depth_mean,
                depth_std,
                depth_evenness, 
                breadth,
                n_alns,
                avg_read_length,
                stdev_read_length,
                avg_alignment_length,
                avg_identity,
                stdev_identity
                FROM merged
            """

        # Export coverage results directly to TSV
        con.execute(
            f"COPY ({coverage_query}) TO '{out_files['coverage']}' "
            "(HEADER, DELIMITER '\t')"
        )

        # Register alignment DataFrame
        con.register("alignments", df)

        # Define columns for multimap output
        base_columns = [
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

        # Check if CIGAR columns exist and add them to the query
        cigar_columns = ["cigar", "qaln", "taln"] if "cigar" in df.columns else []
        all_columns = base_columns + cigar_columns
        columns_str = ", ".join(all_columns)

        # Export multimap results directly to TSV
        con.execute(
            f"COPY (SELECT {columns_str} FROM alignments) TO '{out_files['multimap']}' "
            "(HEADER, DELIMITER '\t')"
        )

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

            # Register gene abundance DataFrames
            con.register("gene_abundances", gene_abundances)
            con.register("gene_abundances_agg", gene_abundances_agg)

            # Export gene abundances to gzipped TSV
            con.execute(
                f"COPY (SELECT * FROM gene_abundances) TO '{out_files['group_abundances']}' "
                "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
            )

            con.execute(
                f"COPY (SELECT * FROM gene_abundances_agg) TO '{out_files['group_abundances_agg']}'"
                "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
            )

            if anvio:
                gene_abundances_anvio = convert_to_anvio(
                    df=gene_abundances, annotation_source=annotation_source
                )
                con.register("gene_abundances_anvio", gene_abundances_anvio)
                con.execute(
                    f"COPY (SELECT * FROM gene_abundances_anvio) TO '{out_files['group_abundances_anvio']}' "
                    "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
                )


def main() -> None:
    args, filters = get_arguments()

    setup_logging(args.debug)

    out_files = create_output_files(prefix=args.prefix, input_file=args.input)
    tmp_dir_obj, tmp_files = setup_temporary_directory(base_dir=args.tmp_dir)

    try:
        final_stats, unique_subjects, inverse_indices, numpy_arrays, db_file = (
            process_data(
                args,
                filters,
                tmp_dir_obj,
                tmp_files,
                out_files,
                disable_initial_filtering=args.disable_initial_filtering,
            )
        )

        if args.skip_reassign:
            log.info("Filtering alignments")
            log.warning("Skipping multimapping resolution...")
            filtered_ids_df = pd.DataFrame(
                {
                    "query_numeric_id": numpy_arrays["query_numeric_id"],
                    "subject_numeric_id": numpy_arrays["subject_numeric_id"],
                    "bitScore": numpy_arrays["bitScore"],
                    "alnLength": numpy_arrays["alnLength"],
                    "subjectStart": numpy_arrays["subjectStart"],
                    "subjectEnd": numpy_arrays["subjectEnd"],
                    "percIdentity": numpy_arrays["percIdentity"],
                    "row_hash": numpy_arrays["row_hash"],
                }
            )
        else:
            filtered_ids_df = filter_arrays(
                final_stats,
                unique_subjects,
                inverse_indices,
                numpy_arrays,
                tmp_files,
                args,
            )

        log.info(f"Number of alignments after filtering: {filtered_ids_df.shape[0]:,}")
        df = process_filtered_data(filtered_ids_df, db_file, tmp_files, args)

        np_arrays = analyze_alignments(df)

        # Always calculate final statistics regardless of initial filtering
        log.info("Getting coverage statistics")
        final_stats, unique_subjects, inverse_indices, numpy_arrays = (
            calculate_statistics(
                np_arrays,
                tmp_files,
                num_threads=args.threads,
                rm_dups=False,
                max_memory=args.max_memory,
            )
        )
        del inverse_indices

        # Always apply final filtering if filters exist
        if filters:
            log.info("Applying final filters")
            final_stats = apply_filters(final_stats, filters)

        log.info(f"References kept: {final_stats.shape[0]:,}")
        df = df[df["subject_numeric_id"].isin(final_stats["subject_numeric_id"])]

        save_results(
            final_stats=final_stats,
            df=df,
            out_files=out_files,
            tmp_files=tmp_files,
            threads=args.threads,
            mapping_file=args.mapping_file,
            anvio=args.anvio,
            annotation_source=args.annotation_source,
        )

        log.info("ALL DONE.")

    except Exception as e:
        # Clean up mmap files even if there's an error
        cleanup_mmap_files(tmp_files["mmap"])
        raise e
    finally:
        # Let the TemporaryDirectory cleanup handle itself
        tmp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
