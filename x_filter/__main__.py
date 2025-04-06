from x_filter.logging_setup import setup_logging, get_logger
from x_filter.utils import get_arguments, apply_filters, create_output_files
from x_filter.ops import (
    setup_temporary_directory,
    process_input_data,
    set_memory_limit,
    ColumnInfo,
    export_to_parquet,
    process_parquet_to_memmap,
)
from x_filter.stats import calculate_statistics
from x_filter.reassign import reassign
from x_filter.aggregate import aggregate_gene_abundances, convert_to_anvio

# Standard library imports
import os
import time
import logging
import gc
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional
import threading  # Add this import for proper thread synchronization

# Third-party imports
import numpy as np
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

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


def efficient_filter_arrays(
    final_stats: pd.DataFrame,
    numpy_arrays: Dict[str, np.ndarray],
    tmp_files: Dict[str, str],
    args: Any,
    max_chunk_size: int = 100_000_000,  # Add configurable max chunk size parameter
) -> str:
    """
    Efficiently filter arrays with parallelized filtering and writing
    
    Args:
        final_stats: DataFrame with filtering statistics
        numpy_arrays: Dictionary of numpy arrays to filter
        tmp_files: Dictionary with paths to temporary directories
        args: Command line arguments
        max_chunk_size: Maximum chunk size for memory-efficient processing
    
    Returns:
        Path to file with filtered IDs
    """
    num_threads = int(args.threads) if hasattr(args, "threads") else 1
    filtered_ids_file = os.path.join(tmp_files["db"], "filtered_ids.parquet")
    essential_columns = ["query_numeric_id", "row_hash"]

    # If no filtering needed
    if final_stats.empty:
        log.info("No filtering criteria - using all alignments")

        # Use parallel chunks for writing
        chunk_size = 100_000_000  # Larger chunks for better parallelization
        total_chunks = (
            len(numpy_arrays["query_numeric_id"]) + chunk_size - 1
        ) // chunk_size
        log.info(
            f"Writing {total_chunks} chunks in parallel with {num_threads} threads"
        )

        # Create a schema once
        schema = pa.schema(
            [
                (col, pa.from_numpy_dtype(numpy_arrays[col].dtype))
                for col in essential_columns
            ]
        )

        # Create temporary chunk files then merge
        chunk_files = []

        def process_chunk(chunk_idx):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(numpy_arrays["query_numeric_id"]))

            # Create chunk file path
            chunk_file = os.path.join(tmp_files["db"], f"chunk_{chunk_idx}.parquet")
            chunk_files.append(chunk_file)

            # Create arrays and table
            arrays = [
                pa.array(numpy_arrays[col][start:end]) for col in essential_columns
            ]
            table = pa.Table.from_arrays(arrays, essential_columns)

            # Write chunk
            pq.write_table(table, chunk_file)
            return end - start

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_chunk, i) for i in range(total_chunks)]

            with tqdm(total=total_chunks, desc="Writing chunks") as pbar:
                for future in futures:
                    future.result()
                    pbar.update(1)

        # Merge chunks
        log.info("Merging chunks")
        tables = [pq.read_table(file) for file in chunk_files]
        merged_table = pa.concat_tables(tables)
        pq.write_table(merged_table, filtered_ids_file)

        # Clean up chunk files
        for file in chunk_files:
            try:
                os.remove(file)
            except:
                pass

        if args.skip_reassign:
            return filtered_ids_file
        else:
            # For reassign, run it on full arrays
            reassigned_df = reassign(
                numpy_arrays,
                tmp_files,
                iters=args.n_iters,
                max_memory=args.max_memory,
                num_threads=num_threads,
            )

            # Write only essential columns
            reassigned_file = os.path.join(tmp_files["db"], "reassigned_ids.parquet")
            reassigned_df[essential_columns].to_parquet(reassigned_file)
            return reassigned_file

    # Filtering case
    target_subjects = np.array(
        list(set(final_stats["subject_numeric_id"].values)),
        dtype=numpy_arrays["subject_numeric_id"].dtype,
    )
    log.info(f"Filtering to {len(target_subjects):,} subjects")

    # Parallelize filtering and writing
    chunk_size = 100_000_000  # Larger chunks for better parallelization
    total_chunks = (
        len(numpy_arrays["subject_numeric_id"]) + chunk_size - 1
    ) // chunk_size
    log.info(f"Processing {total_chunks} chunks in parallel with {num_threads} threads")

    # Create temporary chunk files
    chunk_files = []
    match_counts = [np.int64(0)] * total_chunks  # Use np.int64 for match counts
    current_pos = np.int64(0)  # Explicitly use 64-bit integer
    position_lock = threading.Lock()  # Create a proper lock for thread synchronization
    accumulated_matches = np.zeros(total_chunks, dtype=np.int64)  # Track matches per chunk

    def filter_and_write_chunk(chunk_idx):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(numpy_arrays["subject_numeric_id"]))
        
        # Get chunk and find matches
        chunk_subjects = numpy_arrays["subject_numeric_id"][start:end]
        mask = np.isin(chunk_subjects, target_subjects)
        chunk_matches = np.sum(mask)

        if chunk_matches > 0:
            # Use proper lock for thread safety
            with position_lock:
                nonlocal current_pos  # Explicitly state we're using the outer current_pos
                current_pos += chunk_matches  # Update position atomically

            accumulated_matches[chunk_idx] = chunk_matches  # Store for debugging

            # Create chunk file path
            chunk_file = os.path.join(tmp_files["db"], f"chunk_{chunk_idx}.parquet")
            chunk_files.append(chunk_file)

            # Create arrays and table with only essential columns
            arrays = [
                pa.array(numpy_arrays[col][start:end][mask])
                for col in essential_columns
            ]
            table = pa.Table.from_arrays(arrays, essential_columns)

            # Write chunk
            pq.write_table(table, chunk_file)

            match_counts[chunk_idx] = chunk_matches
            return chunk_matches
        return 0

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(filter_and_write_chunk, i) for i in range(total_chunks)
        ]

        with tqdm(
            total=total_chunks, desc="Filtering chunks", leave=False, ncols=80
        ) as pbar:
            for future in futures:
                future.result()
                pbar.update(1)

    # Check if we found any matches
    total_matches = sum(match_counts)
    log.info(f"Found {total_matches:,} matching elements")

    if total_matches == 0:
        log.warning(
            "No matches found! This may indicate a problem with the filtering criteria."
        )
        return None

    # Merge chunks
    log.info("Merging filtered chunks")
    if chunk_files:
        tables = [pq.read_table(file) for file in chunk_files]
        merged_table = pa.concat_tables(tables)
        pq.write_table(merged_table, filtered_ids_file)

    # Clean up chunk files
    for file in chunk_files:
        try:
            os.remove(file)
        except:
            pass

    if args.skip_reassign:
        return filtered_ids_file
    else:
        # For reassign, use memory-mapped arrays for filtered data
        log.info("Creating memory-mapped filtered arrays for reassignment")

        # Create memory-mapped arrays in the temporary directory
        filtered_arrays = {}
        for key in [
            "query_numeric_id",
            "subject_numeric_id",
            "bitScore",
            "alnLength",
            "subjectStart",
            "subjectEnd",
            "percIdentity",
            "row_hash",
            "slen",
        ]:
            if key in numpy_arrays:
                mmap_path = os.path.join(tmp_files["mmap"], f"filtered_{key}.dat")
                filtered_arrays[key] = np.memmap(
                    mmap_path,
                    dtype=numpy_arrays[key].dtype,
                    mode="w+",
                    shape=(total_matches,),
                )

        # Process chunks to build filtered arrays
        current_pos = np.int64(0)  # Explicitly use 64-bit integer
        chunk_size = min(chunk_size, max_chunk_size)  # Use the configurable parameter
        position_lock = threading.Lock()  # Create a proper lock for thread synchronization
        total_chunks = (len(numpy_arrays["subject_numeric_id"]) + chunk_size - 1) // chunk_size
        accumulated_matches = np.zeros(total_chunks, dtype=np.int64)  # Track matches per chunk

        def build_filtered_chunk(chunk_idx):
            nonlocal current_pos
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(numpy_arrays["subject_numeric_id"]))

            # Get chunk and find matches
            chunk_subjects = numpy_arrays["subject_numeric_id"][start:end]
            mask = np.isin(chunk_subjects, target_subjects)
            chunk_matches = np.int64(np.sum(mask))  # Get number of matches as 64-bit int
            accumulated_matches[chunk_idx] = chunk_matches  # Store for debugging

            if chunk_matches > 0:
                # Use proper lock for thread safety
                with position_lock:
                    pos = current_pos  # Get current position
                    current_pos += chunk_matches  # Update position atomically
                    
                # Log more debug info for very large chunks
                if chunk_matches > 10_000_000:
                    log.debug(f"Chunk {chunk_idx}: Processing {chunk_matches:,} matches at position {pos:,}")

                # Copy filtered data to memory-mapped arrays
                for key in filtered_arrays:
                    filtered_arrays[key][pos : pos + chunk_matches] = numpy_arrays[key][
                        start:end
                    ][mask]

                # Explicitly flush changes
                for arr in filtered_arrays.values():
                    arr.flush()

                return chunk_matches
            return np.int64(0)  # Return 64-bit zero

        # Reset match counts for better tracking
        match_counts = [np.int64(0)] * total_chunks  # Use np.int64 for match counts
        current_pos = np.int64(0)  # Reset position counter

        # Process chunks in parallel with fewer workers for big data
        max_workers = min(num_threads, 8) if total_matches > 10_000_000_000 else num_threads
        log.info(f"Using {max_workers} parallel workers for building filtered arrays")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(build_filtered_chunk, i) for i in range(total_chunks)
            ]

            with tqdm(
                total=total_chunks, desc="Building filtered arrays", ncols=80
            ) as pbar:
                for i, future in enumerate(futures):
                    match_counts[i] = future.result()
                    pbar.update(1)

        # Verify we got the expected number of matches
        actual_matches = np.int64(sum(match_counts))  # Explicitly use 64-bit sum
        if actual_matches != total_matches:
            log.warning(f"Expected {total_matches:,} matches but got {actual_matches:,}")
            
            # More detailed diagnostics
            expected_from_chunks = np.sum(accumulated_matches)
            log.warning(f"Sum of accumulated matches: {expected_from_chunks:,}")
            if expected_from_chunks != actual_matches:
                log.warning("Mismatch between accumulated and returned match counts!")
                
            # Check for overflow
            if actual_matches < 0 or total_matches < 0:
                log.error("Integer overflow detected in match counts!")
                
            # Resize memory-mapped arrays to actual size for safety
            log.info(f"Resizing memory-mapped arrays to {actual_matches:,} elements")
            for key in filtered_arrays:
                try:
                    # Close the current memmap
                    filtered_arrays[key].flush()
                    del filtered_arrays[key]
                except:
                    pass
                
                # Create a new memmap with correct size
                new_mmap = np.memmap(
                    os.path.join(tmp_files["mmap"], f"filtered_{key}.dat"),
                    dtype=numpy_arrays[key].dtype,
                    mode="r+",
                    shape=(actual_matches,),
                )
                filtered_arrays[key] = new_mmap
                new_mmap.flush()

        # Perform reassignment
        log.info("Performing multimapping resolution")
        reassigned_df = reassign(
            filtered_arrays,
            tmp_files,
            iters=args.n_iters,
            max_memory=args.max_memory,
            num_threads=num_threads,
        )

        # Write only essential columns
        reassigned_file = os.path.join(tmp_files["db"], "reassigned_ids.parquet")
        reassigned_df[essential_columns].to_parquet(reassigned_file)
        return reassigned_file


def create_filtered_df_from_arrays(numpy_arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Create filtered DataFrame directly from numpy arrays."""
    return pd.DataFrame(
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


def process_filtered_data(
    filtered_ids_path: str,
    db_file: str,
    tmp_files: Dict[str, str],
    args: Any,
) -> str:
    """Process filtered data using DuckDB and write results to optimized Parquet (single file or directory)."""
    log.info(f"Processing filtered IDs from {filtered_ids_path}")

    with duckdb.connect(database=db_file) as connection:
        # Database configuration
        connection.execute(f"SET threads={args.threads}")
        connection.execute(f"SET temp_directory='{tmp_files['db']}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET enable_progress_bar=true")
        if args.max_memory:
            formatted_memory = set_memory_limit(args.max_memory)
            connection.execute(f"SET memory_limit='{formatted_memory}'")
            connection.execute(f"SET max_memory='{formatted_memory}'")

        # Define columns with DuckDB and NumPy types
        columns_info = {
            "queryId": ColumnInfo("queryId", "VARCHAR", "object"),
            "subjectId": ColumnInfo("subjectId", "VARCHAR", "object"),
            "percIdentity": ColumnInfo("percIdentity", "FLOAT4", "float32"),
            "alnLength": ColumnInfo("alnLength", "INTEGER", "int32"),
            "mismatchCount": ColumnInfo("mismatchCount", "SMALLINT", "int16"),
            "gapOpenCount": ColumnInfo("gapOpenCount", "SMALLINT", "int16"),
            "queryStart": ColumnInfo("queryStart", "INTEGER", "int32"),
            "queryEnd": ColumnInfo("queryEnd", "INTEGER", "int32"),
            "subjectStart": ColumnInfo("subjectStart", "INTEGER", "int32"),
            "subjectEnd": ColumnInfo("subjectEnd", "INTEGER", "int32"),
            "eVal": ColumnInfo("eVal", "DOUBLE", "float64"),
            "bitScore": ColumnInfo("bitScore", "FLOAT4", "float32"),
            "qlen": ColumnInfo("qlen", "INTEGER", "int32"),
            "slen": ColumnInfo("slen", "INTEGER", "int32"),
            "query_numeric_id": ColumnInfo("query_numeric_id", "BIGINT", "int64"),
            "subject_numeric_id": ColumnInfo("subject_numeric_id", "BIGINT", "int64"),
            "row_hash": ColumnInfo("row_hash", "BIGINT", "int64"),
        }

        # Check for additional columns (cigar, qaln, taln)
        table_info = connection.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'filtered_blast'"
        ).fetchall()
        table_columns = [col[0].lower() for col in table_info]
        if all(col in table_columns for col in ["cigar", "qaln", "taln"]):
            columns_info.update(
                {
                    "cigar": ColumnInfo("cigar", "VARCHAR", "object"),
                    "qaln": ColumnInfo("qaln", "VARCHAR", "object"),
                    "taln": ColumnInfo("taln", "VARCHAR", "object"),
                }
            )

        # Create a temporary view with filtered results
        connection.execute(
            f"""
            CREATE VIEW filtered_results AS
            SELECT {', '.join(col.name for col in columns_info.values())}
            FROM filtered_blast AS blast
            SEMI JOIN read_parquet('{filtered_ids_path}') AS ids
            ON blast.query_numeric_id = ids.query_numeric_id
            AND blast.row_hash = ids.row_hash
        """
        )

        # Get total rows for export
        total_rows = connection.execute(
            "SELECT COUNT(*) FROM filtered_results"
        ).fetchone()[0]
        log.info(f"Filtered results contain {total_rows:,} rows")

        # Export to Parquet using optimized function
        output_path = export_to_parquet(
            db_file=db_file,
            columns_info=columns_info,
            output_dir=tmp_files["tmp"],
            total_rows=total_rows,
            num_threads=args.threads,
            max_memory=args.max_memory,
            temp_dir=tmp_files["tmp"],
            keep_db=args.keep_db,
            output_files={},  # Not used here, but required by function signature
            table_name="filtered_results",
        )

        # Check if output_path is a directory or a single file
        if os.path.isdir(output_path):
            # If it's a directory, return the wildcard pattern
            final_output_path = f"{output_path}/*.parquet"
            log.info(
                f"Results written to directory {output_path} with files matching {final_output_path}"
            )
        else:
            # If it's a single file, return the file path directly
            final_output_path = output_path
            log.info(f"Results written to single file {final_output_path}")

    return final_output_path


def cleanup_mmap_files(mmap_folder: str) -> None:
    """Clean up memory mapped files in the given folder with robust handling"""
    if not os.path.exists(mmap_folder):
        return

    # First do a garbage collection to release any lingering references
    gc.collect()

    # Process all files in the directory and subdirectories
    for root, dirs, files in os.walk(mmap_folder, topdown=False):
        for filename in files:
            if filename.startswith(".nfs"):
                continue  # Skip NFS temporary files

            file_path = os.path.join(root, filename)
            try:
                # For memmap files, try to ensure they're properly closed
                if filename.endswith(".dat"):
                    try:
                        # Try to open and immediately close to check if it's locked
                        with open(file_path, "a"):
                            pass
                    except PermissionError:
                        # If file is locked, force garbage collection and wait
                        log.warning(
                            f"File {file_path} appears to be locked. Forcing cleanup..."
                        )
                        gc.collect()
                        time.sleep(2)

                # Try to delete the file
                if os.path.exists(file_path):
                    os.unlink(file_path)

            except Exception as e:
                log.warning(f"Error deleting file {file_path}: {e}")
                # In case of failure, try to make the file writable and retry
                try:
                    os.chmod(file_path, 0o666)
                    os.unlink(file_path)
                except Exception:
                    pass

        # Now try to delete the empty directories (bottom-up due to topdown=False)
        if root != mmap_folder:  # Don't delete the main mmap folder yet
            try:
                os.rmdir(root)
            except Exception as e:
                log.warning(f"Error deleting directory {root}: {e}")


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


def analyze_alignments(
    result_file: str, chunk_size: int = 1_000_000
) -> Dict[str, np.ndarray]:
    """Memory efficient version using chunked processing."""
    log.info(f"Analyzing alignments")

    with duckdb.connect() as con:
        con.execute("SET memory_limit='4GB'")  # Limit memory usage

        # Use streaming counts
        single_count = 0
        multi_count = 0
        total_unique = 0

        # Process in chunks
        query = f"""
        SELECT query_numeric_id, COUNT(*) as count
        FROM read_parquet('{result_file}')
        GROUP BY query_numeric_id
        """

        for chunk in con.execute(query).fetch_arrow_chunks():
            df_chunk = chunk.to_pandas()
            single_count += (df_chunk["count"] == 1).sum()
            multi_count += (df_chunk["count"] > 1).sum()
            total_unique += len(df_chunk)
            del df_chunk  # Explicit cleanup

        log.info(f"Number of unique reads: {total_unique:,}")
        log.info(f"Number of single alignment: {single_count:,}")
        log.info(f"Number of multi-alignment: {multi_count:,}")

        # Use memory-mapped output for large arrays
        arrays = {}
        mmap_dir = "tmp_mmap"
        os.makedirs(mmap_dir, exist_ok=True)

        # Get total count first
        total_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{result_file}')"
        ).fetchone()[0]

        # Create memory-mapped arrays
        for col in [
            "subject_numeric_id",
            "subjectStart",
            "subjectEnd",
            "alnLength",
            "qlen",
            "percIdentity",
            "slen",
        ]:
            mmap_file = os.path.join(mmap_dir, f"{col}.mmap")
            dtype = np.int32 if col not in ["percIdentity"] else np.float32
            arrays[col] = np.memmap(
                mmap_file, dtype=dtype, mode="w+", shape=(total_rows,)
            )

        # Process in chunks
        offset = 0
        chunk_query = f"""
        SELECT subject_numeric_id, subjectStart, subjectEnd, alnLength,
               qlen, percIdentity, slen
        FROM read_parquet('{result_file}')
        """

        for chunk in con.execute(chunk_query).fetch_arrow_chunks():
            df_chunk = chunk.to_pandas()
            chunk_size = len(df_chunk)

            for col in arrays:
                arrays[col][offset : offset + chunk_size] = df_chunk[col].values

            offset += chunk_size
            del df_chunk

        # Ensure arrays are flushed
        for arr in arrays.values():
            arr.flush()

    return arrays


def analyze_alignments_mmap(
    result_file: str,
    tmp_files: Dict[str, str],
    num_threads: int = 1,
    max_memory: Optional[int] = None,
) -> Dict[str, np.memmap]:
    """
    Analyze alignments from a Parquet file or directory using memory-mapped arrays for large datasets.

    Args:
        result_file: Path to Parquet file or directory of Parquet files
        tmp_files: Dictionary with paths to temporary directories
        num_threads: Number of threads to use for processing
        max_memory: Maximum memory to use in bytes (optional)

    Returns:
        Dictionary of memory-mapped arrays with alignment data
    """
    log.info(f"Analyzing alignments from {result_file} using memory mapping")

    # Create memmap subfolder
    memmap_dir = tmp_files["mmap"]
    os.makedirs(memmap_dir, exist_ok=True)

    # Define columns to extract with their types (DuckDB type, NumPy type)
    columns_info = {
        "subject_numeric_id": ("BIGINT", "int64"),
        "subjectStart": ("INTEGER", "int32"),
        "subjectEnd": ("INTEGER", "int32"),
        "alnLength": ("INTEGER", "int32"),
        "qlen": ("INTEGER", "int32"),
        "percIdentity": ("FLOAT4", "float32"),
        "slen": ("INTEGER", "int32"),
    }

    with duckdb.connect() as con:
        # Configure DuckDB
        con.execute(f"SET threads={num_threads}")
        con.execute(f"SET temp_directory='{tmp_files['db']}'")
        con.execute("SET enable_progress_bar=true")

        # Create temporary table from result file
        con.execute(
            f"CREATE TEMPORARY TABLE alignments AS SELECT * FROM read_parquet('{result_file}')"
        )

        # Get counts for query stats
        query_counts = con.execute(
            """
            SELECT query_numeric_id, COUNT(*) as count
            FROM alignments
            GROUP BY query_numeric_id
            """
        ).fetchall()

        # Calculate alignment stats
        single_count = sum(1 for qid, cnt in query_counts if cnt == 1)
        multi_count = sum(1 for qid, cnt in query_counts if cnt > 1)
        total_unique = len(query_counts)

        # Log statistics
        log.info(f"Number of unique reads: {total_unique:,}")
        log.info(f"Number of single alignment: {single_count:,}")
        log.info(f"Number of multi-alignment: {multi_count:,}")

        # Get total row count
        total_rows = con.execute("SELECT COUNT(*) FROM alignments").fetchone()[0]
        log.info(f"Total alignments to process: {total_rows:,}")

        # Export data to a temporary parquet file
        temp_parquet_file = os.path.join(tmp_files["db"], "temp_alignments.parquet")

        # Export the needed columns to parquet
        export_query = f"""
        COPY (
            SELECT
                subject_numeric_id,
                subjectStart,
                subjectEnd,
                alnLength,
                qlen,
                percIdentity,
                slen
            FROM alignments
        ) TO '{temp_parquet_file}' (FORMAT 'parquet')
        """
        con.execute(export_query)

    # Process the exported parquet file into memory-mapped arrays
    mmap_arrays = process_parquet_to_memmap(
        db_file=temp_parquet_file,  # Use the exported parquet file
        total_rows=total_rows,
        columns_info=columns_info,
        memmap_dir=memmap_dir,
        temp_dir=tmp_files["db"],
        num_threads=num_threads,
        max_memory=max_memory,
        skip_export=True,  # Skip export since we already created the parquet file
    )

    # Clean up temporary parquet file
    try:
        os.remove(temp_parquet_file)
    except Exception as e:
        log.warning(f"Failed to remove temporary parquet file: {e}")

    return mmap_arrays


def save_results(
    final_stats: pd.DataFrame,
    result_file: str,
    out_files: Dict[str, str],
    mapping_file: str,
    anvio: bool,
    annotation_source: str,
    tmp_files: Dict[str, str],
    threads: int = 1,
) -> None:
    with duckdb.connect() as con:
        # Register final_stats
        con.register("final_stats", final_stats)

        # Create a temporary view for result files
        con.execute(
            f"CREATE TEMPORARY VIEW alignments AS SELECT * FROM read_parquet('{result_file}')"
        )

        # Get column info from the view
        table_info = con.execute("PRAGMA table_info('alignments')").fetchall()
        columns = [col[1].lower() for col in table_info]
        cigar_columns = ", cigar, qaln, taln" if "cigar" in columns else ""
        all_columns = f"queryId, subjectId, percIdentity, alnLength, mismatchCount, gapOpenCount, queryStart, queryEnd, subjectStart, subjectEnd, eVal, bitScore, qlen, slen{cigar_columns}"

        # Create unique subjects view
        unique_subjects_query = """
            SELECT DISTINCT subjectId, subject_numeric_id
            FROM alignments
        """
        con.execute(
            "CREATE TEMPORARY TABLE unique_subjects AS " + unique_subjects_query
        )

        # Export coverage results
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
        con.execute(
            f"COPY ({coverage_query}) TO '{out_files['coverage']}' (HEADER, DELIMITER '\t')"
        )

        # Export multimap results
        con.execute(
            f"COPY (SELECT {all_columns} FROM alignments) TO '{out_files['multimap']}' (HEADER, DELIMITER '\t')"
        )

        # Gene abundances (if applicable)
        if mapping_file:
            log.info("Aggregating gene abundances")
            # Temporarily create DataFrame for gene abundances
            result_df = con.execute("SELECT * FROM alignments").df()
            gene_abundances, gene_abundances_agg = aggregate_gene_abundances(
                mapping_file=mapping_file,
                gene_abundances=final_stats,
                num_threads=threads,
                temp_dir=tmp_files["db"],
            )

            if gene_abundances is None:
                log.info("Couldn't map anything to the references.")
                return

            con.register("gene_abundances", gene_abundances)
            con.register("gene_abundances_agg", gene_abundances_agg)

            con.execute(
                f"COPY (SELECT * FROM gene_abundances) TO '{out_files['group_abundances']}' "
                "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
            )
            con.execute(
                f"COPY (SELECT * FROM gene_abundances_agg) TO '{out_files['group_abundances_agg']}' "
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


def close_and_cleanup_mmap_arrays(
    arrays: Dict[str, np.ndarray], tmp_files: Dict[str, str]
) -> None:
    """Explicitly close memory-mapped arrays and perform thorough cleanup"""
    # First delete all references to the arrays
    if arrays:
        log.info("Explicitly closing memory-mapped arrays")
        for name, arr in arrays.items():
            if isinstance(arr, np.memmap):
                try:
                    # For numpy memmap arrays, explicitly close/delete
                    del arr
                except Exception as e:
                    log.warning(f"Error closing memmap array {name}: {e}")

        # Clear the dictionary itself
        arrays.clear()
    # Force multiple garbage collections
    gc.collect()
    gc.collect()
    # Allow some time for OS to release file locks
    time.sleep(1)
    # Now cleanup the directory with extra retries
    retry_cleanup_mmap_files(tmp_files["mmap"])


def retry_cleanup_mmap_files(mmap_path, max_attempts=3, delay=1):
    """
    Attempt to clean up memory-mapped files with retry logic.

    Args:
        mmap_path: Path to the mmap file or directory containing mmap files
        max_attempts: Maximum number of retry attempts per file
        delay: Delay in seconds between retry attempts

    Returns:
        tuple: (success_count, failed_files) - Number of successfully cleaned files and list of files that couldn't be cleaned
    """
    log = logging.getLogger(__name__)
    success_count = 0
    failed_files = []

    if not mmap_path:
        log.debug("No memory-mapped path provided to clean up")
        return success_count, failed_files

    # Handle both file and directory cases
    if os.path.isdir(mmap_path):
        files_to_clean = glob.glob(os.path.join(mmap_path, "*"))
        # Also try to remove the directory itself after cleaning files
        dir_to_remove = mmap_path
    else:
        files_to_clean = [mmap_path] if os.path.exists(mmap_path) else []
        dir_to_remove = None

    # Clean individual files
    for file_path in files_to_clean:
        for attempt in range(max_attempts):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    log.debug(f"Successfully removed mmap file: {file_path}")
                    success_count += 1
                break
            except (OSError, PermissionError) as e:
                if attempt < max_attempts - 1:
                    log.debug(
                        f"Attempt {attempt+1}/{max_attempts} to remove {file_path} failed: {e}. Retrying..."
                    )
                    time.sleep(delay)
                else:
                    log.warning(
                        f"Failed to remove mmap file {file_path} after {max_attempts} attempts: {e}"
                    )
                    failed_files.append(file_path)

    # Try to remove the directory if it was a directory
    if dir_to_remove and os.path.exists(dir_to_remove):
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(dir_to_remove)
                log.debug(f"Successfully removed mmap directory: {dir_to_remove}")
                break
            except (OSError, PermissionError) as e:
                if attempt < max_attempts - 1:
                    log.debug(
                        f"Attempt {attempt+1}/{max_attempts} to remove directory {dir_to_remove} failed: {e}. Retrying..."
                    )
                    time.sleep(delay)
                else:
                    log.warning(
                        f"Failed to remove mmap directory {dir_to_remove} after {max_attempts} attempts: {e}"
                    )
                    failed_files.append(dir_to_remove)

    return success_count, failed_files


def safe_cleanup(tmp_dir_obj):
    """
    A safer version of tempfile's cleanup that handles potential issues with memory-mapped files.
    This function is designed to replace the standard cleanup method of TemporaryDirectory
    to better handle cases where memory-mapped files might still be locked.

    Args:
        tmp_dir_obj: The TemporaryDirectory object whose cleanup method is being replaced
    """
    import time
    import tempfile
    import gc

    # Get the directory path before any cleanup happens
    dir_path = tmp_dir_obj.name if hasattr(tmp_dir_obj, "name") else None
    if not dir_path or not os.path.exists(dir_path):
        return

    # Force garbage collection to release file handles
    gc.collect()
    gc.collect()
    # Allow some time for resources to be released
    time.sleep(0.5)

    try:
        # Try to use the original cleanup method first
        original_cleanup = getattr(tempfile.TemporaryDirectory, "_cleanup", None)
        if original_cleanup:
            original_cleanup(tmp_dir_obj)
        else:
            # If we can't access the original cleanup, use rmtree directly
            shutil.rmtree(dir_path, ignore_errors=True)
        log.debug(f"Successfully cleaned up temporary directory: {dir_path}")
    except Exception as e:
        log.warning(f"Standard cleanup of temporary directory {dir_path} failed: {e}")
        # Try our more aggressive cleanup approach
        try:
            # If dir still exists, try force_delete_mmap_folder
            if os.path.exists(dir_path):
                log.info(f"Attempting aggressive cleanup of {dir_path}")
                force_delete_mmap_folder(dir_path)
        except Exception as e2:
            log.warning(f"Failed to perform aggressive cleanup of {dir_path}: {e2}")


def force_delete_mmap_folder(mmap_folder: str) -> None:
    """Aggressive cleanup of memory-mapped files - to be called before tmp_dir_obj.cleanup()"""
    import gc
    import time

    if not os.path.exists(mmap_folder):
        return

    log.info(f"Forcibly cleaning mmap directory: {mmap_folder}")

    # First try to dereference any remaining arrays and force garbage collection
    gc.collect()
    gc.collect()
    time.sleep(1)  # Give OS time to release locks

    # Get list of all files
    all_files = []
    for root, dirs, files in os.walk(mmap_folder, topdown=False):
        for f in files:
            all_files.append(os.path.join(root, f))

    if all_files:
        log.info(f"Found {len(all_files)} files to remove")

        # Try to delete each file individually
        for file_path in all_files:
            try:
                os.unlink(file_path)
            except Exception as e:
                try:
                    # Try to make writable and delete
                    os.chmod(file_path, 0o666)
                    os.unlink(file_path)
                except Exception:
                    log.warning(f"Failed to remove: {file_path}")

    # Try to remove directory tree with force option
    try:
        # Using system command as a last resort
        if os.path.exists(mmap_folder):
            log.info(f"Using rm -rf to force delete: {mmap_folder}")
            os.system(f"rm -rf {mmap_folder}")
    except Exception as e:
        log.warning(f"Failed to force-delete mmap folder: {e}")


def main() -> None:
    args, filters = get_arguments()
    setup_logging(args.debug)
    out_files = create_output_files(prefix=args.prefix, input_file=args.input)
    tmp_dir_obj, tmp_files = setup_temporary_directory(base_dir=args.tmp_dir)
    np_arrays = None  # Track the arrays for cleanup

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

        filtered_ids_path = efficient_filter_arrays(
            final_stats, 
            numpy_arrays, 
            tmp_files, 
            args,
            max_chunk_size=10_000_000,  # Can be adjusted based on dataset size and memory
        )
        if filtered_ids_path is None:
            log.warning("No matching IDs found. Exiting.")
            return

        log.info(
            f"Number of alignments after filtering: Reading from {filtered_ids_path}"
        )
        result_file = process_filtered_data(filtered_ids_path, db_file, tmp_files, args)

        # Analyze directly from file
        np_arrays = analyze_alignments_mmap(
            result_file=result_file,
            tmp_files=tmp_files,
            num_threads=args.threads,
            max_memory=args.max_memory,
        )

        # Recalculate statistics
        log.info("Getting coverage statistics")
        final_stats, unique_subjects, inverse_indices, numpy_arrays = (
            calculate_statistics(
                np_arrays,
                tmp_files,
                num_threads=args.threads,
                max_memory=args.max_memory,
            )
        )
        del inverse_indices

        # Apply final filters
        if filters:
            log.info("Applying final filters")
            final_stats = apply_filters(final_stats, filters)
            log.info(f"References kept: {final_stats.shape[0]:,}")
            # Filter result_file based on final_stats
            filtered_result_file = os.path.join(
                tmp_files["db"], "final_filtered_results.parquet"
            )
            with duckdb.connect() as con:
                con.register("final_stats", final_stats)
                con.execute(
                    f"""
                    COPY (
                        SELECT r.*
                        FROM read_parquet('{result_file}') r
                        INNER JOIN final_stats fs
                        ON r.subject_numeric_id = fs.subject_numeric_id
                    ) TO '{filtered_result_file}' (FORMAT 'parquet')
                """
                )
            result_file = filtered_result_file

        # Save results
        save_results(
            final_stats=final_stats,
            result_file=result_file,
            out_files=out_files,
            tmp_files=tmp_files,
            threads=args.threads,
            mapping_file=args.mapping_file,
            anvio=args.anvio,
            annotation_source=args.annotation_source,
        )
    except Exception as e:
        log.error(f"Error occurred: {e}")
        raise e
    finally:
        # Cleanup all references explicitly
        time.sleep(1)  # Allow some time for OS to release file locks
        close_and_cleanup_mmap_arrays(np_arrays, tmp_files)
        tmp_dir_obj.cleanup()
        log.info("ALL DONE.")


if __name__ == "__main__":
    main()
