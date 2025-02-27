# x_filter/core/preprocessing.py

import os
import time
import numpy as np
import pandas as pd
import duckdb
from typing import Dict, Tuple, Union, Optional, List, Any
from pathlib import Path

from x_filter.utils.logging import get_logger, LogContext
from x_filter.core.io import (
    detect_input_type,
    format_memory_limit,
    load_existing_mmap_arrays,
)
from x_filter.utils.memory import track_memory

log = get_logger(__name__)


@track_memory(name="process_input_data", detailed=True)
def process_input_data(
    input_file: str,
    tmp_dir: str,
    tmp_files: Dict[str, str],
    out_files: Optional[Dict[str, str]] = None,
    num_threads: int = 1,
    evalue: float = 1e-10,
    bitscore: float = 60.0,
    max_memory: Union[str, int, float] = "4G",
    disable_initial_filtering: bool = False,
    mmap_folder_dir: Optional[str] = None,
    keep_db: bool = False,
    deduplicate: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray], str]:
    """
    Process input data and create memory-mapped arrays.

    Args:
        input_file: Path to input file (BLASTx results)
        tmp_dir: Temporary directory
        tmp_files: Dictionary of temporary file paths
        out_files: Optional dictionary of output file paths
        num_threads: Number of threads to use
        evalue: E-value threshold for filtering
        bitscore: Bit score threshold for filtering
        max_memory: Maximum memory to use
        disable_initial_filtering: Whether to disable initial filtering
        mmap_folder_dir: Optional directory for existing memory-mapped files
        keep_db: Whether to keep database files
        deduplicate: Whether to remove duplicate alignments

    Returns:
        Tuple of (stats_dataframe, unique_subjects, inverse_indices, numpy_arrays, db_file)
    """
    with LogContext(log, f"Processing input data from {input_file}"):
        db_dir = tmp_files["db"]
        db_file = os.path.join(db_dir, "blast.db")

        if keep_db and out_files:
            memmap_dir = out_files["mmap"]
            # Create memmap directory if it doesn't exist
            if not os.path.exists(memmap_dir):
                os.makedirs(memmap_dir)
        else:
            memmap_dir = tmp_files["mmap"]

        # Define column types for memory-mapped arrays
        column_data_types = {
            "percIdentity": ("FLOAT4", "float32"),
            "alnLength": ("INTEGER", "int32"),
            "subjectStart": ("INTEGER", "int32"),
            "subjectEnd": ("INTEGER", "int32"),
            "qlen": ("INTEGER", "int32"),
            "slen": ("INTEGER", "int32"),
            "subject_numeric_id": ("BIGINT", "int64"),
            "query_numeric_id": ("BIGINT", "int64"),
            "row_hash": ("BIGINT", "int64"),
            "bitScore": ("FLOAT4", "float32"),
        }

        # Check if we should use existing memory-mapped files
        if mmap_folder_dir:
            log.info(f"Using existing memory-mapped files from {mmap_folder_dir}")
            np_arrays = load_existing_mmap_arrays(mmap_folder_dir)

            # Create an empty dataframe and arrays for the case when we skip initial filtering
            if disable_initial_filtering:
                return pd.DataFrame(), np.array([]), np.array([]), np_arrays, ""

            # For the regular case, we need to calculate statistics
            from x_filter.core.coverage import calculate_coverage_statistics

            stats_df, unique_subjects, inverse_indices, _ = (
                calculate_coverage_statistics(
                    np_arrays, tmp_files, num_threads=num_threads, max_memory=max_memory
                )
            )

            return stats_df, unique_subjects, inverse_indices, np_arrays, ""

        # Detect input type
        input_type = detect_input_type(input_file)
        log.info(f"Detected input type: {input_type}")

        # Process input based on type
        if input_type == "parquet":
            log.info(f"Creating DuckDB database from Parquet")
            with duckdb.connect(database=db_file) as connection:
                connection.execute(f"SET threads={num_threads}")
                connection.execute(f"SET temp_directory='{tmp_dir}'")
                connection.execute("SET preserve_insertion_order=false")
                connection.execute("SET enable_progress_bar=true")

                if isinstance(max_memory, (int, float)) or (
                    isinstance(max_memory, str) and max_memory.isnumeric()
                ):
                    formatted_memory = format_memory_limit(max_memory, ratio=0.6)
                    connection.execute(f"SET memory_limit='{formatted_memory}'")
                    connection.execute(f"SET max_memory='{formatted_memory}'")

                # Handle Parquet directory
                parquet_path = (
                    input_file
                    if os.path.isfile(input_file)
                    else f"{input_file}/*.parquet"
                )

                # Create view
                connection.execute(
                    f"CREATE VIEW base_filtered_blast AS "
                    f"SELECT * FROM parquet_scan('{parquet_path}')"
                )

                # Apply deduplication if requested
                if deduplicate:
                    connection.execute(
                        "CREATE VIEW filtered_blast AS "
                        "SELECT DISTINCT ON (row_hash) * "
                        "FROM base_filtered_blast"
                    )
                else:
                    connection.execute(
                        "CREATE VIEW filtered_blast AS "
                        "SELECT * FROM base_filtered_blast"
                    )

                # Get total rows
                total_rows = connection.execute(
                    "SELECT COUNT(*) FROM filtered_blast"
                ).fetchone()[0]

                connection.commit()

        elif input_type == "tsv":
            log.info(f"Creating DuckDB database from TSV")

            # Create filtered blast table
            total_rows = create_filtered_blast_table(
                db_file=db_file,
                input_file=input_file,
                temp_dir=tmp_dir,
                num_threads=num_threads,
                max_memory=max_memory,
                evalue_threshold=evalue,
                bitscore_threshold=bitscore,
            )

            # Apply deduplication if requested
            if deduplicate:
                with duckdb.connect(database=db_file) as connection:
                    connection.execute(
                        "CREATE VIEW base_filtered_blast AS "
                        "SELECT * FROM filtered_blast"
                    )

                    connection.execute(
                        "CREATE TABLE temp_filtered_blast AS "
                        "SELECT DISTINCT ON (row_hash) * "
                        "FROM base_filtered_blast"
                    )

                    connection.execute("DROP TABLE filtered_blast")
                    connection.execute("DROP VIEW base_filtered_blast")
                    connection.execute(
                        "ALTER TABLE temp_filtered_blast RENAME TO filtered_blast"
                    )

                    new_total = connection.execute(
                        "SELECT COUNT(*) FROM filtered_blast"
                    ).fetchone()[0]

                    log.info(
                        f"Deduplicated {total_rows:,} rows to {new_total:,} unique rows "
                        f"({(total_rows - new_total):,} duplicates removed)"
                    )

                    total_rows = new_total
                    connection.commit()

        else:  # input_type == "duckdb"
            log.info("Using existing DuckDB database")
            db_file = input_file

            with duckdb.connect(db_file) as connection:
                if deduplicate:
                    connection.execute(
                        "CREATE VIEW base_filtered_blast AS "
                        "SELECT * FROM filtered_blast"
                    )

                    connection.execute("DROP TABLE filtered_blast")

                    connection.execute(
                        "CREATE VIEW filtered_blast AS "
                        "SELECT DISTINCT ON (row_hash) * "
                        "FROM base_filtered_blast"
                    )

                total_rows = connection.execute(
                    "SELECT COUNT(*) FROM filtered_blast"
                ).fetchone()[0]

                connection.commit()

        log.info(f"Total rows after initial processing: {total_rows:,}")

        # Convert to memory-mapped arrays
        np_arrays = process_to_memmap(
            db_file=db_file,
            total_rows=total_rows,
            columns_info=column_data_types,
            memmap_dir=memmap_dir,
            temp_dir=tmp_dir,
            num_threads=num_threads,
            max_memory=max_memory,
            keep_db=keep_db,
            output_files=out_files,
        )

        # If initial filtering is disabled, return empty stats
        if disable_initial_filtering:
            return pd.DataFrame(), np.array([]), np.array([]), np_arrays, db_file

        # Calculate coverage statistics
        from x_filter.core.coverage import calculate_coverage_statistics

        stats_df, unique_subjects, inverse_indices, _ = calculate_coverage_statistics(
            np_arrays, tmp_files, num_threads=num_threads, max_memory=max_memory
        )

        return stats_df, unique_subjects, inverse_indices, np_arrays, db_file


def create_filtered_blast_table(
    db_file: str,
    input_file: str,
    temp_dir: str,
    num_threads: int,
    max_memory: Optional[Union[str, int, float]],
    evalue_threshold: float,
    bitscore_threshold: float,
) -> int:
    """
    Create a filtered blast table in DuckDB from a TSV file.

    Args:
        db_file: Path to DuckDB database file
        input_file: Path to input file
        temp_dir: Temporary directory
        num_threads: Number of threads to use
        max_memory: Maximum memory to use
        evalue_threshold: E-value threshold for filtering
        bitscore_threshold: Bit score threshold for filtering

    Returns:
        Number of rows in the filtered table
    """
    with LogContext(log, "Creating filtered blast table"):
        with duckdb.connect(database=db_file) as connection:
            # Configure DuckDB
            connection.execute(f"SET threads={num_threads}")
            connection.execute(f"SET temp_directory='{temp_dir}'")
            connection.execute("SET preserve_insertion_order=false")
            connection.execute("SET enable_progress_bar=true")

            if max_memory:
                formatted_memory = format_memory_limit(max_memory, ratio=0.5)
                connection.execute(f"SET max_memory='{formatted_memory}'")

            # Get number of columns from input file
            num_columns = len(
                connection.execute(
                    f"SELECT * FROM read_csv_auto('{input_file}') LIMIT 1"
                ).description
            )

            log.debug(f"Detected {num_columns} columns in input file")

            hash_columns = ", ".join([f"column{i:02d}" for i in range(num_columns)])
            additional_columns = (
                """
                column14 AS cigar,
                column15 AS qaln,
                column16 AS taln,
                """
                if num_columns == 17
                else ""
            )

            # Create filtered table with explicit type casts
            create_table_sql = f"""
                CREATE TABLE filtered_blast AS
                WITH input_data AS (
                    SELECT *,
                        CAST(column10 AS DOUBLE) as evalue,
                        CAST(column11 AS FLOAT) as bitscore
                    FROM read_csv_auto('{input_file}', parallel=true)
                    WHERE CAST(column10 AS DOUBLE) <= {evalue_threshold} 
                    AND CAST(column11 AS FLOAT) >= {bitscore_threshold}
                )
                SELECT
                    column00 AS queryId,
                    CAST(hash(column01) % 9223372036854775807 AS BIGINT) AS subject_numeric_id,
                    CAST(hash(column00) % 9223372036854775807 AS BIGINT) AS query_numeric_id,
                    column01 AS subjectId,
                    CAST(column02 AS FLOAT4) AS percIdentity,
                    CAST(column03 AS INTEGER) AS alnLength,
                    CAST(column04 AS SMALLINT) AS mismatchCount,
                    CAST(column05 AS SMALLINT) AS gapOpenCount,
                    CAST(column06 AS INTEGER) AS queryStart,
                    CAST(column07 AS INTEGER) AS queryEnd,
                    CAST(column08 AS INTEGER) AS subjectStart,
                    CAST(column09 AS INTEGER) AS subjectEnd,
                    evalue AS eVal,
                    CAST(bitscore AS FLOAT4) AS bitScore,
                    CAST(column12 AS INTEGER) AS qlen,
                    CAST(column13 AS INTEGER) AS slen,
                    {additional_columns}
                    CAST(hash({hash_columns}) % 9223372036854775807 AS BIGINT) AS row_hash
                FROM input_data
            """
            connection.execute(create_table_sql)

            # Get total rows
            total_rows = connection.execute(
                "SELECT COUNT(*) FROM filtered_blast"
            ).fetchone()[0]

            log.info(f"Number of alignments after filtering: {total_rows:,}")

            connection.commit()

            return total_rows


def process_to_memmap(
    db_file: str,
    total_rows: int,
    columns_info: Dict[str, Tuple[str, str]],
    memmap_dir: str,
    temp_dir: str,
    num_threads: int,
    max_memory: Optional[Union[str, int, float]] = None,
    keep_db: bool = False,
    output_files: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Process database to memory-mapped arrays.

    Args:
        db_file: Path to DuckDB database file
        total_rows: Total number of rows in the table
        columns_info: Dictionary of column information (name: (duckdb_type, numpy_type))
        memmap_dir: Directory for memory-mapped files
        temp_dir: Temporary directory
        num_threads: Number of threads to use
        max_memory: Maximum memory to use
        keep_db: Whether to keep database files
        output_files: Optional dictionary of output file paths

    Returns:
        Dictionary of memory-mapped arrays
    """
    with LogContext(log, "Processing database to memory-mapped arrays"):
        os.makedirs(memmap_dir, exist_ok=True)

        # Create memory-mapped arrays
        memory_arrays = {}

        # Try loading everything directly if memory allows
        try:
            with duckdb.connect(database=db_file) as connection:
                connection.execute(f"SET threads={num_threads}")

                # Construct SQL query with explicit casts
                select_columns = []
                for col, (duckdb_type, _) in columns_info.items():
                    select_columns.append(f"CAST({col} AS {duckdb_type}) as {col}")

                select_sql = f"SELECT {', '.join(select_columns)} FROM filtered_blast"
                result = connection.execute(select_sql).fetchnumpy()

                # Convert to numpy arrays
                for col, (_, numpy_type) in columns_info.items():
                    # Create memory-mapped file
                    memmap_path = os.path.join(memmap_dir, f"{col}.dat")
                    memmap_array = np.memmap(
                        memmap_path,
                        dtype=np.dtype(numpy_type),
                        mode="w+",
                        shape=(total_rows,),
                    )

                    # Copy data to memory-mapped file
                    memmap_array[:] = result[col].astype(numpy_type, copy=False)
                    memmap_array.flush()

                    # Store array
                    memory_arrays[col] = memmap_array

                log.info(f"Created {len(memory_arrays)} memory-mapped arrays")
                return memory_arrays

        except Exception as e:
            log.warning(f"Failed to load data directly into memory: {e}")
            log.info("Falling back to chunked processing")

        # Process in chunks
        chunk_size = calculate_optimal_chunk_size(
            total_rows=total_rows,
            column_data_types=columns_info,
            max_memory=(
                max_memory
                if max_memory
                else int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 0.7)
            ),
        )

        log.info(f"Processing in chunks of {chunk_size:,} rows")

        # Create memory-mapped arrays
        for col, (_, numpy_type) in columns_info.items():
            memmap_path = os.path.join(memmap_dir, f"{col}.dat")
            memory_arrays[col] = np.memmap(
                memmap_path, dtype=np.dtype(numpy_type), mode="w+", shape=(total_rows,)
            )

        # Process chunks
        with duckdb.connect(database=db_file) as connection:
            connection.execute(f"SET threads={num_threads}")
            connection.execute(f"SET temp_directory='{temp_dir}'")

            # Construct SQL query with explicit casts
            select_columns = []
            for col, (duckdb_type, _) in columns_info.items():
                select_columns.append(f"CAST({col} AS {duckdb_type}) as {col}")

            offset = 0
            from tqdm import tqdm

            with tqdm(total=total_rows, desc="Processing rows", unit="rows") as pbar:
                while offset < total_rows:
                    remaining_rows = total_rows - offset
                    current_chunk_size = min(chunk_size, remaining_rows)

                    # Fetch chunk
                    chunk_sql = f"""
                        SELECT {', '.join(select_columns)}
                        FROM filtered_blast
                        LIMIT {current_chunk_size} OFFSET {offset}
                    """
                    chunk_data = connection.execute(chunk_sql).fetchnumpy()

                    # Copy to memory-mapped arrays
                    for col, (_, numpy_type) in columns_info.items():
                        memory_arrays[col][offset : offset + current_chunk_size] = (
                            chunk_data[col]
                        )

                    offset += current_chunk_size
                    pbar.update(current_chunk_size)

        # Flush all arrays
        for array in memory_arrays.values():
            array.flush()

        return memory_arrays


def calculate_optimal_chunk_size(
    total_rows: int,
    column_data_types: Dict[str, Tuple[str, str]],
    max_memory: Union[str, int, float],
    memory_safety_factor: float = 0.7,
    min_chunk_size: int = 100_000,
) -> int:
    """
    Calculate optimal chunk size for batch processing.

    Args:
        total_rows: Total number of rows
        column_data_types: Dictionary of column data types
        max_memory: Maximum memory to use
        memory_safety_factor: Safety factor for memory usage
        min_chunk_size: Minimum chunk size

    Returns:
        Optimal chunk size
    """
    # Convert string to bytes if needed
    if isinstance(max_memory, str):
        units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

        max_memory = max_memory.upper().strip()
        if max_memory[-1] in units:
            number = float(max_memory[:-1])
            unit = max_memory[-1]
            max_memory = int(number * units[unit])
        else:
            max_memory = int(max_memory)

    # Apply safety factor
    safe_memory = int(max_memory * memory_safety_factor)

    # Calculate memory required per row
    bytes_per_row = 0
    for _, (_, numpy_type) in column_data_types.items():
        bytes_per_row += np.dtype(numpy_type).itemsize

    # Calculate chunk size based on available memory
    # Use 2x the row size to account for temporary objects
    chunk_size = safe_memory // (bytes_per_row * 2)

    # Ensure chunk size is reasonable
    chunk_size = max(min_chunk_size, min(chunk_size, total_rows))

    # Round to a multiple of min_chunk_size
    chunk_size = (chunk_size // min_chunk_size) * min_chunk_size

    return chunk_size
