from typing import Dict, Tuple, Optional, List
import duckdb
import numpy as np
import os
import logging
from tqdm import tqdm
import tempfile
from x_filter.memory_tracker import track_memory
from x_filter.common import detect_input_type, validate_mmap_folder
import psutil
import queue
from itertools import count
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import threading
import collections


log = logging.getLogger("my_logger")


def validate_mmap_files(
    mmap_folder: str,
    expected_rows: int,
) -> Tuple[bool, List[str]]:
    """
    Validate that memory-mapped files have the expected number of rows.

    Args:
        mmap_folder: Path to folder containing memory-mapped files
        expected_rows: Expected number of rows in each mmap file

    Returns:
        Tuple containing:
        - bool: True if validation passes, False otherwise
        - List[str]: List of error messages if any
    """
    errors = []

    # Define expected dtypes for mmap files
    dtypes = {
        "percIdentity": "float32",
        "alnLength": "int32",
        "subjectStart": "int32",
        "subjectEnd": "int32",
        "qlen": "int32",
        "slen": "int32",
        "subject_numeric_id": "int64",
        "query_numeric_id": "int64",
        "row_hash": "int64",
        "bitScore": "float32",
    }

    # Get mmap file lengths
    mmap_lengths = {}
    for col, dtype in dtypes.items():
        file_path = os.path.join(mmap_folder, f"{col}.dat")
        if not os.path.exists(file_path):
            errors.append(f"Missing memory-mapped file: {col}.dat")
            continue

        # Calculate number of rows based on file size and dtype
        file_size = os.path.getsize(file_path)
        item_size = np.dtype(dtype).itemsize
        num_rows = file_size // item_size

        if file_size % item_size != 0:
            errors.append(
                f"File size for {col}.dat is not a multiple of its dtype size"
            )

        mmap_lengths[col] = num_rows

    if errors:
        return False, errors

    # Check all mmap files have same length
    first_length = next(iter(mmap_lengths.values()))
    mismatched_lengths = {
        col: length for col, length in mmap_lengths.items() if length != first_length
    }

    if mismatched_lengths:
        errors.append(
            "Inconsistent lengths across mmap files: "
            + ", ".join(
                f"{col}: {length}" for col, length in mismatched_lengths.items()
            )
        )

    # Compare with expected rows
    if first_length != expected_rows:
        errors.append(
            f"Memory-mapped files have {first_length} rows but expected {expected_rows} rows"
        )

    return len(errors) == 0, errors


def load_existing_mmap_arrays(
    folder_path: str, expected_rows: Optional[int] = None
) -> Dict[str, np.memmap]:
    """
    Load existing memory-mapped arrays from a folder with row count validation.

    Args:
        folder_path: Path to folder containing .dat files
        expected_rows: Optional expected number of rows for validation

    Returns:
        Dict[str, np.memmap]: Dictionary of memory-mapped arrays
    """
    # Validate folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")

    # If expected_rows provided, validate mmap files
    if expected_rows is not None:
        success, errors = validate_mmap_files(folder_path, expected_rows)
        if not success:
            raise ValueError(
                "Memory-mapped files validation failed:\n"
                + "\n".join(f"- {error}" for error in errors)
            )

    # Define expected dtypes for each column
    dtypes = {
        "percIdentity": "float32",
        "alnLength": "int32",
        "subjectStart": "int32",
        "subjectEnd": "int32",
        "qlen": "int32",
        "slen": "int32",
        "subject_numeric_id": "int64",
        "query_numeric_id": "int64",
        "row_hash": "int64",
        "bitScore": "float32",
    }

    # Get total rows from first mmap file
    first_file = os.path.join(folder_path, f"{next(iter(dtypes))}.dat")
    total_rows = (
        os.path.getsize(first_file) // np.dtype(dtypes[next(iter(dtypes))]).itemsize
    )

    # Load all mmap arrays
    mmap_arrays = {}
    for col, dtype in dtypes.items():
        file_path = os.path.join(folder_path, f"{col}.dat")
        expected_size = total_rows * np.dtype(dtype).itemsize
        actual_size = os.path.getsize(file_path)

        if actual_size != expected_size:
            raise ValueError(
                f"Size mismatch for {col}: expected {expected_size} bytes, got {actual_size} bytes"
            )

        mmap_arrays[col] = np.memmap(
            file_path, dtype=dtype, mode="r", shape=(total_rows,)
        )

    logging.info(
        f"Successfully loaded {len(mmap_arrays)} memory-mapped arrays with {total_rows:,} rows"
    )
    return mmap_arrays


def create_filtered_blast_from_parquet(
    db_file: str,
    input_file: str,
    temp_dir: str,
    num_threads: int,
    max_memory: Optional[int],
) -> int:
    """Create the filtered blast table as a view over Parquet input"""
    with duckdb.connect(database=db_file) as connection:
        # Configure DuckDB
        connection.execute(f"SET threads={num_threads}")
        connection.execute(f"SET temp_directory='{temp_dir}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET enable_progress_bar=true")
        if max_memory:
            formatted_memory = set_memory_limit(max_memory, ratio=0.6)
            connection.execute(f"SET memory_limit='{formatted_memory}'")
            connection.execute(f"SET max_memory='{formatted_memory}'")

        # Handle Parquet directory
        parquet_path = (
            input_file if os.path.isfile(input_file) else f"{input_file}/*.parquet"
        )

        # Create a view over the parquet data
        connection.execute(
            f"""
            CREATE VIEW filtered_blast AS 
            SELECT * FROM parquet_scan('{parquet_path}')
        """
        )

        # Get total rows
        total_rows = connection.execute(
            "SELECT COUNT(*) FROM filtered_blast"
        ).fetchone()[0]
        log.info(f"Created filtered_blast view over Parquet with {total_rows:,} rows")

        connection.commit()

    return total_rows


def get_system_memory() -> int:
    """Get total system memory in bytes"""
    return psutil.virtual_memory().total


def set_memory_limit(max_memory: int, ratio: float = 1.0) -> str:
    """Convert memory limit to DuckDB format with optional ratio adjustment"""
    adjusted_memory = int(max_memory * ratio)
    if adjusted_memory >= 1024 * 1024 * 1024:  # If adjusted_memory is 1GB or more
        memory_value = adjusted_memory // (1024 * 1024 * 1024)  # Integer division
        unit = "G"
    else:
        memory_value = adjusted_memory
        unit = "B"
    return f"{memory_value}{unit}"


def setup_temporary_directory(
    base_dir: Optional[str] = None,
) -> Tuple[tempfile.TemporaryDirectory, Dict[str, str]]:
    """Setup temporary directory structure"""
    if base_dir is None:
        base_dir = os.getcwd()
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        temp_dir = tempfile.TemporaryDirectory(dir=base_dir, prefix="xfilter-")
    else:
        base_dir = os.path.abspath(base_dir)
        if not os.path.exists(base_dir):
            raise OSError(f"Base directory {base_dir} does not exist.")
        if len(base_dir) > 107:
            raise OSError(f"Base directory {base_dir} exceeds 107 characters.")
        temp_dir = tempfile.TemporaryDirectory(dir=base_dir, prefix="xfilter-")

    temp_dir_path = temp_dir.name
    log.info(f"Temporary directory: {temp_dir_path}")

    temp_subdirectories = {
        "mmap": os.path.join(temp_dir_path, "mmap"),
        "db": os.path.join(temp_dir_path, "db"),
        "tmp": os.path.join(temp_dir_path),
    }

    for path in temp_subdirectories.values():
        if not os.path.exists(path):
            os.makedirs(path)

    return temp_dir, temp_subdirectories


def calculate_memory_requirements(
    chunk_size: int,
    column_data_types: Dict[str, Tuple[str, str]],
    memory_overhead_factor: float = 3.0,
) -> int:
    """Calculate memory required for processing columns in batch"""
    total_memory = 0
    for _, (_, numpy_type) in column_data_types.items():
        dtype = np.dtype(numpy_type)
        bytes_per_row = dtype.itemsize
        column_memory = chunk_size * bytes_per_row * memory_overhead_factor
        total_memory += column_memory
    return int(total_memory)


# def calculate_optimal_chunk_size(
#     total_rows: int,
#     column_data_types: Dict[str, Tuple[str, str]],
#     max_memory: int,
#     memory_safety_factor: float = 0.8,
#     min_chunk_size: int = 100_000,
# ) -> int:
#     """Calculate optimal chunk size for batch processing"""
#     safe_memory = max_memory * memory_safety_factor

#     # Binary search for optimal chunk size
#     min_size = min_chunk_size
#     max_size = total_rows
#     optimal_chunk_size = min_size

#     while min_size <= max_size:
#         current_size = (min_size + max_size) // 2
#         total_memory = calculate_memory_requirements(
#             chunk_size=current_size,
#             column_data_types=column_data_types,
#         )

#         if total_memory <= safe_memory:
#             optimal_chunk_size = current_size
#             min_size = current_size + 1
#         else:
#             max_size = current_size - 1

#     # Round to nearest multiple of min_chunk_size
#     optimal_chunk_size = max(
#         min_chunk_size, (optimal_chunk_size // min_chunk_size) * min_chunk_size
#     )

#     return optimal_chunk_size


# def process_db_columns_to_memmap_batch(
#     db_file: str,
#     total_rows: int,
#     columns_info: Dict[str, Tuple[str, str]],
#     memmap_dir: str,
#     temp_dir: str,
#     num_threads: int,
#     chunk_size: int,
#     max_memory: Optional[int] = None,
# ) -> Dict[str, np.memmap]:
#     """Process multiple columns in batches using a single SQL query per chunk"""
#     log.debug(
#         f"Creating memmaps for {len(columns_info)} columns with {total_rows:,} rows"
#     )

#     # Initialize all memmap files
#     memmap_arrays = {}
#     total_size = 0
#     for column_name, (_, numpy_type) in columns_info.items():
#         memmap_file_path = os.path.join(memmap_dir, f"{column_name}.dat")
#         dtype = np.dtype(numpy_type)
#         total_size += total_rows * dtype.itemsize
#         memmap_arrays[column_name] = np.memmap(
#             memmap_file_path, mode="w+", shape=(total_rows,), dtype=dtype
#         )

#     log.info(f"Total memmap size: {total_size / (1024**3):.2f} GB")

#     # Construct optimized SELECT statement with explicit casts
#     select_columns = []
#     for col, (duckdb_type, _) in columns_info.items():
#         select_columns.append(f"CAST({col} AS {duckdb_type}) as {col}")

#     select_sql = f"""
#         SELECT {', '.join(select_columns)}
#         FROM filtered_blast
#         LIMIT ? OFFSET ?
#     """

#     # Process the columns in chunks
#     with duckdb.connect(database=db_file) as connection:
#         connection.execute(f"SET threads={num_threads}")
#         connection.execute(f"SET temp_directory='{temp_dir}'")
#         connection.execute("SET preserve_insertion_order=true")
#         connection.execute("SET enable_progress_bar=false")
#         if max_memory:
#             # Set DuckDB to use only 50% of max memory
#             formatted_memory = set_memory_limit(max_memory, ratio=0.5)
#             connection.execute(f"SET memory_limit='{formatted_memory}'")
#             connection.execute(f"SET max_memory='{formatted_memory}'")

#         offset = 0
#         with tqdm(
#             total=total_rows,
#             desc="Processing rows",
#             unit="rows",
#             # disable=is_debug(),
#             leave=False,
#             position=0,
#         ) as pbar:
#             # Add a second progress bar for memory written
#             with tqdm(
#                 total=total_size,
#                 desc="Memory written",
#                 unit="B",
#                 unit_scale=True,
#                 unit_divisor=1024,
#                 disable=is_debug(),
#                 leave=False,
#                 position=1,
#             ) as mem_pbar:
#                 while offset < total_rows:
#                     remaining_rows = total_rows - offset
#                     current_chunk_size = min(chunk_size, remaining_rows)

#                     # Fetch all columns for this chunk in a single query
#                     chunk_data = connection.execute(
#                         select_sql, [current_chunk_size, offset]
#                     ).fetchnumpy()

#                     # Write each column's data to its memmap
#                     chunk_memory = 0
#                     for column_name, (_, numpy_type) in columns_info.items():
#                         chunk = chunk_data[column_name]
#                         chunk_memory += chunk.nbytes
#                         memmap_arrays[column_name][offset : offset + len(chunk)] = chunk

#                     offset += current_chunk_size
#                     pbar.update(current_chunk_size)
#                     mem_pbar.update(chunk_memory)

#     # Flush and reopen all memmaps in read mode
#     read_memmap_arrays = {}
#     for column_name, (_, numpy_type) in columns_info.items():
#         memmap_file_path = os.path.join(memmap_dir, f"{column_name}.dat")
#         memmap_arrays[column_name].flush()
#         del memmap_arrays[column_name]
#         read_memmap_arrays[column_name] = np.memmap(
#             memmap_file_path, mode="r", dtype=np.dtype(numpy_type)
#         )

#     return read_memmap_arrays
# Standard library imports
import os
import numpy as np
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Tuple, Optional, List, Union
import logging as log
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from dataclasses import dataclass
from pathlib import Path
import math
import threading
from queue import Queue


@dataclass
class ColumnInfo:
    name: str
    duckdb_type: str
    numpy_type: str
    chunk_offset: int = 0

    @classmethod
    def from_tuple(cls, name: str, type_tuple: Tuple[str, str]) -> "ColumnInfo":
        duckdb_type, numpy_type = type_tuple
        return cls(name=name, duckdb_type=duckdb_type, numpy_type=numpy_type)


def format_memory_size(size_in_bytes: int) -> str:
    """Convert bytes to appropriate memory unit string"""
    if size_in_bytes >= 1024**3:  # GB
        return f"{size_in_bytes / (1024**3):.2f}GB"
    elif size_in_bytes >= 1024**2:  # MB
        return f"{size_in_bytes / (1024**2):.2f}MB"
    else:  # KB
        return f"{size_in_bytes / 1024:.2f}KB"


def setup_logging():
    """Configure logging format"""
    log_format = "%(asctime)s [%(threadName)12s] %(levelname)8s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log.basicConfig(level=log.INFO, format=log_format, datefmt=date_format)


def calculate_optimal_row_group_size(
    total_rows: int, columns_info: Dict[str, ColumnInfo], available_memory: int
) -> int:
    """
    Calculate optimal row group size based on Parquet recommendations (512MB-1GB per group)
    """

    # Calculate average row size based on column types
    row_size = 0
    for col_info in columns_info.values():
        # Get size in bytes for each type
        dtype = np.dtype(col_info.numpy_type)
        row_size += dtype.itemsize

    # Target row group size (aim for 768MB = middle of recommended range)
    TARGET_GROUP_SIZE = 768 * 1024 * 1024  # 768MB in bytes

    # Calculate rows needed to reach target size
    rows_for_target = TARGET_GROUP_SIZE // row_size

    # Set bounds based on Parquet recommendations
    MIN_GROUP_SIZE_BYTES = 512 * 1024 * 1024  # 512MB
    MAX_GROUP_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB

    min_rows = MIN_GROUP_SIZE_BYTES // row_size
    max_rows = MAX_GROUP_SIZE_BYTES // row_size

    # Ensure we don't exceed total rows
    max_rows = min(max_rows, total_rows)

    # Adjust based on available memory (ensure group can fit in memory)
    mem_limited_rows = (
        available_memory // 2
    ) // row_size  # Use at most 50% of available memory

    # Select final row count
    row_group_size = min(rows_for_target, mem_limited_rows, max_rows)
    row_group_size = max(row_group_size, min_rows)

    # Log the decision
    group_size_mb = (row_group_size * row_size) / (1024 * 1024)
    log.debug(
        f"""
        Row group size calculation:
        - Row size: {row_size} bytes
        - Target group size: 768 MB
        - Calculated rows per group: {row_group_size:,}
        - Actual group size: {group_size_mb:.2f} MB
        - Total groups: {total_rows / row_group_size:.1f}
    """
    )

    return int(row_group_size)


def export_to_parquet(
    db_file: str,
    columns_info: Dict[str, ColumnInfo],
    output_dir: str,
    total_rows: int,
    chunk_size: Optional[int] = None,
    compression: str = "zstd",
    compression_level: int = 3,
    num_threads: int = 1,
    max_memory: Optional[int] = None,
    temp_dir: Optional[str] = None,
    keep_db: bool = False,
    output_files: Optional[Dict[str, str]] = None,
    table_name: str = "filtered_blast",
) -> str:
    """Export DuckDB table to optimized Parquet file with recommended row group sizes"""

    if chunk_size is None:
        # Calculate optimal size based on Parquet recommendations
        available_mem = psutil.virtual_memory().available
        chunk_size = calculate_optimal_row_group_size(
            total_rows=total_rows,
            columns_info=columns_info,
            available_memory=available_mem,
        )
    if keep_db:
        # Create timestamp for the filename
        output_path = output_files["parquet"]
    else:
        output_path = os.path.join(output_dir, "db", "export.parquet")

    # Configure DuckDB connection
    conn = duckdb.connect(database=db_file)
    conn.execute(f"SET threads={num_threads}")
    conn.execute("SET preserve_insertion_order=true")
    conn.execute("SET enable_progress_bar=true")
    max_memory_str = set_memory_limit(max_memory, ratio=0.6) if max_memory else None
    if max_memory_str:
        conn.execute(f"SET memory_limit='{max_memory_str}'")
        conn.execute(f"SET max_memory='{max_memory_str}'")

    # Prepare column selection with proper casting
    # select_columns = [
    #     f"CAST({col_info.name} AS {col_info.duckdb_type}) as {col_info.name}"
    #     for col_info in columns_info.values()
    # ]

    # Export query with optimized settings
    export_sql = f"""
        COPY {table_name} TO '{output_path}'
            (
                FORMAT PARQUET,
                ROW_GROUP_SIZE {chunk_size},
                COMPRESSION 'ZSTD',
                COMPRESSION_LEVEL 3,
                PER_THREAD_OUTPUT
            )
    """

    log.debug(
        f"""
        Parquet export settings:
        - Row group size: {chunk_size:,} rows ({(chunk_size * sum(np.dtype(ci.numpy_type).itemsize for ci in columns_info.values())) / (1024*1024):.2f} MB)
        - Compression: {compression.upper()}
        - Compression level: {compression_level}
        - Threads: {num_threads}
        - Total rows: {total_rows:,}
        - Number of groups: {math.ceil(total_rows / chunk_size)}
    """
    )

    conn.execute(export_sql)
    conn.close()

    return output_path


def verify_parquet_file(parquet_path: str, total_expected: int) -> List[int]:
    """Verify the Parquet file has enough rows and validate row group consistency"""
    # First check if it's a directory
    if os.path.isdir(parquet_path):
        # It's a directory created by PER_THREAD_OUTPUT, look for all parquet files
        parquet_files = sorted(
            [f for f in os.listdir(parquet_path) if f.endswith(".parquet")]
        )
        if not parquet_files:
            raise ValueError(f"No parquet files found in directory {parquet_path}")

        total_rows = 0
        all_group_sizes = []

        # Process each file in the directory
        for pf in parquet_files:
            full_path = os.path.join(parquet_path, pf)
            reader = pq.ParquetFile(full_path)

            total_groups = reader.metadata.num_row_groups
            for i in range(total_groups):
                rows = reader.metadata.row_group(i).num_rows
                total_rows += rows
                all_group_sizes.append(rows)

        if total_rows != total_expected:
            raise ValueError(
                f"Parquet files in directory have {total_rows} rows but expected {total_expected}"
            )

        return all_group_sizes

    else:
        # Original single-file handling
        reader = pq.ParquetFile(parquet_path)
        total_groups = reader.metadata.num_row_groups

        total_rows = 0
        group_sizes = []
        for i in range(total_groups):
            rows = reader.metadata.row_group(i).num_rows
            total_rows += rows
            group_sizes.append(rows)

        if total_rows != total_expected:
            raise ValueError(
                f"Parquet file has {total_rows} rows but expected {total_expected}"
            )

        return group_sizes


def get_parquet_files(parquet_path: str) -> List[str]:
    """Get list of parquet files to process"""
    path = Path(parquet_path)
    if path.is_file():
        return [str(path)]
    return sorted(str(p) for p in path.glob("*.parquet"))


def process_parquet_chunk(
    thread_id: int,
    parquet_path: str,
    row_range: Tuple[int, int],
    columns_info: Dict[str, ColumnInfo],
    memmap_arrays: Dict[str, np.memmap],
    group_sizes: List[int],
    progress_bar: Optional[tqdm] = None,
) -> None:
    """Process a chunk of Parquet data and write to memory-mapped arrays"""
    start_row, end_row = row_range

    try:
        if os.path.isdir(parquet_path):
            # Handle directory of parquet files
            parquet_files = sorted(
                [
                    os.path.join(parquet_path, f)
                    for f in os.listdir(parquet_path)
                    if f.endswith(".parquet")
                ]
            )
            if not parquet_files:
                raise ValueError(f"No parquet files found in directory {parquet_path}")
        else:
            # Single file case
            parquet_files = [parquet_path]

        # Calculate cumulative group positions
        group_positions = [0]
        current_pos = 0
        for size in group_sizes:
            current_pos += size
            group_positions.append(current_pos)

        write_pos = start_row
        current_group = 0

        # Process each file
        for pfile in parquet_files:
            reader = pq.ParquetFile(pfile)
            file_groups = reader.metadata.num_row_groups

            for group_idx in range(file_groups):
                # Skip groups before our range
                if group_positions[current_group + 1] <= start_row:
                    current_group += 1
                    continue

                # Stop if we've gone past our range
                if group_positions[current_group] >= end_row:
                    break

                # Calculate overlap with our chunk
                group_start = group_positions[current_group]
                group_size = group_sizes[current_group]

                read_start = max(0, start_row - group_start)
                read_end = min(group_size, end_row - group_start)

                if read_start < read_end:
                    rows_to_write = read_end - read_start

                    # Read group data
                    group = reader.read_row_group(group_idx)

                    # Process each column
                    for col_name in columns_info:
                        data = group.column(col_name).to_numpy()
                        if data.shape[0] != group_size:
                            raise ValueError(
                                f"Size mismatch in {pfile}, group {group_idx}: "
                                f"Expected {group_size} rows, got {data.shape[0]} "
                                f"for column {col_name}"
                            )

                        # Write the slice
                        data_slice = data[read_start:read_end]
                        memmap_arrays[col_name][
                            write_pos : write_pos + rows_to_write
                        ] = data_slice

                    write_pos += rows_to_write

                    if progress_bar:
                        with progress_bar.get_lock():
                            progress_bar.update(rows_to_write)

                current_group += 1

                if write_pos >= end_row:
                    break

            if write_pos >= end_row:
                break

        if write_pos != end_row:
            raise ValueError(
                f"Write position mismatch in thread {thread_id}: "
                f"Expected to write to position {end_row}, "
                f"but ended at position {write_pos}"
            )

    except Exception as e:
        log.error(
            f"Error in thread {thread_id} processing chunk {start_row}-{end_row}: {str(e)}"
        )
        raise


def process_parquet_to_memmap(
    db_file: str,
    total_rows: int,
    columns_info: Dict[str, Tuple[str, str]],
    memmap_dir: str,
    temp_dir: str,
    num_threads: int,
    chunk_size: Optional[int] = None,
    max_memory: Optional[int] = None,
    compression: str = "zstd",
    compression_level: int = 3,
    skip_export: bool = False,
    keep_db: bool = False,
    output_files: Optional[Dict[str, str]] = None,
) -> Dict[str, np.memmap]:
    """Convert database to parquet and then to memory-mapped arrays"""

    setup_logging()

    column_specs = {
        name: ColumnInfo.from_tuple(name, type_info)
        for name, type_info in columns_info.items()
    }

    # Export to parquet
    if not skip_export:
        parquet_file = export_to_parquet(
            db_file=db_file,
            columns_info=column_specs,
            output_dir=temp_dir,
            total_rows=total_rows,
            chunk_size=chunk_size,
            compression=compression,
            compression_level=compression_level,
            num_threads=num_threads,
            temp_dir=temp_dir,
            max_memory=max_memory,
            keep_db=keep_db,
            output_files=output_files,
        )
    else:
        parquet_file = db_file

    # Verify parquet file and get group sizes
    group_sizes = verify_parquet_file(parquet_file, total_rows)

    # Create memmap arrays
    memmap_arrays = {}
    for name, col_info in column_specs.items():
        dtype = np.dtype(col_info.numpy_type)
        path = os.path.join(memmap_dir, f"{name}.dat")
        memmap_arrays[name] = np.memmap(
            path, mode="w+", dtype=dtype, shape=(total_rows,)
        )

    # Calculate chunk sizes
    chunk_size = math.ceil(total_rows / num_threads)
    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, total_rows))
        for i in range(num_threads)
    ]

    # Process in parallel
    pbar = tqdm(total=total_rows, desc="Converting to memmap", leave=False, ncols=80)

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as exe:
            futures = [
                exe.submit(
                    process_parquet_chunk,
                    i,
                    parquet_file,
                    chunk_range,
                    column_specs,
                    memmap_arrays,
                    group_sizes,
                    pbar,
                )
                for i, chunk_range in enumerate(chunks)
            ]

            for f in as_completed(futures):
                f.result()

    finally:
        pbar.close()

        # Switch to read mode
        read_arrays = {}
        for name, col_info in column_specs.items():
            memmap_arrays[name].flush()
            path = os.path.join(memmap_dir, f"{name}.dat")
            read_arrays[name] = np.memmap(
                path, mode="r", dtype=np.dtype(col_info.numpy_type), shape=(total_rows,)
            )

        # Cleanup
        for arr in memmap_arrays.values():
            del arr

    if not keep_db and not skip_export:
        try:
            if os.path.isdir(parquet_file):
                # If it's a directory, remove all files in it and then the directory
                for file in os.listdir(parquet_file):
                    os.remove(os.path.join(parquet_file, file))
                os.rmdir(parquet_file)
            else:
                # If it's a single file
                os.remove(parquet_file)
        except Exception as e:
            log.warning(f"Failed to remove temporary Parquet data: {e}")

    return read_arrays


# import os
# import time
# import numpy as np
# import queue
# import logging
# from tqdm import tqdm
# from typing import Dict, Tuple, Optional
# from concurrent.futures import ThreadPoolExecutor
# import duckdb


def create_filtered_blast_table(
    db_file: str,
    input_file: str,
    temp_dir: str,
    num_threads: int,
    max_memory: Optional[int],
    evalue_threshold: float,
    bitscore_threshold: float,
) -> int:
    """Create the filtered blast table and return total row count"""
    with duckdb.connect(database=db_file) as connection:
        # Configure DuckDB with 50% of max memory for table creation
        connection.execute(f"SET threads={num_threads}")
        connection.execute(f"SET temp_directory='{temp_dir}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET enable_progress_bar=true")
        if max_memory:
            formatted_memory = set_memory_limit(max_memory, ratio=0.5)
            # connection.execute(f"SET memory_limit='{formatted_memory}'")
            connection.execute(f"SET max_memory='{formatted_memory}'")

        # Get number of columns from input file
        num_columns = len(
            connection.execute(
                f"SELECT * FROM read_csv_auto('{input_file}') LIMIT 1"
            ).description
        )
        log.info(f"Detected {num_columns} columns in input file")

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

        # Ensure all changes are committed before closing connection
        connection.commit()

    return total_rows


def create_unique_hash_view(connection):
    """Create a view with only unique row_hashes from filtered_blast"""
    connection.execute(
        """
        CREATE VIEW filtered_blast_unique AS
        SELECT * FROM filtered_blast f1
        WHERE f1.row_hash = (
            SELECT MIN(row_hash)
            FROM filtered_blast f2
            WHERE f2.row_hash = f1.row_hash
        )
        """
    )

    total_unique = connection.execute(
        "SELECT COUNT(*) FROM filtered_blast_unique"
    ).fetchone()[0]
    log.info(f"Created view with {total_unique:,} unique row_hashes")


@track_memory(name="process_input_data", detailed=True)
def process_input_data(
    input_file: str,
    temp_directories: Tuple[tempfile.TemporaryDirectory, Dict[str, str]],
    mmap_folder_dir: Optional[str] = None,
    num_threads: int = 1,
    evalue_threshold: float = 1e-5,
    bitscore_threshold: float = 50,
    max_memory: Optional[int] = None,
    keep_db: bool = False,
    output_files: Optional[Dict[str, str]] = None,
    deduplicate: bool = True,
) -> Tuple[Dict[str, np.ndarray], str]:
    """Main function to process input data and create memory-mapped arrays with deduplication

    Args:
        input_file: Path to input file (Parquet, TSV, or DuckDB)
        temp_directories: Tuple of temporary directory and subdirectories
        mmap_folder_dir: Optional directory for existing memory-mapped files
        num_threads: Number of threads to use
        evalue_threshold: E-value threshold for filtering
        bitscore_threshold: Bit score threshold for filtering
        max_memory: Maximum memory to use in bytes
        keep_db: Whether to keep the database file
        output_files: Optional dictionary of output file paths
        deduplicate: Whether to remove duplicate row_hashes

    Returns:
        Tuple of (memory_mapped_arrays, database_file_path)
    """
    temp_dir = temp_directories[0].name
    temp_subdirectories = temp_directories[1]
    db_dir = temp_subdirectories["db"]
    db_file = os.path.join(db_dir, "blast.db")

    if keep_db:
        memmap_dir = output_files["mmap"]
        # Create memmap directory if it doesn't exist
        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)
    else:
        memmap_dir = temp_subdirectories["mmap"]

    # Define column types with explicit DuckDB and NumPy type mapping
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

    log.info(f"Starting processing of input file: {input_file}")

    # Detect input type and create database/view
    input_type = detect_input_type(input_file)
    log.info(f"Detected input type: {input_type}")

    # Process input based on type
    if input_type == "parquet":
        log.info(f"Creating DuckDB database from Parquet at: {db_file}")
        with duckdb.connect(database=db_file) as connection:
            connection.execute(f"SET threads={num_threads}")
            connection.execute(f"SET temp_directory='{temp_dir}'")
            connection.execute("SET preserve_insertion_order=false")
            connection.execute("SET enable_progress_bar=true")
            if max_memory:
                formatted_memory = set_memory_limit(max_memory, ratio=0.6)
                connection.execute(f"SET memory_limit='{formatted_memory}'")
                connection.execute(f"SET max_memory='{formatted_memory}'")

            # Handle Parquet directory
            parquet_path = (
                input_file if os.path.isfile(input_file) else f"{input_file}/*.parquet"
            )

            # Create initial view
            base_view = f"""
                CREATE VIEW base_filtered_blast AS 
                SELECT * FROM parquet_scan('{parquet_path}')
            """
            connection.execute(base_view)

            # Create deduplicated view if requested
            if deduplicate:
                dedup_view = """
                    CREATE VIEW filtered_blast AS
                    SELECT DISTINCT ON (row_hash) *
                    FROM base_filtered_blast
                """
            else:
                dedup_view = """
                    CREATE VIEW filtered_blast AS
                    SELECT * FROM base_filtered_blast
                """
            connection.execute(dedup_view)

            # Get total rows
            total_rows = connection.execute(
                "SELECT COUNT(*) FROM filtered_blast"
            ).fetchone()[0]
            if deduplicate:
                original_rows = connection.execute(
                    "SELECT COUNT(*) FROM base_filtered_blast"
                ).fetchone()[0]
                log.info(
                    f"Deduplicated {original_rows:,} rows to {total_rows:,} unique rows "
                    f"(removed {original_rows - total_rows:,} duplicates)"
                )
            else:
                log.info(f"Created filtered_blast view with {total_rows:,} rows")

            connection.commit()

    elif input_type == "tsv":
        log.info(f"Creating DuckDB database at: {db_file}")
        total_rows = create_filtered_blast_table(
            db_file=db_file,
            input_file=input_file,
            temp_dir=temp_dir,
            num_threads=num_threads,
            max_memory=max_memory,
            evalue_threshold=evalue_threshold,
            bitscore_threshold=bitscore_threshold,
        )

        if deduplicate:
            with duckdb.connect(database=db_file) as connection:
                # Create base view from original table
                connection.execute(
                    """
                    CREATE VIEW base_filtered_blast AS
                    SELECT * FROM filtered_blast
                    """
                )

                # Create new table with deduplicated data
                connection.execute(
                    """
                    CREATE TABLE temp_filtered_blast AS
                    SELECT DISTINCT ON (row_hash) *
                    FROM base_filtered_blast
                    """
                )

                # Drop original table and view
                connection.execute("DROP TABLE filtered_blast")
                connection.execute("DROP VIEW base_filtered_blast")

                # Rename temp table to final table
                connection.execute(
                    "ALTER TABLE temp_filtered_blast RENAME TO filtered_blast"
                )

                # Get new total
                new_total = connection.execute(
                    "SELECT COUNT(*) FROM filtered_blast"
                ).fetchone()[0]
                log.info(
                    f"Deduplicated {total_rows:,} rows to {new_total:,} unique rows "
                    f"(removed {total_rows - new_total:,} duplicates)"
                )
                total_rows = new_total
                connection.commit()

    else:  # input_type == "duckdb"
        log.info("Using existing DuckDB database")
        db_file = input_file

        with duckdb.connect(db_file) as connection:
            if deduplicate:
                # Execute statements separately
                connection.execute(
                    """
                    CREATE VIEW base_filtered_blast AS
                    SELECT * FROM filtered_blast
                """
                )

                connection.execute("DROP TABLE filtered_blast")

                connection.execute(
                    """
                    CREATE VIEW filtered_blast AS
                    SELECT DISTINCT ON (row_hash) *
                    FROM base_filtered_blast
                """
                )

            total_rows = connection.execute(
                "SELECT COUNT(*) FROM filtered_blast"
            ).fetchone()[0]
            if deduplicate:
                # Create base view first
                connection.execute(
                    """
                    CREATE VIEW base_filtered_blast AS
                    SELECT * FROM filtered_blast
                    """
                )

                # Create new table with deduplicated data
                connection.execute(
                    """
                    CREATE TABLE temp_filtered_blast AS
                    SELECT DISTINCT ON (row_hash) *
                    FROM base_filtered_blast
                    """
                )

                # Drop original table and view
                connection.execute("DROP TABLE filtered_blast")
                connection.execute("DROP VIEW base_filtered_blast")

                # Rename temp table to final table
                connection.execute(
                    "ALTER TABLE temp_filtered_blast RENAME TO filtered_blast"
                )

    # If mmap folder is provided, load and validate existing arrays
    if mmap_folder_dir:
        log.info(f"Loading existing memory-mapped arrays from: {mmap_folder_dir}")
        memory_mapped_arrays = load_existing_mmap_arrays(
            folder_path=mmap_folder_dir,
            expected_rows=total_rows,  # Pass the row count for validation
        )
        return memory_mapped_arrays, db_file

    # Handle direct memory or new mmap creation
    if not max_memory:
        # Direct memory implementation
        log.info("Loading data into memory using optimized batch fetch")
        memory_arrays = {}
        try:
            with duckdb.connect(database=db_file) as connection:
                connection.execute(f"SET threads={num_threads}")
                select_columns = []
                for col, (duckdb_type, numpy_type) in column_data_types.items():
                    select_columns.append(f"CAST({col} AS {duckdb_type}) as {col}")

                select_sql = f"SELECT {', '.join(select_columns)} FROM filtered_blast"
                result = connection.execute(select_sql).fetchnumpy()

                for col, (_, numpy_type) in column_data_types.items():
                    memory_arrays[col] = result[col].astype(numpy_type, copy=False)

            return memory_arrays, db_file

        except Exception as e:
            log.error(f"Error during data loading: {str(e)}")
            raise

    else:
        # Memory-mapped implementation
        log.info("Creating memory-mapped arrays with batch processing")
        if input_type == "parquet" and not deduplicate:
            # Use the parquet file directly for creating mmaps only if not deduplicating
            log.info("Using existing Parquet file for memory-mapped arrays")
            memory_mapped_arrays = process_parquet_to_memmap(
                db_file=input_file,  # Original parquet file
                total_rows=total_rows,
                columns_info=column_data_types,
                memmap_dir=memmap_dir,
                temp_dir=temp_dir,
                num_threads=num_threads,
                max_memory=max_memory,
                skip_export=True,  # Skip export since we already have parquet
                keep_db=keep_db,
                output_files=output_files,
            )
        else:
            # Export to parquet and create mmaps
            memory_mapped_arrays = process_parquet_to_memmap(
                db_file=db_file,
                total_rows=total_rows,
                columns_info=column_data_types,
                memmap_dir=memmap_dir,
                temp_dir=temp_dir,
                num_threads=num_threads,
                max_memory=max_memory,
                skip_export=False,  # Need to export for non-parquet inputs or when deduplicating
                keep_db=keep_db,
                output_files=output_files,
            )
        return memory_mapped_arrays, db_file
