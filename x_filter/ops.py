from typing import Dict, Tuple, Optional, List, Any
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
import uuid  # Added
from x_filter.db_manager import DatabaseManager
from x_filter.resource_management import ResourceManager


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

    # Define expected dtypes for mmap files - removed row_hash
    dtypes = {
        "percIdentity": "float32",
        "alnLength": "int32",
        "subjectStart": "int32",
        "subjectEnd": "int32",
        "qlen": "int32",
        "slen": "int32",
        "subject_numeric_id": "int64",
        "query_numeric_id": "int64",
        "bitScore": "float32",
        "rowid": "int64",  # Use rowid instead of row_hash
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

    # Define expected dtypes for each column - removed row_hash
    dtypes = {
        "percIdentity": "float32",
        "alnLength": "int32",
        "subjectStart": "int32",
        "subjectEnd": "int32",
        "qlen": "int32",
        "slen": "int32",
        "subject_numeric_id": "int64",
        "query_numeric_id": "int64",
        "bitScore": "float32",
        "rowid": "int64",  # Use rowid instead of row_hash
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


def detect_tsv_headers(file_path: str) -> bool:
    """
    Detect if TSV file has column headers by checking the first row.
    
    Args:
        file_path: Path to TSV file (can be a glob pattern for directories)
        
    Returns:
        bool: True if headers are detected, False otherwise
    """
    import glob
    
    # Handle glob patterns for directories
    if '*' in file_path:
        files = glob.glob(file_path)
        if not files:
            raise ValueError(f"No files found matching pattern: {file_path}")
        # Use first file for header detection
        actual_file = files[0]
    else:
        actual_file = file_path
    
    try:
        # Read first few lines to detect headers
        if actual_file.endswith('.gz'):
            import gzip
            with gzip.open(actual_file, 'rt') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
        else:
            with open(actual_file, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
        
        if not first_line or not second_line:
            return False
            
        first_fields = first_line.split('\t')
        second_fields = second_line.split('\t')
        
        # Check if first line contains expected column names
        expected_headers = {
            'queryId', 'subjectId', 'percIdentity', 'alnLength', 'mismatchCount',
            'gapOpenCount', 'queryStart', 'queryEnd', 'subjectStart', 'subjectEnd',
            'eVal', 'bitScore', 'qlen', 'slen'
        }
        
        # Convert to lowercase for case-insensitive comparison
        first_fields_lower = [f.lower() for f in first_fields]
        
        # Check if at least 10 of the expected headers are present
        matches = sum(1 for header in expected_headers if header.lower() in first_fields_lower)
        
        # Also check if second line contains numeric data (typical for BLAST results)
        try:
            # Try to parse some numeric fields from second line
            float(second_fields[2])  # percIdentity
            int(second_fields[3])    # alnLength
            float(second_fields[10]) # eVal
            float(second_fields[11]) # bitScore
            return matches >= 10  # If we can parse numbers and have headers, it's a header file
        except (ValueError, IndexError):
            # If we can't parse second line as numbers but first line looks like headers
            return matches >= 10
            
    except Exception as e:
        log.warning(f"Could not detect headers in {actual_file}: {e}")
        return False


def create_filtered_blast_table(
    db_file: str,
    input_file: str,
    temp_dir: str,
    num_threads: int,
    max_memory: Optional[int],
    evalue_threshold: float,
    bitscore_threshold: float,
    percent_identity_threshold: float = 0.0,  # Added parameter
) -> int:
    """Create the filtered blast table and return total row count"""
    with DatabaseManager(
        database=db_file,
        temp_dir=temp_dir,
        threads=num_threads,
        memory_limit=max_memory,
        max_memory_pct=50,  # Conservative memory usage for initial loading
        enable_progress=True
    ) as db_manager:
        # Handle both single files and directories
        if os.path.isdir(input_file):
            # For directories, use glob pattern to read all TSV files
            tsv_pattern = os.path.join(input_file, "*.tsv")
            # Also check for other common extensions and compressed files
            patterns = [
                os.path.join(input_file, "*.tsv"),
                os.path.join(input_file, "*.txt"),
                os.path.join(input_file, "*.blast"),
                os.path.join(input_file, "*.m8"),
                os.path.join(input_file, "*.tsv.gz"),
                os.path.join(input_file, "*.txt.gz"),
                os.path.join(input_file, "*.blast.gz"),
                os.path.join(input_file, "*.m8.gz"),
            ]
            
            # Find which pattern has files
            csv_source = None
            for pattern in patterns:
                import glob
                if glob.glob(pattern):
                    csv_source = pattern
                    log.info(f"📁 Reading TSV files from pattern: {pattern}")
                    break
            
            if csv_source is None:
                raise ValueError(f"No TSV files found in directory: {input_file}")
        else:
            # Single file
            csv_source = input_file
            log.info(f"📄 Reading single TSV file: {csv_source}")

        # Detect if files have headers
        has_headers = detect_tsv_headers(csv_source)
        log.info(f"📋 Headers detected: {'Yes' if has_headers else 'No'}")

        # Get number of columns from input file
        read_csv_options = "header=true" if has_headers else "header=false"
        num_columns = len(
            db_manager.execute(
                f"SELECT * FROM read_csv_auto('{csv_source}', {read_csv_options}) LIMIT 1"
            ).description
        )
        log.info(f"📊 Detected {num_columns} columns in input file(s)")

        # Define column references based on whether we have headers
        if has_headers:
            # Use actual column names
            col_refs = {
                'queryId': 'queryId',
                'subjectId': 'subjectId', 
                'percIdentity': 'percIdentity',
                'alnLength': 'alnLength',
                'mismatchCount': 'mismatchCount',
                'gapOpenCount': 'gapOpenCount',
                'queryStart': 'queryStart',
                'queryEnd': 'queryEnd',
                'subjectStart': 'subjectStart',
                'subjectEnd': 'subjectEnd',
                'eVal': 'eVal',
                'bitScore': 'bitScore',
                'qlen': 'qlen',
                'slen': 'slen'
            }
            # Additional columns for 17-column format
            if num_columns >= 17:
                col_refs.update({
                    'cigar': 'cigar',
                    'qaln': 'qaln', 
                    'taln': 'taln'
                })
        else:
            # Use positional column references
            col_refs = {
                'queryId': 'column00',
                'subjectId': 'column01',
                'percIdentity': 'column02', 
                'alnLength': 'column03',
                'mismatchCount': 'column04',
                'gapOpenCount': 'column05',
                'queryStart': 'column06',
                'queryEnd': 'column07',
                'subjectStart': 'column08',
                'subjectEnd': 'column09',
                'eVal': 'column10',
                'bitScore': 'column11',
                'qlen': 'column12',
                'slen': 'column13'
            }
            # Additional columns for 17-column format  
            if num_columns >= 17:
                col_refs.update({
                    'cigar': 'column14',
                    'qaln': 'column15',
                    'taln': 'column16'
                })

        additional_columns = (
            f",\n                {col_refs['cigar']} AS cigar,\n                {col_refs['qaln']} AS qaln,\n                {col_refs['taln']} AS taln"
            if num_columns >= 17 else ""
        )

        # Create filtered table with explicit type casts - include percent identity filter
        log.info(f"🔍 Applying filters: E-value ≤ {evalue_threshold}, Bit score ≥ {bitscore_threshold}, Identity ≥ {percent_identity_threshold}%")
        
        create_table_sql = f"""
            CREATE TABLE filtered_blast AS
            WITH input_data AS (
                SELECT *,
                    CAST({col_refs['eVal']} AS DOUBLE) as evalue,
                    CAST({col_refs['bitScore']} AS FLOAT) as bitscore,
                    CAST({col_refs['percIdentity']} AS FLOAT) as percIdentity_cast
                FROM read_csv_auto('{csv_source}', parallel=true, {read_csv_options})
                WHERE CAST({col_refs['eVal']} AS DOUBLE) <= {evalue_threshold} 
                AND CAST({col_refs['bitScore']} AS FLOAT) >= {bitscore_threshold}
                AND CAST({col_refs['percIdentity']} AS FLOAT) >= {percent_identity_threshold}
            )
            SELECT
                {col_refs['queryId']} AS queryId,
                CAST(hash({col_refs['subjectId']}) % 9223372036854775807 AS BIGINT) AS subject_numeric_id,
                CAST(hash({col_refs['queryId']}) % 9223372036854775807 AS BIGINT) AS query_numeric_id,
                {col_refs['subjectId']} AS subjectId,
                CAST({col_refs['percIdentity']} AS FLOAT4) AS percIdentity,
                CAST({col_refs['alnLength']} AS INTEGER) AS alnLength,
                CAST({col_refs['mismatchCount']} AS SMALLINT) AS mismatchCount,
                CAST({col_refs['gapOpenCount']} AS SMALLINT) AS gapOpenCount,
                CAST({col_refs['queryStart']} AS INTEGER) AS queryStart,
                CAST({col_refs['queryEnd']} AS INTEGER) AS queryEnd,
                CAST({col_refs['subjectStart']} AS INTEGER) AS subjectStart,
                CAST({col_refs['subjectEnd']} AS INTEGER) AS subjectEnd,
                evalue AS eVal,
                CAST(bitscore AS FLOAT4) AS bitScore,
                CAST({col_refs['qlen']} AS INTEGER) AS qlen,
                CAST({col_refs['slen']} AS INTEGER) AS slen{additional_columns}
            FROM input_data
        """
        db_manager.execute(create_table_sql)

        # Get total rows
        total_rows = db_manager.execute(
            "SELECT COUNT(*) FROM filtered_blast"
        ).fetchone()[0]
        log.info(f"✅ Filtering complete: {total_rows:,} alignments retained")

    return total_rows


def create_filtered_blast_from_parquet(
    db_file: str,
    input_file: str,
    temp_dir: str,
    num_threads: int,
    max_memory: Optional[int],
) -> int:
    """Create the filtered blast table as a view over Parquet input"""
    with DatabaseManager(
        database=db_file,
        temp_dir=temp_dir,
        threads=num_threads,
        memory_limit=max_memory,
        max_memory_pct=60,
        enable_progress=True
    ) as db_manager:
        # Handle Parquet directory
        parquet_path = (
            input_file if os.path.isfile(input_file) else f"{input_file}/*.parquet"
        )

        # Create a view over the parquet data - don't add rowid here
        db_manager.execute(
            f"""
            CREATE VIEW filtered_blast AS 
            SELECT *
            FROM parquet_scan('{parquet_path}')
        """
        )

        # Get total rows
        total_rows = db_manager.execute(
            "SELECT COUNT(*) FROM filtered_blast"
        ).fetchone()[0]
        log.info(f"Created filtered_blast view over Parquet with {total_rows:,} rows")

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

    # Create directories with explicit permissions
    for path in temp_subdirectories.values():
        if not os.path.exists(path):
            os.makedirs(path, mode=0o755, exist_ok=True)
            log.debug(f"Created directory with permissions 755: {path}")

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
    db_manager: DatabaseManager,
    columns_info: Dict[str, ColumnInfo],
    output_dir: str,
    total_rows: int,
    chunk_size: Optional[int] = None,
    compression: str = "zstd",
    compression_level: int = 3,
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

    log.info(f"💾 Exporting to Parquet format (row groups: {chunk_size:,} rows)")
    
    db_manager.execute(export_sql)
    
    log.info(f"✅ Export complete: {output_path}")
    return output_path


def export_to_parquet_with_rowid(
    db_manager: DatabaseManager,
    columns_info: Dict[str, ColumnInfo],
    output_dir: str,
    total_rows: int,
    chunk_size: Optional[int] = None,
    compression: str = "zstd",
    compression_level: int = 3,
    keep_db: bool = False,
    output_files: Optional[Dict[str, str]] = None,
) -> str:
    """Export DuckDB table to optimized Parquet file with rowid included directly in COPY"""

    if chunk_size is None:
        # Calculate optimal size based on Parquet recommendations
        available_mem = psutil.virtual_memory().available
        chunk_size = calculate_optimal_row_group_size(
            total_rows=total_rows,
            columns_info=columns_info,
            available_memory=available_mem,
        )
    
    if keep_db:
        output_path = output_files["parquet"]
    else:
        output_path = os.path.join(output_dir, "db", "export.parquet")

    # Create column list with explicit casts and include rowid
    select_columns = []
    for col_name, col_info in columns_info.items():
        if col_name == "rowid":
            select_columns.append("rowid")
        else:
            select_columns.append(f"CAST({col_name} AS {col_info.duckdb_type}) as {col_name}")
    
    # Export directly with COPY (SELECT ...) TO ... - single scan
    export_sql = f"""
        COPY (
            SELECT {', '.join(select_columns)}
            FROM filtered_blast
        ) TO '{output_path}'
        (
            FORMAT PARQUET,
            ROW_GROUP_SIZE {chunk_size},
            COMPRESSION 'ZSTD',
            COMPRESSION_LEVEL 3,
            PER_THREAD_OUTPUT
        )
    """

    log.info(f"💾 Exporting to Parquet format (row groups: {chunk_size:,} rows)")
    
    db_manager.execute(export_sql)
    
    log.info(f"✅ Export complete: {output_path}")
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

    column_specs = {
        name: ColumnInfo.from_tuple(name, type_info)
        for name, type_info in columns_info.items()
    }

    # Export to parquet - include rowid directly in COPY statement
    if not skip_export:
        with DatabaseManager(
            database=db_file,
            temp_dir=temp_dir,
            threads=num_threads,
            memory_limit=max_memory,
            max_memory_pct=60,
            enable_progress=True
        ) as db_manager:
            # Export directly with rowid included - no temporary table needed
            parquet_file = export_to_parquet_with_rowid(
                db_manager=db_manager,
                columns_info=column_specs,
                output_dir=temp_dir,
                total_rows=total_rows,
                chunk_size=chunk_size,
                compression=compression,
                compression_level=compression_level,
                keep_db=keep_db,
                output_files=output_files,
            )
    else:
        parquet_file = db_file

    # Verify parquet file and get group sizes
    log.info("🔍 Verifying Parquet file integrity...")
    group_sizes = verify_parquet_file(parquet_file, total_rows)
    log.info(f"✅ Parquet verification complete: {len(group_sizes)} row groups")

    # Create memmap arrays
    log.info("🗂️  Creating memory-mapped arrays...")
    memmap_arrays = {}
    total_memmap_size = 0
    for name, col_info in column_specs.items():
        dtype = np.dtype(col_info.numpy_type)
        path = os.path.join(memmap_dir, f"{name}.dat")
        size = total_rows * dtype.itemsize
        total_memmap_size += size
        memmap_arrays[name] = np.memmap(
            path, mode="w+", dtype=dtype, shape=(total_rows,)
        )

    log.info(f"📊 Memory-mapped arrays: {format_memory_size(total_memmap_size)} total")

    # Calculate chunk sizes
    chunk_size = math.ceil(total_rows / num_threads)
    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, total_rows))
        for i in range(num_threads)
    ]

    # Process in parallel
    log.info(f"⚡ Converting to memory-mapped format using {num_threads} threads...")
    pbar = tqdm(
        total=total_rows, 
        desc="Converting to memmap", 
        leave=False, 
        ncols=80,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

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
        log.info("🔄 Finalizing memory-mapped arrays...")
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
            log.debug("🗑️  Cleaned up temporary Parquet files")
        except Exception as e:
            log.warning(f"Failed to remove temporary Parquet data: {e}")

    log.info(f"✅ Memory-mapped conversion complete: {len(read_arrays)} arrays ready")
    return read_arrays


def process_db_to_memmap_direct(
    db_file: str,
    total_rows: int,
    columns_info: Dict[str, Tuple[str, str]],
    memmap_dir: str,
    temp_dir: str,
    num_threads: int,
    max_memory: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, np.memmap]:
    """
    Process DuckDB table directly to memory-mapped arrays with optimized parallel processing.
    Uses rowid-based range queries instead of OFFSET/LIMIT for much better performance.
    """
    log.info("Starting optimized direct DuckDB to memmap conversion")
    
    if chunk_size is None:
        # Instantiate ResourceManager
        resource_manager = ResourceManager(max_memory=max_memory, max_threads=num_threads)
        
        # Calculate memory per row - use the numpy_type (second element of tuple)
        row_size = sum(np.dtype(type_tuple[1]).itemsize for _, type_tuple in columns_info.items())
        
        # Use ResourceManager to calculate optimal chunk size - make it larger for better performance
        chunk_size = resource_manager.calculate_optimal_chunk_size(
            total_elements=total_rows,
            element_size=row_size,
            operation_overhead=1.5,  # Reduced overhead factor
            min_chunk_size=10_000_000,  # Larger minimum chunks
            max_chunk_size=500_000_000,  # Much larger maximum chunks
        )

    log.info(f"Using optimized chunk size: {chunk_size:,} rows")
    
    # Initialize all memmap files
    memmap_arrays = {}
    total_size = 0
    for column_name, (_, numpy_type) in columns_info.items():
        memmap_file_path = os.path.join(memmap_dir, f"{column_name}.dat")
        dtype = np.dtype(numpy_type)
        total_size += total_rows * dtype.itemsize
        memmap_arrays[column_name] = np.memmap(
            memmap_file_path, mode="w+", shape=(total_rows,), dtype=dtype
        )
    
    log.info(f"Created memmap files totaling: {format_memory_size(total_size)}")
    
    # Get rowid range to avoid using OFFSET/LIMIT
    with DatabaseManager(
        database=db_file,
        temp_dir=temp_dir,
        threads=1,  # Single connection for range query
        memory_limit=max_memory,
        max_memory_pct=30,
        enable_progress=False
    ) as db_manager:
        # Get min and max rowid for range-based queries
        result = db_manager.execute("SELECT MIN(rowid), MAX(rowid) FROM filtered_blast").fetchone()
        min_rowid, max_rowid = result
        log.info(f"Rowid range: {min_rowid} to {max_rowid}")
    
    # Create rowid-based chunks instead of offset-based
    rowid_step = (max_rowid - min_rowid + 1) // num_threads
    if rowid_step == 0:
        rowid_step = 1
    
    # Create thread-safe progress tracking
    progress_lock = threading.Lock()
    total_processed = threading.Event()
    processed_rows = {"count": 0}
    
    def process_rowid_range(thread_id: int, start_rowid: int, end_rowid: int):
        """Process a range of rowids in a separate thread"""
        try:
            # Each thread gets its own database connection
            with DatabaseManager(
                database=db_file,
                temp_dir=temp_dir,
                threads=1,  # One thread per connection
                memory_limit=max_memory,
                max_memory_pct=max(20, 80 // num_threads),  # Distribute memory among threads
                enable_progress=False
            ) as thread_db:
                connection = thread_db.connection
                
                # Optimize connection for bulk reading
                connection.execute("SET preserve_insertion_order=true")
                connection.execute("SET enable_object_cache=true")
                
                # Construct SELECT with rowid range - much faster than OFFSET/LIMIT
                select_columns = []
                for col, (duckdb_type, _) in columns_info.items():
                    if col == "rowid":
                        select_columns.append("rowid")
                    else:
                        select_columns.append(f"CAST({col} AS {duckdb_type}) as {col}")
                
                # Use rowid-based WHERE clause instead of OFFSET/LIMIT
                select_sql = f"""
                    SELECT {', '.join(select_columns)}
                    FROM filtered_blast
                    WHERE rowid >= ? AND rowid < ?
                    ORDER BY rowid
                """
                
                current_rowid = start_rowid
                thread_processed = 0
                
                while current_rowid < end_rowid:
                    chunk_end = min(current_rowid + chunk_size, end_rowid)
                    
                    # Fetch chunk using rowid range
                    chunk_data = connection.execute(
                        select_sql, [current_rowid, chunk_end]
                    ).fetchnumpy()
                    
                    if len(chunk_data) == 0:
                        break
                    
                    actual_rows = len(next(iter(chunk_data.values())))
                    if actual_rows == 0:
                        break
                    
                    # Find the position to write based on rowid values
                    rowids = chunk_data['rowid']
                    write_positions = rowids - min_rowid  # Convert rowid to array index
                    
                    # Write data efficiently using advanced indexing
                    for column_name, (_, numpy_type) in columns_info.items():
                        chunk = chunk_data[column_name]
                        memmap_arrays[column_name][write_positions] = chunk
                    
                    current_rowid = chunk_end
                    thread_processed += actual_rows
                    
                    # Update global progress less frequently
                    if thread_processed % (chunk_size // 10) == 0:
                        with progress_lock:
                            processed_rows["count"] += thread_processed
                            thread_processed = 0
                
                # Final update for remaining rows
                with progress_lock:
                    processed_rows["count"] += thread_processed
                    
        except Exception as e:
            log.error(f"Error in thread {thread_id}: {str(e)}")
            raise
    
    # Create progress bar
    pbar = tqdm(
        total=total_rows,
        desc="Converting to memmap",
        unit="rows",
        unit_scale=False,
        leave=False,
    )
    
    # Start parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        for i in range(num_threads):
            start_rowid = min_rowid + i * rowid_step
            end_rowid = min_rowid + (i + 1) * rowid_step if i < num_threads - 1 else max_rowid + 1
            
            future = executor.submit(process_rowid_range, i, start_rowid, end_rowid)
            futures.append(future)
        
        # Monitor progress
        while any(not f.done() for f in futures):
            time.sleep(5)  # Update every 5 seconds
            with progress_lock:
                current_count = processed_rows["count"]
                pbar.n = current_count
                pbar.refresh()
        
        # Wait for all threads to complete
        for future in futures:
            future.result()  # This will raise any exceptions that occurred
    
    pbar.n = total_rows
    pbar.refresh()
    pbar.close()
    
    # Final flush and convert to read-only - only flush once at the end
    log.info("Finalizing memmap arrays...")
    read_memmap_arrays = {}
    for column_name, (_, numpy_type) in columns_info.items():
        memmap_file_path = os.path.join(memmap_dir, f"{column_name}.dat")
        memmap_arrays[column_name].flush()  # Single flush at the end
        del memmap_arrays[column_name]
        read_memmap_arrays[column_name] = np.memmap(
            memmap_file_path, mode="r", dtype=np.dtype(numpy_type), shape=(total_rows,)
        )
    
    log.info("Optimized direct DuckDB to memmap conversion completed")
    return read_memmap_arrays


@track_memory(name="process_input_data", detailed=True)
def process_input_data(
    input_file: str,
    temp_directories: Tuple[tempfile.TemporaryDirectory, Dict[str, str]],
    mmap_folder_dir: Optional[str] = None,
    num_threads: int = 1,
    evalue_threshold: float = 1e-5,
    bitscore_threshold: float = 50,
    percent_identity_threshold: float = 0.0,  # Added parameter
    max_memory: Optional[int] = None,
    keep_db: bool = False,
    output_files: Optional[Dict[str, str]] = None,
    use_direct_conversion: bool = False,  # New parameter to control conversion method
) -> Tuple[Dict[str, np.ndarray], str]:
    """Main function to process input data and create memory-mapped arrays"""
    temp_dir = temp_directories[0].name
    temp_subdirectories = temp_directories[1]
    db_dir = temp_subdirectories["db"]
    db_file = os.path.join(db_dir, "blast.db")

    if keep_db:
        memmap_dir = output_files["mmap"]
        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)
    else:
        memmap_dir = temp_subdirectories["mmap"]

    # Define column types - add rowid to replace row_hash
    column_data_types = {
        "percIdentity": ("FLOAT4", "float32"),
        "alnLength": ("INTEGER", "int32"),
        "subjectStart": ("INTEGER", "int32"),
        "subjectEnd": ("INTEGER", "int32"),
        "qlen": ("INTEGER", "int32"),
        "slen": ("INTEGER", "int32"),
        "subject_numeric_id": ("BIGINT", "int64"),
        "query_numeric_id": ("BIGINT", "int64"),
        "bitScore": ("FLOAT4", "float32"),
        "rowid": ("BIGINT", "int64"),  # Add rowid as replacement for row_hash
    }

    log.info(f"🚀 Starting data processing pipeline")
    log.info(f"📂 Input: {input_file}")

    # Detect input type and create database/view
    input_type = detect_input_type(input_file)
    log.info(f"🔍 Detected input type: {input_type.upper()}")

    # Process input based on type
    if input_type == "parquet":
        total_rows = create_filtered_blast_from_parquet(
            db_file=db_file,
            input_file=input_file,
            temp_dir=temp_dir,
            num_threads=num_threads,
            max_memory=max_memory,
        )
    elif input_type == "tsv":
        total_rows = create_filtered_blast_table(
            db_file=db_file,
            input_file=input_file,
            temp_dir=temp_dir,
            num_threads=num_threads,
            max_memory=max_memory,
            evalue_threshold=evalue_threshold,
            bitscore_threshold=bitscore_threshold,
            percent_identity_threshold=percent_identity_threshold,  # Added parameter
        )
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    # Choose conversion method based on dataset size and user preference
    if use_direct_conversion and total_rows > 1_000_000:
        log.info("⚡ Using direct DuckDB → memmap conversion (high performance)")
        read_arrays = process_db_to_memmap_direct(
            db_file=db_file,
            total_rows=total_rows,
            columns_info=column_data_types,
            memmap_dir=memmap_dir,
            temp_dir=temp_dir,
            num_threads=num_threads,
            max_memory=max_memory,
        )
    else:
        log.info("💾 Using Parquet intermediate format")
        # Process Parquet to memory-mapped arrays
        read_arrays = process_parquet_to_memmap(
            db_file=db_file,
            total_rows=total_rows,
            columns_info=column_data_types,
            memmap_dir=memmap_dir,
            temp_dir=temp_dir,
            num_threads=num_threads,
            max_memory=max_memory,
            keep_db=keep_db,
            output_files=output_files,
        )
    
    log.info(f"✅ Data processing complete: {len(read_arrays)} column arrays ready")
    return read_arrays, db_file


def is_debug() -> bool:
    """Check if debug mode is enabled"""
    return log.getEffectiveLevel() <= logging.DEBUG


def cleanup_temporary_files(
    temp_paths: List[str],
    keep_files: bool = False
) -> None:
    """
    Clean up temporary files and directories.
    
    Args:
        temp_paths: List of file/directory paths to clean up
        keep_files: If True, skip cleanup (for debugging)
    """
    if keep_files:
        log.info("Keeping temporary files for debugging")
        return
        
    for path in temp_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
                log.debug(f"Removed temporary file: {path}")
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
                log.debug(f"Removed temporary directory: {path}")
        except Exception as e:
            log.warning(f"Failed to remove temporary path {path}: {e}")

