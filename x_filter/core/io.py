# x_filter/core/io.py

import os
import tempfile
import duckdb
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, Optional, List, Any
from pathlib import Path

from x_filter.utils.logging import get_logger, LogContext

log = get_logger(__name__)


def setup_temporary_directory(
    base_dir: Optional[str] = None,
) -> Tuple[tempfile.TemporaryDirectory, Dict[str, str]]:
    """
    Set up temporary directory structure for processing.

    Args:
        base_dir: Base directory for temporary files

    Returns:
        Tuple of (TemporaryDirectory, dict of subdirectory paths)
    """
    with LogContext(log, "Setting up temporary directories"):
        if base_dir is None:
            base_dir = os.getcwd()

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Create temporary directory with xfilter prefix
        temp_dir = tempfile.TemporaryDirectory(dir=base_dir, prefix="xfilter-")
        temp_dir_path = temp_dir.name

        log.debug(f"Created temporary directory: {temp_dir_path}")

        # Create subdirectories
        temp_subdirectories = {
            "mmap": os.path.join(temp_dir_path, "mmap"),
            "db": os.path.join(temp_dir_path, "db"),
        }

        for path in temp_subdirectories.values():
            if not os.path.exists(path):
                os.makedirs(path)

        return temp_dir, temp_subdirectories


def cleanup_temp_files(temp_dir: Union[str, tempfile.TemporaryDirectory]) -> None:
    """
    Clean up temporary files and directories.

    Args:
        temp_dir: Temporary directory or path to clean up
    """
    with LogContext(log, "Cleaning up temporary files"):
        try:
            if isinstance(temp_dir, tempfile.TemporaryDirectory):
                temp_dir.cleanup()
            elif os.path.exists(temp_dir):
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for file in files:
                        try:
                            os.unlink(os.path.join(root, file))
                        except OSError as e:
                            log.warning(f"Failed to delete file {file}: {e}")

                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except OSError as e:
                            log.warning(f"Failed to delete directory {dir}: {e}")

                try:
                    os.rmdir(temp_dir)
                except OSError as e:
                    log.warning(f"Failed to delete temporary directory: {e}")
        except Exception as e:
            log.warning(f"Error during cleanup: {e}")


def detect_input_type(filepath: str) -> str:
    """
    Detect the type of input file.

    Args:
        filepath: Path to input file

    Returns:
        String identifying file type: "duckdb", "parquet", or "tsv"
    """
    log.debug(f"Detecting file type for {filepath}")

    # Check if it's a DuckDB database
    try:
        with duckdb.connect(filepath) as conn:
            tables = conn.execute("SHOW TABLES").fetchall()
            if any("filtered_blast" in table[0] for table in tables):
                log.debug(f"Detected DuckDB database with filtered_blast table")
                return "duckdb"
    except Exception:
        pass

    # Check if it's a Parquet file/directory
    if os.path.isdir(filepath):
        if any(f.endswith(".parquet") for f in os.listdir(filepath)):
            log.debug(f"Detected directory containing Parquet files")
            return "parquet"
    elif filepath.endswith(".parquet"):
        log.debug(f"Detected Parquet file")
        return "parquet"

    # Check if it's a TSV file (possibly compressed)
    try:
        # Try to read the first few lines
        if filepath.endswith(".gz"):
            import gzip

            with gzip.open(filepath, "rt") as f:
                first_line = f.readline()
        else:
            with open(filepath, "r") as f:
                first_line = f.readline()

        if "\t" in first_line:
            log.debug(f"Detected TSV file")
            return "tsv"
    except Exception:
        pass

    # If we can't determine the file type
    raise ValueError(
        f"Unable to determine file type for {filepath}. "
        "File must be a DuckDB database with 'filtered_blast' table, "
        "a Parquet file/directory, or a TSV file."
    )


def format_memory_limit(max_memory: Union[str, int, float], ratio: float = 1.0) -> str:
    """
    Format memory limit for DuckDB.

    Args:
        max_memory: Memory limit (string like '4G' or number of bytes)
        ratio: Ratio to apply to the memory limit

    Returns:
        Formatted memory string
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

    # Apply ratio
    adjusted_memory = int(max_memory * ratio)

    # Format according to DuckDB requirements
    if adjusted_memory >= 1024**3:  # >= 1GB
        memory_value = adjusted_memory // (1024**3)
        unit = "G"
    elif adjusted_memory >= 1024**2:  # >= 1MB
        memory_value = adjusted_memory // (1024**2)
        unit = "M"
    else:
        memory_value = adjusted_memory
        unit = "B"

    return f"{memory_value}{unit}"


def load_existing_mmap_arrays(
    folder_path: str, expected_rows: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Load existing memory-mapped arrays from a folder.

    Args:
        folder_path: Path to folder containing .dat files
        expected_rows: Optional expected number of rows for validation

    Returns:
        Dictionary of memory-mapped arrays
    """
    with LogContext(log, f"Loading memory-mapped arrays from {folder_path}"):
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        # Define expected dtypes for each column
        dtypes = {
            "percIdentity": np.float32,
            "alnLength": np.int32,
            "subjectStart": np.int32,
            "subjectEnd": np.int32,
            "qlen": np.int32,
            "slen": np.int32,
            "subject_numeric_id": np.int64,
            "query_numeric_id": np.int64,
            "row_hash": np.int64,
            "bitScore": np.float32,
        }

        # Verify all files exist
        missing_files = []
        for col in dtypes:
            file_path = os.path.join(folder_path, f"{col}.dat")
            if not os.path.isfile(file_path):
                missing_files.append(f"{col}.dat")

        if missing_files:
            raise ValueError(
                f"Missing required memory-mapped files in {folder_path}: "
                f"{', '.join(missing_files)}"
            )

        # Get total rows from first file
        first_file = os.path.join(folder_path, f"{list(dtypes.keys())[0]}.dat")
        first_dtype = dtypes[list(dtypes.keys())[0]]
        total_rows = os.path.getsize(first_file) // np.dtype(first_dtype).itemsize

        if expected_rows is not None and total_rows != expected_rows:
            raise ValueError(
                f"Memory-mapped files have {total_rows} rows "
                f"but expected {expected_rows} rows"
            )

        # Load all arrays
        mmap_arrays = {}
        for col, dtype in dtypes.items():
            file_path = os.path.join(folder_path, f"{col}.dat")
            expected_size = total_rows * np.dtype(dtype).itemsize
            actual_size = os.path.getsize(file_path)

            if actual_size != expected_size:
                raise ValueError(
                    f"Size mismatch for {col}: expected {expected_size} bytes, "
                    f"got {actual_size} bytes"
                )

            mmap_arrays[col] = np.memmap(
                file_path, dtype=dtype, mode="r", shape=(total_rows,)
            )

        log.info(
            f"Loaded {len(mmap_arrays)} memory-mapped arrays with {total_rows:,} rows"
        )
        return mmap_arrays
