import os
import duckdb
from typing import Union
import gzip


def detect_input_type(filepath: str) -> str:
    """
    Detect if input file is a DuckDB database, TSV file, or Parquet file/folder.
    Returns "duckdb", "tsv", or "parquet"
    """
    # First try to open as DuckDB
    try:
        with duckdb.connect(filepath) as conn:
            # Check if filtered_blast table exists
            tables = conn.execute("SHOW TABLES").fetchall()
            if any("filtered_blast" in table[0] for table in tables):
                return "duckdb"
    except:
        pass

    # Check if it's a Parquet file/folder
    if os.path.isdir(filepath):
        if any(f.endswith(".parquet") for f in os.listdir(filepath)):
            return "parquet"
    elif filepath.endswith(".parquet"):
        return "parquet"

    # Try as TSV
    try:
        with get_open_func(filepath)(filepath, "rt") as f:
            first_line = f.readline()
            if "\t" in first_line:
                return "tsv"
    except:
        pass

    raise ValueError(
        f"Input file {filepath} is neither a valid DuckDB database with 'filtered_blast' table, "
        "a processed Parquet file/folder, nor a valid TSV file"
    )


def validate_mmap_folder(folder_path: str) -> None:
    """
    Validate that a folder contains all required memory-mapped arrays.

    Args:
        folder_path: Path to folder containing .dat files
    """
    required_columns = [
        "percIdentity",
        "alnLength",
        "subjectStart",
        "subjectEnd",
        "qlen",
        "slen",
        "subject_numeric_id",
        "query_numeric_id",
        "row_hash",
        "bitScore",
    ]

    missing_files = []
    for col in required_columns:
        file_path = os.path.join(folder_path, f"{col}.dat")
        if not os.path.isfile(file_path):
            missing_files.append(f"{col}.dat")

    if missing_files:
        raise ValueError(
            f"Missing required memory-mapped files in {folder_path}: {', '.join(missing_files)}"
        )


def get_open_func(filename: str) -> Union[gzip.open, open]:
    return gzip.open if get_compression_type(filename) == "gz" else open


def get_compression_type(filename: str) -> str:
    magic_dict = {
        "gz": (b"\x1f", b"\x8b", b"\x08"),
        "bz2": (b"\x42", b"\x5a", b"\x68"),
        "zip": (b"\x50", b"\x4b", b"\x03", b"\x04"),
    }
    max_len = max(len(x) for x in magic_dict)

    with open(filename, "rb") as unknown_file:
        file_start = unknown_file.read(max_len)

    for file_type, magic_bytes in magic_dict.items():
        if file_start.startswith(magic_bytes):
            if file_type in ["bz2", "zip"]:
                sys.exit(f"Error: cannot use {file_type} format - use gzip instead")
            return file_type
    return "plain"
