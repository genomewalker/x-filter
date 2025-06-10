import os
import duckdb
import sys
from typing import Union
import gzip


def detect_input_type(input_path: str) -> str:
    """
    Detect the type of input file or directory.

    Args:
        input_path: Path to input file or directory

    Returns:
        str: One of 'tsv', 'parquet', 'duckdb'

    Raises:
        ValueError: If input type cannot be determined
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    if os.path.isfile(input_path):
        # Handle single files
        if input_path.endswith(".db") or input_path.endswith(".duckdb"):
            return "duckdb"
        elif input_path.endswith(".parquet"):
            return "parquet"
        elif input_path.endswith((".tsv", ".txt", ".blast", ".m8")):
            return "tsv"
        else:
            # Try to detect by content for files without clear extensions
            try:
                with get_open_func(input_path)(input_path, "rt") as f:
                    first_line = f.readline().strip()
                    if "\t" in first_line:
                        return "tsv"
            except:
                pass
            raise ValueError(f"Cannot determine input type for file: {input_path}")

    elif os.path.isdir(input_path):
        # Handle directories
        files = os.listdir(input_path)

        # Check for parquet files
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if parquet_files:
            return "parquet"

        # Check for TSV files
        tsv_files = [f for f in files if f.endswith((".tsv", ".txt", ".blast", ".m8"))]
        if tsv_files:
            return "tsv"

        # Check for compressed TSV files
        compressed_tsv_files = [
            f for f in files if f.endswith((".tsv.gz", ".txt.gz", ".blast.gz", ".m8.gz"))
        ]
        if compressed_tsv_files:
            return "tsv"

        raise ValueError(f"No supported file types found in directory: {input_path}")

    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")


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
