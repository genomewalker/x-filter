from typing import Dict, Tuple, Optional
import duckdb
import numpy as np
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
import pyarrow.parquet as pq
import pandas as pd

from x_filter.utils import is_debug

log = logging.getLogger("my_logger")


def set_memory_limit(max_memory: int) -> str:
    if max_memory >= 1024 * 1024 * 1024:  # If max_memory is 1GB or more
        memory_value = max_memory // (1024 * 1024 * 1024)  # Integer division
        unit = "G"
    else:
        memory_value = max_memory
        unit = "B"

    return f"{memory_value}{unit}"


def setup_temporary_directory(
    base_dir: Optional[str] = None,
) -> Tuple[tempfile.TemporaryDirectory, Dict[str, str]]:
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
        "parquet": os.path.join(temp_dir_path, "parquet"),
    }

    for path in temp_subdirectories.values():
        if not os.path.exists(path):
            os.makedirs(path)

    return temp_dir, temp_subdirectories


def process_db_column_to_memmap_optimized(
    parquet_file_path: str,
    total_rows: int,
    column_name: str,
    memmap_file_path: str,
    chunk_size: int,
    dtype: np.dtype,
) -> Tuple[str, np.memmap]:
    memmap_array = np.memmap(
        memmap_file_path, mode="w+", shape=(total_rows,), dtype=dtype
    )
    parquet_file = pq.ParquetFile(parquet_file_path)

    row_index = 0
    for batch in parquet_file.iter_batches(
        batch_size=chunk_size, columns=[column_name]
    ):
        chunk_numpy = (
            batch.column(0).to_numpy(zero_copy_only=True)
            if dtype != "object"
            else batch.column(0).to_pandas().to_numpy()
        )
        memmap_array[row_index : row_index + len(chunk_numpy)] = chunk_numpy
        row_index += len(chunk_numpy)

    memmap_array.flush()
    del memmap_array

    memmap_array = np.memmap(memmap_file_path, mode="r", dtype=dtype)

    return column_name, memmap_array


def db_to_memory_mapped_arrays_optimized(
    parquet_file: str,
    mmap_folder: str,
    total_rows: int,
    chunk_size: int = 1_000_000,
    max_workers: int = 1,
    column_data_types: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, np.memmap], str]:
    memory_mapped_paths = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        with tqdm(
            total=len(column_data_types),
            desc="Saving columns",
            unit="column",
            ncols=80,
            disable=is_debug(),
            leave=False,
        ) as progress_bar:
            for column_name, dtype in column_data_types.items():
                memmap_file_path = f"{mmap_folder}/{column_name}.npy"
                futures.append(
                    executor.submit(
                        process_db_column_to_memmap_optimized,
                        parquet_file,
                        total_rows,
                        column_name,
                        memmap_file_path,
                        chunk_size,
                        dtype,
                    )
                )

            for future in as_completed(futures):
                column_name, memmap_array = future.result()
                memory_mapped_paths[column_name] = memmap_array
                progress_bar.update(1)

    # Flush all memory-mapped arrays
    for memmap_array in memory_mapped_paths.values():
        memmap_array.flush()

    # Delete references to original memmap objects
    del memory_mapped_paths

    # Reopen memory-mapped arrays
    reopened_memory_mapped_arrays = {}
    for column_name, memmap_file_path in column_data_types.items():
        reopened_memory_mapped_arrays[column_name] = np.memmap(
            f"{mmap_folder}/{column_name}.npy",
            dtype=column_data_types[column_name],
            mode="r+",
        )

    return reopened_memory_mapped_arrays, parquet_file


def process_input_data(
    input_file: str,
    temp_directories: Tuple[tempfile.TemporaryDirectory, Dict[str, str]],
    num_threads: int = 1,
    evalue_threshold: float = 1e-5,
    bitscore_threshold: float = 50,
    max_memory: Optional[int] = None,
    chunk_size: int = 5_000_000,
) -> Tuple[Dict[str, np.ndarray], str]:
    temp_dir = temp_directories[0].name
    temp_subdirectories = temp_directories[1]
    db_dir = temp_subdirectories["db"]
    parquet_dir = temp_subdirectories["parquet"]

    duckdb_column_types = {
        "queryId": "VARCHAR",
        "subjectId": "VARCHAR",
        "percIdentity": "FLOAT",
        "alnLength": "INTEGER",
        "mismatchCount": "SMALLINT",
        "gapOpenCount": "SMALLINT",
        "queryStart": "INTEGER",
        "queryEnd": "INTEGER",
        "subjectStart": "INTEGER",
        "subjectEnd": "INTEGER",
        "eVal": "DOUBLE",
        "bitScore": "FLOAT",
        "qlen": "INTEGER",
        "slen": "INTEGER",
    }

    column_data_types = {
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

    log.info("Loading and processing data")
    num_columns = len(pd.read_csv(input_file, sep="\t", header=None, nrows=1).columns)
    if num_columns == 17:
        duckdb_column_types.update(
            {"cigar": "VARCHAR", "qaln": "VARCHAR", "taln": "VARCHAR"}
        )
    elif num_columns != 14:
        raise ValueError(f"Invalid number of columns: {num_columns}")

    db_file = os.path.join(db_dir, "blast.db")
    blast_parquet_file = os.path.join(parquet_dir, "blast.parquet")
    blast_unique_parquet_file = os.path.join(parquet_dir, "filtered_uniques.parquet")

    with duckdb.connect(database=db_file) as connection:
        connection.execute(f"SET threads={num_threads};")
        connection.execute(f"SET temp_directory='{temp_dir}';")
        connection.execute("SET enable_progress_bar=true;")
        connection.execute("SET preserve_insertion_order=true;")
        connection.execute("SET force_compression='auto';")

        if max_memory:
            formatted_memory = set_memory_limit(max_memory)
            log.info(f"Setting memory limit to {formatted_memory}")
            connection.execute(f"SET memory_limit='{formatted_memory}';")
            connection.execute(f"SET max_memory = '{formatted_memory}';")

        log.info("Reading alignments")
        connection.execute(
            f"""
            COPY '{input_file}'
            TO '{blast_parquet_file}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE '100000');
        """
        )

        total_rows = pq.ParquetFile(blast_parquet_file).metadata.num_rows
        log.info(f"Number of alignments: {total_rows:,}")

        log.info(
            f"Filtering data with e-value <= {evalue_threshold} and bit-score >= {bitscore_threshold}"
        )

        additional_columns = (
            """
            (column14) AS cigar,
            (column15) AS qaln,
            (column16) AS saln,
        """
            if num_columns == 17
            else ""
        )

        hash_columns = ", ".join(
            [f"CAST(column{i:02d} AS VARCHAR)" for i in range(1, num_columns)]
        )

        connection.execute(
            f"""
                COPY (
                    SELECT
                        (column00) AS queryId,
                        hash((column01)) % 9223372036854775807 AS subject_numeric_id,
                        hash((column00)) % 9223372036854775807 AS query_numeric_id,
                        (column01) AS subjectId,
                        (column02) AS percIdentity,
                        (column03) AS alnLength,
                        (column04) AS mismatchCount,
                        (column05) AS gapOpenCount,
                        (column06) AS queryStart,
                        (column07) AS queryEnd,
                        (column08) AS subjectStart,
                        (column09) AS subjectEnd,
                        (column10) AS eVal,
                        (column11) AS bitScore,
                        (column12) AS qlen,
                        (column13) AS slen,
                        {additional_columns}
                        hash(
                            {hash_columns}
                        ) % 9223372036854775807 AS row_hash
                    FROM '{blast_parquet_file}'
                    WHERE column10 <= {evalue_threshold} AND column11 >= {bitscore_threshold}
                ) TO '{blast_unique_parquet_file}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE '100000');
                """
        )

    os.remove(blast_parquet_file)
    total_rows = pq.ParquetFile(blast_unique_parquet_file).metadata.num_rows
    log.info(f"Number of alignments: {total_rows:,}")

    if not max_memory:
        log.info("Loading data into memory")
        data = pq.read_table(blast_unique_parquet_file).to_pandas()
        numpy_arrays = {
            col: data[col].to_numpy(dtype=dtype)
            for col, dtype in column_data_types.items()
            if col in data.columns
        }
        return numpy_arrays, blast_unique_parquet_file
    else:
        log.info("Creating memory-mapped arrays")
        return db_to_memory_mapped_arrays_optimized(
            parquet_file=blast_unique_parquet_file,
            mmap_folder=temp_subdirectories["mmap"],
            chunk_size=chunk_size,
            max_workers=num_threads,
            column_data_types=column_data_types,
            total_rows=total_rows,
        )
