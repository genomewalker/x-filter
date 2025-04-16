import os
import gc
import numpy as np
import tqdm
from typing import Union, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from numba import njit, prange
import numba
from x_filter.resource_management import ResourceManager
from x_filter.logging_setup import get_logger
import threading

log = get_logger()


@njit(parallel=True)
def fast_count_bucket_sizes(
    chunk_hashes: np.ndarray, num_buckets: int, n_threads: int = 1
) -> np.ndarray:
    """
    Count elements per bucket using thread-local counters.
    This helps determine bucket sizes before we allocate them.
    """
    numba.set_num_threads(n_threads)
    thread_counts = np.zeros((n_threads, num_buckets), dtype=np.int64)

    for i in prange(len(chunk_hashes)):
        thread_id = numba.get_thread_id()
        bucket_id = chunk_hashes[i] % num_buckets
        thread_counts[thread_id, bucket_id] += 1

    counts = np.zeros(num_buckets, dtype=np.int64)
    for t in range(n_threads):
        for b in range(num_buckets):
            counts[b] += thread_counts[t, b]

    return counts


@njit(parallel=True)
def fast_distribute_chunk(
    chunk_hashes: np.ndarray,
    chunk_indices: np.ndarray,
    num_buckets: int,
    bucket_arrays: List[np.ndarray],
    bucket_positions: np.ndarray,
) -> None:
    """
    Distribute a chunk of elements into their respective buckets in parallel.
    Empty buckets are represented as arrays with size 0 to be Numba-compatible.
    """
    local_positions = np.zeros(num_buckets, dtype=np.int64)

    for i in prange(len(chunk_hashes)):
        bucket_id = chunk_hashes[i] % num_buckets
        if bucket_arrays[bucket_id].size > 0:  # Check for non-empty buckets
            pos = bucket_positions[bucket_id] + local_positions[bucket_id]
            bucket_arrays[bucket_id][pos] = chunk_indices[i]
            local_positions[bucket_id] += 1

    for bucket_id in prange(num_buckets):
        bucket_positions[bucket_id] += local_positions[bucket_id]


def process_bucket(args: tuple) -> Tuple[np.ndarray, int]:
    """
    Unified bucket processing function.
    Uses smaller chunk sizes and np.unique for deduplication.
    Returns (unique_indices, count).
    """
    bucket_file, row_hashes_file, row_hashes_dtype, bucket_count = args

    # Adjust chunk_size based on memory constraints
    chunk_size = min(bucket_count, 1_000_000)

    unique_indices = []
    bucket = np.memmap(bucket_file, dtype=np.int64, mode="r", shape=(bucket_count,))
    row_hashes = np.memmap(row_hashes_file, dtype=row_hashes_dtype, mode="r")

    try:
        for start in range(0, bucket_count, chunk_size):
            end = min(start + chunk_size, bucket_count)
            indices_chunk = bucket[start:end]
            values_chunk = row_hashes[indices_chunk]

            # Use np.unique to find unique elements
            _, unique_inverse = np.unique(values_chunk, return_index=True)
            chunk_uniques = indices_chunk[unique_inverse]

            if len(chunk_uniques) > 0:
                unique_indices.extend(chunk_uniques)

        result = np.array(unique_indices, dtype=np.int64)
        return result, len(result)

    finally:
        if "bucket" in locals():
            del bucket
        if "row_hashes" in locals():
            del row_hashes


def resize_memmap(
    old_file: str, new_file: str, dtype: np.dtype, new_size: int, data: np.ndarray
) -> np.memmap:
    """
    Resize a memory-mapped file by creating a new file and copying the old data into it.
    """
    try:
        # Create a new memory-mapped file with the larger size
        new_array = np.memmap(new_file, dtype=dtype, mode="w+", shape=(new_size,))
        # Copy the old data into the new memory-mapped file
        new_array[: len(data)] = data
        # Flush changes to disk
        new_array.flush()
        return new_array
    except Exception as e:
        log.error(f"Error resizing memmap file: {e}")
        raise


def get_representative_indices(
    row_hashes: np.memmap,
    max_memory: Union[str, float, int] = "4G",
    mmap_folder: Optional[str] = None,
    output_filename: str = "final_indices.dat",
    return_reverse_indices: bool = False,
    prefix: str = "",
    num_threads: int = 1,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    A unified get_representative_indices function that:
    - Buckets a large array based on hashes
    - Counts bucket sizes
    - Distributes elements into buckets
    - Processes each bucket in parallel using `process_bucket`
      to find unique indices efficiently.
    - Optionally returns reverse indices if `return_reverse_indices` is True.
    - Adds a prefix to memmapped file names for better identification.

    Threads are managed via ResourceManager and applied to Numba.
    """
    if mmap_folder is None:
        mmap_folder = os.getcwd()
    os.makedirs(mmap_folder, exist_ok=True)

    # Initialize resource manager
    resource_mgr = ResourceManager(max_memory=max_memory, max_threads=num_threads)  # Use provided num_threads
    num_threads = resource_mgr.max_threads  # Update num_threads based on resource manager
    numba.set_num_threads(num_threads)

    arr_info = resource_mgr.analyze_array(row_hashes)
    # Estimate concurrent chunks for distribution phase: chunk_hashes, chunk_indices, local_bucket write ~ 3
    num_concurrent_distribute = 3
    strategy = resource_mgr.calculate_chunk_size(arr_info, num_concurrent_chunks=num_concurrent_distribute)
    chunk_size = strategy.chunk_size

    # Calculate optimal number of buckets
    min_buckets = max(num_threads * 4, 64)
    bytes_per_element = row_hashes.dtype.itemsize
    target_elements_per_bucket = max(
        chunk_size,
        min(
            len(row_hashes) // min_buckets,
            int((resource_mgr.available_memory * 0.2) // bytes_per_element),
        ),
    )

    num_buckets = max(
        min_buckets,
        min(
            len(row_hashes) // target_elements_per_bucket,
            int(np.sqrt(len(row_hashes))),
        ),
    )

    log.info(f"Processing {len(row_hashes):,} elements using {num_buckets} buckets")
    log.info(f"Target elements per bucket: {target_elements_per_bucket:,}")
    log.info(f"Using chunk size of {chunk_size:,} for distribution")
    log.info(f"Using {num_threads} threads for Numba operations")

    # First pass: count elements per bucket
    log.info("Counting elements per bucket...")
    bucket_counts = np.zeros(num_buckets, dtype=np.int64)
    # Use a potentially larger chunk size for counting as it's less memory intensive per chunk
    count_chunk_size = chunk_size * 4  # Example: 4x the distribution chunk size

    # Use the num_threads determined by ResourceManager for Numba counting
    for start in tqdm.tqdm(range(0, len(row_hashes), count_chunk_size), ncols=80, desc="Counting buckets"):
        end = min(start + count_chunk_size, len(row_hashes))
        chunk_hashes = row_hashes[start:end]
        # Pass num_threads to the Numba function
        chunk_counts = fast_count_bucket_sizes(chunk_hashes, num_buckets, n_threads=num_threads)
        bucket_counts += chunk_counts

    # Create bucket files
    log.info("Creating bucket arrays...")
    bucket_files = [
        (
            os.path.join(mmap_folder, f"{prefix}bucket_{i}.mmap")
            if bucket_counts[i] > 0
            else None
        )
        for i in range(num_buckets)
    ]
    bucket_arrays = [
        (
            np.memmap(
                bucket_files[i], dtype=np.int64, mode="w+", shape=(bucket_counts[i],)
            )
            if bucket_files[i] is not None
            else None
        )
        for i in range(num_buckets)
    ]

    # Second pass: distribute elements to local buckets and merge
    log.info("Distributing elements to buckets in parallel...")

    local_buckets = [
        [np.empty(bucket_counts[i], dtype=np.int64) for i in range(num_buckets)]
        for _ in range(num_threads)
    ]
    local_positions = [
        np.zeros(num_buckets, dtype=np.int64) for _ in range(num_threads)
    ]

    def distribute_chunk(start: int, end: int, thread_id: int):
        chunk_hashes = row_hashes[start:end]
        chunk_indices = np.arange(start, end, dtype=np.int64)
        local_bucket = local_buckets[thread_id]
        local_pos = local_positions[thread_id]

        for i in range(len(chunk_hashes)):
            bucket_id = chunk_hashes[i] % num_buckets
            pos = local_pos[bucket_id]
            local_bucket[bucket_id][pos] = chunk_indices[i]
            local_pos[bucket_id] += 1

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                distribute_chunk,
                start,
                min(start + chunk_size, len(row_hashes)),
                thread_id,
            )
            for thread_id, start in enumerate(range(0, len(row_hashes), chunk_size))
        ]
        for future in tqdm.tqdm(futures, ncols=80):
            future.result()

    # Merge local buckets into global buckets
    log.info("Merging local buckets...")
    bucket_positions = np.zeros(num_buckets, dtype=np.int64)

    for thread_id in range(num_threads):
        for bucket_id in range(num_buckets):
            local_bucket = local_buckets[thread_id][bucket_id]
            local_pos = local_positions[thread_id][bucket_id]

            if local_pos > 0:
                start = bucket_positions[bucket_id]
                end = start + local_pos
                bucket_arrays[bucket_id][start:end] = local_bucket[:local_pos]
                bucket_positions[bucket_id] = end

    # Verify bucket sizes
    log.info("Verifying bucket sizes after distribution...")
    for i in range(num_buckets):
        if bucket_arrays[i] is not None:
            assert (
                bucket_positions[i] == bucket_counts[i]
            ), f"Bucket {i} size mismatch: Expected {bucket_counts[i]}, Found {bucket_positions[i]}"

    # Flush and close bucket arrays
    for bucket_array in bucket_arrays:
        if bucket_array is not None:
            bucket_array.flush()
    del bucket_arrays
    gc.collect()

    # Process buckets in parallel using processes
    total_unique = 0
    output_path = os.path.join(mmap_folder, f"{prefix}{output_filename}")
    result = np.memmap(output_path, dtype=np.int64, mode="w+", shape=(len(row_hashes),))

    log.info("Processing buckets in parallel...")
    bucket_args = [
        (bucket_files[i], row_hashes.filename, row_hashes.dtype, bucket_counts[i])
        for i in range(num_buckets)
        if bucket_counts[i] > 0
    ]

    temp_merged_file = os.path.join(mmap_folder, f"{prefix}temp_merged_unique.mmap")
    max_memory_bytes = resource_mgr.available_memory * 0.8
    merge_chunk_size = int(max_memory_bytes // row_hashes.dtype.itemsize)

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for args in bucket_args:
            futures.append(executor.submit(process_bucket, args))

        temp_merged = np.memmap(
            temp_merged_file,
            dtype=row_hashes.dtype,
            mode="w+",
            shape=(merge_chunk_size,),
        )
        pos = 0

        for future in tqdm.tqdm(futures, ncols=80):
            unique_indices, unique_count = future.result()
            if unique_count > 0:
                if pos + unique_count > temp_merged.size:
                    temp_merged.flush()
                    temp_merged = resize_memmap(
                        temp_merged_file,
                        temp_merged_file,
                        row_hashes.dtype,
                        temp_merged.size * 2,
                        temp_merged[:pos],
                    )

                temp_merged[pos : pos + unique_count] = unique_indices
                pos += unique_count

        merged_unique = temp_merged[:pos]
        merged_unique.sort()
        temp_merged.flush()

    if return_reverse_indices:
        log.info("Calculating reverse indices...")

        # Process in chunks to avoid memory issues
        reverse_indices = np.memmap(
            os.path.join(mmap_folder, f"{prefix}reverse_indices.mmap"),
            dtype=np.int64,
            mode="w+",
            shape=(len(row_hashes),),
        )

        def process_chunk(start: int, end: int):
            chunk = row_hashes[start:end]
            reverse_indices[start:end] = np.searchsorted(merged_unique, chunk)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for start in range(0, len(row_hashes), chunk_size):
                end = min(start + chunk_size, len(row_hashes))
                futures.append(executor.submit(process_chunk, start, end))

            for future in tqdm.tqdm(futures, ncols=80):
                future.result()

        reverse_indices.flush()
        return merged_unique, reverse_indices

    # Clean up bucket files
    for bucket_file in bucket_files:
        if bucket_file is not None:
            try:
                os.unlink(bucket_file)
            except OSError:
                pass

    os.unlink(temp_merged_file)
    log.info(f"Found {len(merged_unique):,} unique elements")
    return merged_unique
