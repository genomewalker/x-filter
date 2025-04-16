import os
import gc
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Union, Optional
from multiprocessing import shared_memory

import numpy as np
import tqdm
import numba
from numba import njit, prange, set_num_threads

from x_filter.logging_setup import get_logger
from x_filter.memory_tracker import track_memory
from x_filter.resource_management import ResourceManager

log = get_logger()

# Optimized cache parameters
CACHE_LINE_SIZE = 64  # bytes
L1_CACHE_SIZE = 32 * 1024  # 32KB
L2_CACHE_SIZE = 256 * 1024  # 256KB
L3_CACHE_SIZE = 8 * 1024 * 1024  # 8MB


@njit(parallel=True, fastmath=True)
def process_chunk(
    chunk_hashes: np.ndarray,
    chunk_indices: np.ndarray,
    n_buckets: int,
    sub_chunk_size: int = 10485760,  # 1MB worth of elements
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a chunk of data with optimized memory access patterns."""
    n_threads = numba.get_num_threads()
    n = len(chunk_hashes)

    # Ensure contiguous memory layout and proper alignment
    chunk_hashes = np.ascontiguousarray(chunk_hashes)
    chunk_indices = np.ascontiguousarray(chunk_indices)

    # Pre-allocate thread-local storage with padding to avoid false sharing
    cache_line_elements = CACHE_LINE_SIZE // 8  # 8 bytes per int64
    padded_buckets = (
        (n_buckets + cache_line_elements - 1) // cache_line_elements
    ) * cache_line_elements
    thread_counts = np.zeros((n_threads, padded_buckets), dtype=np.int64)

    # Process in cache-friendly sub-chunks
    for start in range(0, n, sub_chunk_size):
        end = min(start + sub_chunk_size, n)

        # Count elements per bucket per thread with vectorization hints
        for i in prange(start, end):
            thread_id = numba.get_thread_id()
            # Optimized hash computation
            hash_val = chunk_hashes[i]
            bucket_id = (hash_val ^ (hash_val >> 16) ^ (hash_val >> 32)) & (
                n_buckets - 1
            )
            thread_counts[thread_id, bucket_id] += 1

    # Calculate bucket sizes and offsets
    bucket_counts = np.sum(thread_counts[:, :n_buckets], axis=0)
    total_elements = np.sum(bucket_counts)

    # Allocate output arrays
    indices = np.empty(total_elements, dtype=np.int64)
    assignments = np.empty(total_elements, dtype=np.int64)

    # Calculate thread offsets with vectorization
    thread_offsets = np.zeros((n_threads, n_buckets), dtype=np.int64)
    offset = 0
    for bucket in range(n_buckets):
        for thread in range(n_threads):
            thread_offsets[thread, bucket] = offset
            offset += thread_counts[thread, bucket]

    # Reset thread positions with padding
    thread_positions = np.zeros((n_threads, padded_buckets), dtype=np.int64)

    # Distribute elements in parallel with cache optimization
    for i in prange(n):
        thread_id = numba.get_thread_id()
        hash_val = chunk_hashes[i]
        bucket_id = (hash_val ^ (hash_val >> 16) ^ (hash_val >> 32)) & (n_buckets - 1)
        pos = (
            thread_offsets[thread_id, bucket_id]
            + thread_positions[thread_id, bucket_id]
        )
        indices[pos] = chunk_indices[i]
        assignments[pos] = bucket_id
        thread_positions[thread_id, bucket_id] += 1

    return indices, assignments, bucket_counts


@njit(parallel=True, fastmath=True)
def find_unique_sorted(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Find unique elements with optimized sorting."""
    if len(values) == 0:
        return indices[:0]

    # Ensure contiguous memory layout
    values = np.ascontiguousarray(values)
    indices = np.ascontiguousarray(indices)

    # Sort values and indices simultaneously with optimized algorithm
    sort_idx = np.argsort(
        values, kind="mergesort"
    )  # Stable sort for better cache usage
    sorted_values = values[sort_idx]
    sorted_indices = indices[sort_idx]

    # Pre-allocate arrays
    n = len(values)
    unique_mask = np.ones(n, dtype=np.bool_)

    # Identify unique values in parallel with SIMD optimization
    for i in prange(1, n):
        unique_mask[i] = sorted_values[i] != sorted_values[i - 1]

    return sorted_indices[unique_mask]


class BucketManager:
    """Manages bucket distribution and processing."""

    def __init__(self, num_buckets: int, mmap_folder: str):
        """Initialize bucket manager."""
        self.num_buckets = num_buckets
        self.mmap_folder = mmap_folder
        self.bucket_files = []
        self.bucket_arrays = []
        self.bucket_positions = None
        self.memory_locks = []
        self._initialize()

    def _initialize(self):
        """Initialize all arrays and resources."""
        try:
            # Reset all containers
            self.bucket_files = [
                os.path.join(self.mmap_folder, f"bucket_{i}.mmap")
                for i in range(self.num_buckets)
            ]
            self.bucket_arrays = [None] * self.num_buckets
            self.bucket_positions = np.zeros(self.num_buckets, dtype=np.int64)
            self.memory_locks = [Lock() for _ in range(self.num_buckets)]

            log.info(f"Initialized {self.num_buckets} buckets")
        except Exception as e:
            log.error(f"Error in initialization: {e}")
            raise

    def create_bucket_arrays(self, bucket_sizes: np.ndarray):
        """Create memory-mapped arrays for buckets."""
        if len(bucket_sizes) != self.num_buckets:
            raise ValueError(f"Invalid bucket_sizes length: {len(bucket_sizes)}")

        # First, cleanup any existing arrays
        self.cleanup(silent=True)

        # Then create new arrays
        for i in range(self.num_buckets):
            try:
                if bucket_sizes[i] > 0:
                    self.bucket_arrays[i] = np.memmap(
                        self.bucket_files[i],
                        dtype=np.int64,
                        mode="w+",
                        shape=(bucket_sizes[i],),
                    )
                else:
                    self.bucket_arrays[i] = None
            except Exception as e:
                log.error(f"Error creating bucket {i}: {e}")
                self.bucket_arrays[i] = None

    def write_to_bucket(self, bucket_id: int, data: np.ndarray, position: int):
        """Thread-safe bucket writing."""
        if not 0 <= bucket_id < self.num_buckets:
            return

        with self.memory_locks[bucket_id]:
            try:
                bucket_array = self.bucket_arrays[bucket_id]
                if bucket_array is not None:
                    end_pos = position + len(data)
                    if end_pos <= len(bucket_array):
                        bucket_array[position:end_pos] = data
            except Exception as e:
                log.error(f"Error writing to bucket {bucket_id}: {e}")

    def get_bucket_data(self, bucket_id: int) -> Optional[np.ndarray]:
        """Safely get bucket data."""
        if not 0 <= bucket_id < self.num_buckets:
            return None

        try:
            if (
                self.bucket_arrays[bucket_id] is not None
                and self.bucket_positions[bucket_id] > 0
            ):
                return self.bucket_arrays[bucket_id][: self.bucket_positions[bucket_id]]
        except Exception as e:
            log.error(f"Error accessing bucket {bucket_id}: {e}")
        return None

    def cleanup(self, silent: bool = False):
        """Clean up bucket files and reset arrays."""
        if not silent:
            log.info(f"Starting cleanup of {self.num_buckets} buckets")

        # Store arrays to clean
        arrays_to_clean = list(self.bucket_arrays)
        files_to_clean = list(self.bucket_files)

        # Reset containers first
        self.bucket_arrays = [None] * self.num_buckets
        self.bucket_positions = np.zeros(self.num_buckets, dtype=np.int64)

        # Clean arrays
        for i, array in enumerate(arrays_to_clean):
            try:
                if array is not None:
                    array.flush()
                    del array
            except Exception as e:
                if not silent:
                    log.warning(f"Error cleaning array {i}: {e}")

        # Clean files
        for filepath in files_to_clean:
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
            except Exception as e:
                if not silent:
                    log.warning(f"Error removing file {filepath}: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup(silent=True)
        except:
            pass


def get_representative_indices(
    row_hashes: np.memmap,
    num_threads: int,
    max_memory: Union[str, int, float] = "4G",
    mmap_folder: Optional[str] = None,
    output_filename: str = "final_indices.dat",
) -> np.ndarray:
    """Find representative indices using optimized parallel processing."""
    if len(row_hashes) == 0:
        return np.array([], dtype=np.int64)

    if mmap_folder is None:
        mmap_folder = os.getcwd()
    os.makedirs(mmap_folder, exist_ok=True)

    # Initialize resource management
    resource_mgr = ResourceManager(max_memory=max_memory, max_threads=num_threads)

    try:
        # Initialize buckets
        max_num_buckets = 2**8
        num_buckets = min(max_num_buckets, len(row_hashes) // 10_000_000 + 1)
        bucket_mgr = BucketManager(num_buckets, mmap_folder)
        bucket_sizes = np.zeros(num_buckets, dtype=np.int64)

        # Calculate chunk size
        chunk_size = min(10_000_000, len(row_hashes) // (num_threads * 2))
        num_chunks = (len(row_hashes) + chunk_size - 1) // chunk_size

        # First pass: count bucket sizes
        log.info("Counting bucket sizes...")
        with tqdm.tqdm(
            total=len(row_hashes), desc="Counting", ncols=80, leave=False
        ) as pbar:
            for start in range(0, len(row_hashes), chunk_size):
                end = min(start + chunk_size, len(row_hashes))
                chunk_hashes = row_hashes[start:end].copy()
                chunk_indices = np.arange(start, end, dtype=np.int64)

                _, _, counts = process_chunk(chunk_hashes, chunk_indices, num_buckets)
                bucket_sizes += counts
                pbar.update(end - start)

        # Create bucket arrays
        bucket_mgr.create_bucket_arrays(bucket_sizes)

        # Second pass: distribute elements
        log.info("Distributing elements...")
        with tqdm.tqdm(
            total=len(row_hashes),
            desc="Distributing",
            ncols=80,
            leave=False,
        ) as pbar:
            for start in range(0, len(row_hashes), chunk_size):
                end = min(start + chunk_size, len(row_hashes))

                try:
                    # Process in smaller sub-chunks
                    sub_chunk_size = chunk_size // 10
                    for sub_start in range(start, end, sub_chunk_size):
                        sub_end = min(sub_start + sub_chunk_size, end)

                        chunk_hashes = row_hashes[sub_start:sub_end].copy()
                        chunk_indices = np.arange(sub_start, sub_end, dtype=np.int64)

                        indices, assignments, _ = process_chunk(
                            chunk_hashes, chunk_indices, num_buckets
                        )

                        # Write to buckets
                        offset = 0
                        for bucket_id in range(num_buckets):
                            mask = assignments == bucket_id
                            count = np.sum(mask)
                            if count > 0:
                                bucket_data = indices[offset : offset + count]
                                pos = bucket_mgr.bucket_positions[bucket_id]
                                bucket_mgr.write_to_bucket(bucket_id, bucket_data, pos)
                                bucket_mgr.bucket_positions[bucket_id] += count
                            offset += count

                        pbar.update(sub_end - sub_start)

                except Exception as e:
                    log.error(f"Error processing chunk {start}-{end}: {e}")

        # Find unique elements
        log.info("Finding unique elements...")
        output_path = os.path.join(mmap_folder, output_filename)
        result = np.memmap(
            output_path, dtype=np.int64, mode="w+", shape=(len(row_hashes),)
        )
        total_unique = 0

        # Process buckets
        with tqdm.tqdm(
            total=num_buckets,
            desc="Processing buckets",
            ncols=80,
            leave=False,
        ) as pbar:
            for bucket_id in range(num_buckets):
                try:
                    bucket_data = bucket_mgr.get_bucket_data(bucket_id)
                    if bucket_data is not None:
                        bucket_hashes = row_hashes[bucket_data]
                        unique_indices = find_unique_sorted(bucket_hashes, bucket_data)

                        if len(unique_indices) > 0:
                            result[
                                total_unique : total_unique + len(unique_indices)
                            ] = unique_indices
                            total_unique += len(unique_indices)
                except Exception as e:
                    log.error(f"Error processing bucket {bucket_id}: {e}")
                finally:
                    pbar.update(1)

        # Cleanup and finalize
        bucket_mgr.cleanup()

        # Trim result array
        result = np.memmap(
            output_path, dtype=np.int64, mode="r+", shape=(total_unique,)
        )

        dedup_ratio = total_unique / len(row_hashes)
        log.info(f"\nFound {total_unique:,} unique elements ({dedup_ratio:.2%} unique)")

        return result

    except Exception as e:
        log.error(f"Fatal error in get_representative_indices: {e}")
        raise
    finally:
        if "bucket_mgr" in locals():
            bucket_mgr.cleanup()


def calculate_parallel_distribution(
    chunk_data: np.ndarray, num_buckets: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate distribution of data across buckets."""
    n = len(chunk_data)
    indices = np.arange(n, dtype=np.int64)

    # Compute hash-based bucket assignments
    hash_vals = chunk_data.copy()
    bucket_ids = (hash_vals ^ (hash_vals >> 16) ^ (hash_vals >> 32)) & (num_buckets - 1)

    # Count elements per bucket
    bucket_counts = np.bincount(bucket_ids, minlength=num_buckets)

    return indices, bucket_ids, bucket_counts


def process_chunk_worker(
    args: Tuple,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Worker function for parallel chunk processing."""
    start, end, shm_name, array_length, num_buckets, num_threads = args

    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray((array_length,), dtype=np.int64, buffer=shm.buf)

    try:
        # Get chunk data
        chunk_data = shared_array[start:end]

        # Set number of threads for numba functions
        set_num_threads(num_threads)

        # Process chunk
        indices, assignments, counts = calculate_parallel_distribution(
            chunk_data, num_buckets
        )

        return indices, assignments, counts, start, end
    finally:
        # Clean up shared memory attachment
        shm.close()


def find_bucket_unique_worker(args: Tuple) -> np.ndarray:
    """Worker function for finding unique values in bucket."""
    bucket_data, shm_name, array_length, num_threads = args

    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray((array_length,), dtype=np.int64, buffer=shm.buf)

    try:
        # Set number of threads for numba operations
        set_num_threads(num_threads)

        # Get unique values from bucket
        bucket_values = shared_array[bucket_data]
        return find_unique_sorted(bucket_values, bucket_data)
    finally:
        # Clean up shared memory attachment
        shm.close()


@njit(parallel=True, cache=False, fastmath=True)
def update_coverage_array(
    flattened_coverage: np.ndarray,
    inverse_subject_indices: np.ndarray,
    subject_start_positions: np.ndarray,
    subject_end_positions: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
    n_partitions: int = 1024,  # Increased number of partitions
    num_threads: int = 1,
) -> None:
    """Optimized version for memory-mapped arrays"""
    set_num_threads(num_threads)
    n_intervals = len(inverse_subject_indices)
    total_length = len(flattened_coverage)
    partition_size = (total_length + n_partitions - 1) // n_partitions

    # Process partitions in parallel with smaller chunks
    for p_id in prange(n_partitions):
        p_start = p_id * partition_size
        p_end = min(p_start + partition_size, total_length)

        # Use a fixed small buffer size per partition
        buffer_size = 1_000_000  # 1M elements max per buffer
        # Use smaller fixed buffer per thread to improve cache utilization
        # buffer_size = min(65536, partition_size)  # Smaller buffer, better cache hits

        positions = np.empty(buffer_size, dtype=np.int64)
        changes = np.empty(buffer_size, dtype=np.int32)
        event_count = 0

        # Process intervals in chunks
        for i in range(n_intervals):
            if event_count >= buffer_size - 2:  # Leave room for both start and end
                # Sort and process current buffer
                if event_count > 0:
                    sort_idx = np.argsort(positions[:event_count])
                    for j in range(event_count):
                        pos = positions[sort_idx[j]]
                        if p_start <= pos < p_end:
                            flattened_coverage[pos] += changes[sort_idx[j]]
                event_count = 0

            subj_idx = inverse_subject_indices[i]
            abs_start = start_positions[subj_idx] + subject_start_positions[i]
            abs_end = start_positions[subj_idx] + min(
                subject_end_positions[i], subject_lengths[subj_idx] - 1
            )

            # Only process if interval intersects partition
            if abs_end >= p_start and abs_start < p_end:
                if abs_start >= p_start and abs_start < p_end:
                    positions[event_count] = abs_start
                    changes[event_count] = 1
                    event_count += 1

                if abs_end >= p_start and abs_end < p_end:
                    positions[event_count] = abs_end
                    changes[event_count] = -1
                    event_count += 1

        # Process final buffer
        if event_count > 0:
            sort_idx = np.argsort(positions[:event_count])
            for j in range(event_count):
                pos = positions[sort_idx[j]]
                if p_start <= pos < p_end:
                    flattened_coverage[pos] += changes[sort_idx[j]]


@track_memory(name="perform_cumulative_sum", detailed=True)
@njit(parallel=True, cache=True, fastmath=True)
def perform_cumulative_sum(
    flattened_coverage: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
) -> None:
    """
    Perform cumulative sum on coverage array in parallel by subjects.

    Args:
        flattened_coverage: Array containing coverage values
        start_positions: Start positions for each subject
        subject_lengths: Length of each subject
    """
    for subject_id in prange(len(start_positions)):
        start_idx = start_positions[subject_id]
        end_idx = start_positions[subject_id] + subject_lengths[subject_id]

        cumsum = 0
        for i in range(start_idx, end_idx):
            cumsum += flattened_coverage[i]
            flattened_coverage[i] = cumsum


@track_memory(name="compute_alignment_statistics", detailed=True)
@njit(parallel=True, fastmath=True, cache=True)
def compute_alignment_statistics(
    alignment_lengths: np.ndarray,
    query_lengths: np.ndarray,
    percent_identity: np.ndarray,
    inverse_indices: np.ndarray,
    n_subjects: int,
    num_threads: int = 1,
) -> Dict[str, np.ndarray]:
    """Compute alignment statistics using parallel processing."""
    chunk_size = (len(inverse_indices) + num_threads - 1) // num_threads
    results = np.zeros((num_threads, n_subjects, 7), dtype=np.float64)

    for thread_id in prange(num_threads):
        start = thread_id * chunk_size
        end = min(start + chunk_size, len(inverse_indices))

        for i in range(start, end):
            subject_idx = inverse_indices[i]
            aln_len = alignment_lengths[i]
            read_len = query_lengths[i]
            identity = percent_identity[i]

            results[thread_id, subject_idx, 0] += aln_len
            results[thread_id, subject_idx, 1] += aln_len * aln_len
            results[thread_id, subject_idx, 2] += read_len
            results[thread_id, subject_idx, 3] += read_len * read_len
            results[thread_id, subject_idx, 4] += identity
            results[thread_id, subject_idx, 5] += identity * identity
            results[thread_id, subject_idx, 6] += 1

    final_results = results.sum(axis=0)

    stats = {}
    mask = final_results[:, 6] > 0
    for idx, name in [(0, "aln_len"), (2, "read_len"), (4, "identity")]:
        mean = np.zeros(n_subjects, dtype=np.float64)
        std = np.zeros(n_subjects, dtype=np.float64)

        mean[mask] = final_results[mask, idx] / final_results[mask, 6]
        variance = np.maximum(
            final_results[mask, idx + 1] / final_results[mask, 6] - mean[mask] ** 2, 0
        )
        std[mask] = np.sqrt(variance)

        stats[f"avg_{name}"] = mean
        stats[f"std_{name}"] = std

    return stats


@track_memory(name="compute_coverage_statistics", detailed=True)
@njit(parallel=True, fastmath=True)
def compute_coverage_statistics(
    flattened_coverage: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute coverage statistics using parallel processing."""
    n_subjects = len(start_positions)
    mean_coverage = np.empty(n_subjects, dtype=np.float32)
    variance = np.empty(n_subjects, dtype=np.float32)

    for i in prange(n_subjects):
        start = start_positions[i]
        length = subject_lengths[i]
        end = start + length

        sum_values = 0.0
        sum_squares = 0.0
        for j in range(start, end):
            value = flattened_coverage[j]
            sum_values += value
            sum_squares += value * value

        mean = sum_values / length
        mean_coverage[i] = mean
        variance[i] = (sum_squares / length) - (mean * mean)

    return mean_coverage, np.sqrt(variance)


@track_memory(name="compute_global_coverage_statistics", detailed=True)
@njit(parallel=True, cache=True, fastmath=True)
def compute_global_coverage_statistics(
    flattened_coverage: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
    total_coverage: np.ndarray,
    nonzero_coverage_counts: np.ndarray,
) -> None:
    """Compute global coverage statistics using parallel processing."""
    for i in prange(len(start_positions)):
        start_idx = start_positions[i]
        end_idx = start_idx + subject_lengths[i]
        segment = flattened_coverage[start_idx:end_idx]
        total_coverage[i] = np.sum(segment)
        nonzero_coverage_counts[i] = np.count_nonzero(segment)


@track_memory(name="trim_coverage_by_subject", detailed=True)
@njit(fastmath=True, cache=True)
def trim_coverage_by_subject(
    flattened_coverage: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
    avg_alignment_lengths: np.ndarray,
    trim_multiplier: float = 2.0,
    trim_offset: int = 10,
) -> None:
    """Trim coverage at subject boundaries."""
    for i in range(len(start_positions)):
        trim_length = (
            int(np.ceil(avg_alignment_lengths[i] / 2 * trim_multiplier)) + trim_offset
        )

        if trim_length >= subject_lengths[i]:
            continue

        start_idx = start_positions[i]
        end_idx = start_idx + subject_lengths[i]

        flattened_coverage[start_idx : start_idx + trim_length] = 0
        flattened_coverage[end_idx - trim_length : end_idx] = 0


@njit(parallel=True)
def parallel_unique_sort(arr: np.ndarray) -> np.ndarray:
    """Find unique values using parallel sorting."""
    sorted_arr = np.sort(arr)
    mask = np.ones(len(sorted_arr), dtype=np.bool_)

    for i in prange(1, len(sorted_arr)):
        mask[i] = sorted_arr[i] != sorted_arr[i - 1]

    return sorted_arr[mask]


def process_chunk(chunk_data):
    """Process a single chunk using numpy operations."""
    chunk_sorted = np.sort(chunk_data)
    return chunk_sorted[np.concatenate(([True], chunk_sorted[1:] != chunk_sorted[:-1]))]


def memory_efficient_factorize(
    subject_ids: np.ndarray,
    mmap_folder: str,
    max_memory: Union[str, float, int] = "4G",
    inverse: Optional[np.ndarray] = None,
    num_threads: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Memory efficient implementation of array factorization using chunked processing."""
    log.debug("Starting parallel memory efficient factorization")

    # Initialize resource manager and get chunking strategy
    resource_mgr = ResourceManager(max_memory=max_memory, max_threads=num_threads)
    arr_info = resource_mgr.analyze_array(subject_ids)
    # Estimate concurrent chunks needed: input chunk, indices, assignments, overhead ~ 4
    num_concurrent_factorize = 4
    strategy = resource_mgr.calculate_chunk_size(arr_info, num_concurrent_chunks=num_concurrent_factorize)
    # Use a potentially larger chunk size if memory allows, but respect strategy's calculation
    # Let's stick to the strategy's calculation for consistency.
    chunk_size = strategy.chunk_size # max(100_000_000, strategy.chunk_size) # Reverted this override

    log.info(f"Using chunk size of {chunk_size:,} elements for factorization")

    # Create temporary memmap for storing intermediate unique values
    temp_unique = np.memmap(
        os.path.join(mmap_folder, "temp_unique.dat"),
        dtype=subject_ids.dtype,
        mode="w+",
        shape=(len(subject_ids),),
    )
    current_pos = 0

    # Process chunks in parallel to find unique values
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        for chunk_start in range(0, len(subject_ids), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(subject_ids))
            future = executor.submit(process_chunk, subject_ids[chunk_start:chunk_end])
            futures.append(future)

        with tqdm.tqdm(
            total=len(futures), desc="Finding unique values", ncols=80
        ) as pbar:
            for future in as_completed(futures):
                chunk_unique = future.result()
                temp_unique[current_pos : current_pos + len(chunk_unique)] = (
                    chunk_unique
                )
                current_pos += len(chunk_unique)
                temp_unique.flush()
                pbar.update(1)

    # Final unique pass on the consolidated values
    all_unique = parallel_unique_sort(temp_unique[:current_pos])

    # Clean up temporary file
    del temp_unique
    try:
        os.remove(os.path.join(mmap_folder, "temp_unique.dat"))
    except:
        pass
    gc.collect()

    log.debug(f"Found {len(all_unique):,} unique values")

    # Optimization for inverse mapping
    if inverse is None:
        inverse = np.memmap(
            os.path.join(mmap_folder, "inverse.dat"),
            dtype=np.int64,
            mode="w+",
            shape=(len(subject_ids),),
        )

    # Pre-sort unique values for binary search
    all_unique.sort()  # In-place sort

    # Process inverse mapping with Numba acceleration
    for chunk_start in tqdm.tqdm(
        range(0, len(subject_ids), chunk_size),
        desc="Creating inverse indices",
        ncols=80,
    ):
        chunk_end = min(chunk_start + chunk_size, len(subject_ids))
        chunk_length = chunk_end - chunk_start

        # Create output array for this chunk
        chunk_output = np.empty(chunk_length, dtype=np.int64)

        # Process chunk using Numba
        create_inverse_chunk(
            subject_ids[chunk_start:chunk_end],
            all_unique,
            chunk_output,
            num_threads,
        )

        # Write results back to memmap
        inverse[chunk_start:chunk_end] = chunk_output

        # Flush every 10 chunks to balance I/O
        if (chunk_start // chunk_size) % 10 == 0:
            inverse.flush()

    # Final flush
    inverse.flush()
    return inverse, all_unique


def initialize_mmap_array(
    total_positions: int,
    dtype: np.dtype,
    mmap_folder: str,
    array_name: str = "flattened_coverage",
) -> np.memmap:
    """Initialize memory-mapped array."""
    coverage_path = os.path.join(mmap_folder, f"{array_name}.dat")
    log.debug(f"Creating array of size {total_positions:,} elements")

    return np.memmap(
        coverage_path, dtype=dtype, mode="w+", shape=(total_positions,), order="C"
    )


def initialize_mmap_arrays(
    mmap_folder: str, unique_subjects: np.ndarray, max_subject_lengths: np.ndarray
) -> Tuple[np.memmap, np.memmap]:
    """Initialize memory-mapped arrays for positions and lengths."""
    log.debug("Initializing memory-mapped arrays")
    os.makedirs(mmap_folder, exist_ok=True)

    cumsum = np.concatenate(([0], np.cumsum(max_subject_lengths[:-1])))

    start_positions_file = os.path.join(mmap_folder, "start_positions.dat")
    subject_lengths_file = os.path.join(mmap_folder, "subject_lengths.dat")

    start_positions = np.memmap(
        start_positions_file, dtype=np.int64, mode="w+", shape=(len(unique_subjects),)
    )

    subject_lengths_mmap = np.memmap(
        subject_lengths_file, dtype=np.int32, mode="w+", shape=(len(unique_subjects),)
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_start = executor.submit(write_and_flush, start_positions, cumsum)
        future_lengths = executor.submit(
            write_and_flush, subject_lengths_mmap, max_subject_lengths
        )
        future_start.result()
        future_lengths.result()

    return start_positions, subject_lengths_mmap


def write_and_flush(mmap_array: np.memmap, data: np.ndarray) -> None:
    """Write data to memmap array and ensure it's flushed to disk."""
    mmap_array[:] = data
    mmap_array.flush()

    if hasattr(mmap_array, "filename"):
        try:
            fd = os.open(mmap_array.filename, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except (OSError, IOError) as e:
            log.warning(f"Failed to sync file {mmap_array.filename}: {e}")


@njit(parallel=True, cache=False)  # Disable caching since it's causing warnings
def create_inverse_chunk(chunk_data, unique_vals, output, num_threads=1):
    """Numba-accelerated chunk processing with thread control"""
    # Set number of threads for this function
    set_num_threads(num_threads)

    # Process in parallel with explicit thread control
    chunk_size = (len(chunk_data) + num_threads - 1) // num_threads

    for thread_id in prange(num_threads):
        start_idx = thread_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(chunk_data))

        for i in range(start_idx, end_idx):
            # Binary search implementation
            left, right = 0, len(unique_vals) - 1
            target = chunk_data[i]

            # Fast path for common case
            if target >= unique_vals[right]:
                output[i] = right
                continue
            elif target < unique_vals[0]:
                output[i] = 0
                continue

            # Binary search for the value
            while left <= right:
                mid = (left + right) // 2
                if unique_vals[mid] == target:
                    output[i] = mid
                    break
                elif unique_vals[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1

            # If we exited without finding an exact match, use the closest value
            if left > right:
                output[i] = left if left < len(unique_vals) else right
