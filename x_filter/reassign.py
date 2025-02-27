import numpy as np
import pandas as pd
import logging
import os
import gc
from contextlib import contextmanager
from typing import List, Tuple, Dict, Any, Generator, Union, Optional
from tqdm import tqdm
from numba import njit, prange
from x_filter.logging_setup import get_logger
from x_filter.resource_management import ResourceManager
import numba
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from x_filter.core_processing import (
    BucketManager,
    # calculate_parallel_distribution,
    # process_chunk_worker,
    # find_bucket_unique_worker,
    memory_efficient_factorize,
    initialize_mmap_array,
)
from multiprocessing import Pool, Lock, shared_memory

log = get_logger()


@contextmanager
def temp_memmap(
    filename: str, dtype: np.dtype, mode: str, shape: Tuple[int, ...]
) -> Generator[np.memmap, None, None]:
    """Enhanced temporary memory-mapped array manager."""
    memmap_array = None
    try:
        memmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        yield memmap_array
    finally:
        if memmap_array is not None:
            try:
                memmap_array.flush()
            except Exception as e:
                log.warning(f"Error flushing memmap: {e}")
            del memmap_array
        try:
            if os.path.exists(filename):
                os.unlink(filename)
        except OSError as e:
            log.warning(f"Error deleting file {filename}: {e}")


@njit(parallel=True)
def parallel_unique_sort(arr: np.ndarray) -> np.ndarray:
    """Optimized parallel implementation for finding unique sorted values."""
    if len(arr) == 0:
        return arr
    sorted_arr = np.sort(arr)
    mask = np.ones(len(sorted_arr), dtype=np.bool_)
    for i in prange(1, len(sorted_arr)):
        if sorted_arr[i] == sorted_arr[i - 1]:
            mask[i] = False
    return sorted_arr[mask]


@njit(parallel=True)
def merge_sorted_unique(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Merge two sorted arrays maintaining uniqueness."""
    if len(arr1) == 0:
        return arr2
    if len(arr2) == 0:
        return arr1

    result = np.empty(len(arr1) + len(arr2), dtype=arr1.dtype)
    i = j = k = 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            if k == 0 or result[k - 1] != arr1[i]:
                result[k] = arr1[i]
                k += 1
            i += 1
        else:
            if k == 0 or result[k - 1] != arr2[j]:
                result[k] = arr2[j]
                k += 1
            j += 1

    while i < len(arr1):
        if k == 0 or result[k - 1] != arr1[i]:
            result[k] = arr1[i]
            k += 1
        i += 1

    while j < len(arr2):
        if k == 0 or result[k - 1] != arr2[j]:
            result[k] = arr2[j]
            k += 1
        j += 1

    return result[:k]


@njit(parallel=True)
def create_inverse_indices_parallel(
    chunk_data: np.ndarray,
    unique_values: np.ndarray,
    output: np.ndarray,
    start_idx: int,
) -> None:
    """Create inverse indices for chunk in parallel using binary search."""
    for i in prange(len(chunk_data)):
        value = chunk_data[i]
        # Binary search
        left, right = 0, len(unique_values)
        while left < right:
            mid = (left + right) // 2
            if unique_values[mid] == value:
                output[start_idx + i] = mid
                break
            elif unique_values[mid] < value:
                left = mid + 1
            else:
                right = mid


def resize_memmap(
    old_file: str, new_file: str, dtype: np.dtype, new_size: int, data: np.ndarray
) -> np.memmap:
    """Safely resize a memory-mapped array."""
    try:
        # Create new memmap with larger size
        new_array = np.memmap(new_file, dtype=dtype, mode="w+", shape=(new_size,))
        # Copy existing data
        new_array[: len(data)] = data
        # Flush to ensure data is written
        new_array.flush()
        return new_array
    except Exception as e:
        log.error(f"Error during resize: {e}")
        raise


def process_unique_values(
    input_array: np.memmap,
    mmap_folder: str,
    prefix: str,
    resource_manager: ResourceManager,
) -> Tuple[np.memmap, np.memmap]:
    """Process array to get unique values and their indices efficiently."""
    # Get optimal chunking strategy
    arr_info = resource_manager.analyze_array(input_array)
    strategy = resource_manager.calculate_chunk_size(arr_info)
    chunk_size = strategy.chunk_size

    # Create temporary files for results
    unique_values_file = os.path.join(mmap_folder, f"{prefix}_unique_values.dat")
    inverse_indices_file = os.path.join(mmap_folder, f"{prefix}_inverse_indices.dat")
    temp_merge_file = os.path.join(mmap_folder, f"{prefix}_temp_merge.dat")

    try:
        # Start with a larger initial size (50% of input size)
        initial_size = max(
            chunk_size,
            min(
                len(input_array) // 2,
                int(
                    (resource_manager.available_memory * 0.3)
                    // input_array.dtype.itemsize
                ),
            ),
        )

        log.info(f"Initial allocation size for {prefix}: {initial_size:,} elements")

        # Initialize unique values array
        unique_values = np.memmap(
            unique_values_file,
            dtype=input_array.dtype,
            mode="w+",
            shape=(initial_size,),
        )

        # Process in chunks with streaming merge
        current_unique_count = 0
        with tqdm(
            total=len(input_array), desc=f"Processing {prefix}", ncols=80
        ) as pbar:
            for start in range(0, len(input_array), chunk_size):
                end = min(start + chunk_size, len(input_array))

                # Get chunk and process
                chunk = input_array[start:end]
                chunk_uniques = parallel_unique_sort(chunk)

                # Merge with existing unique values
                if current_unique_count > 0:
                    existing_uniques = unique_values[:current_unique_count]
                    merged = merge_sorted_unique(existing_uniques, chunk_uniques)

                    # Resize if needed
                    if len(merged) > len(unique_values):
                        # Calculate new size with extra padding
                        growth_factor = 1.5
                        new_size = min(
                            int(len(merged) * growth_factor), len(input_array)
                        )

                        log.info(f"Resizing {prefix} array to {new_size:,} elements")

                        # Create new array with larger size
                        unique_values = resize_memmap(
                            unique_values_file,
                            temp_merge_file,
                            input_array.dtype,
                            new_size,
                            merged,
                        )

                        # Clean up and rename
                        os.rename(temp_merge_file, unique_values_file)

                    else:
                        unique_values[: len(merged)] = merged

                    current_unique_count = len(merged)
                else:
                    # First chunk
                    if len(chunk_uniques) > len(unique_values):
                        # Resize if initial size was too small
                        new_size = min(int(len(chunk_uniques) * 1.5), len(input_array))
                        unique_values = resize_memmap(
                            unique_values_file,
                            temp_merge_file,
                            input_array.dtype,
                            new_size,
                            chunk_uniques,
                        )
                        os.rename(temp_merge_file, unique_values_file)
                    else:
                        unique_values[: len(chunk_uniques)] = chunk_uniques
                    current_unique_count = len(chunk_uniques)

                pbar.update(end - start)
                gc.collect()

        # Trim to actual size
        actual_size = current_unique_count
        log.info(f"Final unique {prefix} count: {actual_size:,}")

        os.truncate(unique_values_file, actual_size * unique_values.dtype.itemsize)

        # Reopen with correct size
        unique_values = np.memmap(
            unique_values_file, dtype=input_array.dtype, mode="r+", shape=(actual_size,)
        )

        # Create inverse indices array
        inverse_indices = np.memmap(
            inverse_indices_file, dtype=np.int64, mode="w+", shape=(len(input_array),)
        )

        # Create inverse indices in parallel chunks
        with tqdm(
            total=len(input_array), desc=f"Creating {prefix} indices", ncols=80
        ) as pbar:
            for start in range(0, len(input_array), chunk_size):
                end = min(start + chunk_size, len(input_array))
                chunk = input_array[start:end]
                create_inverse_indices_parallel(
                    chunk, unique_values, inverse_indices, start
                )
                pbar.update(end - start)

        return unique_values, inverse_indices

    except Exception as e:
        log.error(f"Error processing {prefix}: {str(e)}")
        raise


@njit(parallel=True)
def parallel_accumulate_weights(
    chunk_indices: np.ndarray, chunk_scores: np.ndarray, output: np.ndarray
) -> None:
    """Accumulate weights in parallel, equivalent to np.add.at"""
    # Create thread-local accumulators to avoid race conditions
    n_threads = numba.get_num_threads()
    local_outputs = np.zeros((n_threads, len(output)), dtype=output.dtype)

    # Parallel accumulation into thread-local arrays
    for i in prange(len(chunk_indices)):
        thread_id = numba.get_thread_id()
        idx = chunk_indices[i]
        local_outputs[thread_id, idx] += chunk_scores[i]

    # Sequential reduction of thread-local results into output
    for i in range(n_threads):
        for j in range(len(output)):
            if local_outputs[i, j] != 0:
                output[j] += local_outputs[i, j]


def chunked_initialize_weights(
    subject_inverse_indices: np.memmap,
    bitScore: np.memmap,
    max_index: int,
    mmap_folder: str,
    resource_manager: ResourceManager,
) -> np.memmap:
    """Initialize weights using chunked processing with resource management.
    Mathematically equivalent to original implementation but with parallel processing.
    """
    # Get chunking strategy from resource manager
    arr_info = resource_manager.analyze_array(subject_inverse_indices)
    strategy = resource_manager.calculate_chunk_size(arr_info)
    chunk_size = strategy.chunk_size

    # Create memory-mapped array for total weights
    total_weights_file = os.path.join(mmap_folder, "total_weights.dat")
    total_weights = np.memmap(
        total_weights_file, dtype=np.float64, mode="w+", shape=(max_index,)
    )
    total_weights.fill(0)

    try:
        # Process in chunks, mathematically equivalent to np.add.at
        with tqdm(
            total=len(subject_inverse_indices), desc="Calculating weights", ncols=80
        ) as pbar:
            for start in range(0, len(subject_inverse_indices), chunk_size):
                end = min(start + chunk_size, len(subject_inverse_indices))

                # Get chunk data
                chunk_indices = subject_inverse_indices[start:end]
                chunk_scores = bitScore[start:end]

                # Accumulate weights in parallel
                parallel_accumulate_weights(chunk_indices, chunk_scores, total_weights)
                pbar.update(end - start)

        # Handle zero weights exactly as original
        total_weights[total_weights == 0] = np.finfo(np.float64).tiny

        # Create result array
        result_file = os.path.join(mmap_folder, "weights_result.dat")
        result = np.memmap(
            result_file, dtype=np.float64, mode="w+", shape=bitScore.shape
        )

        # Calculate final weights in chunks, preserving original math
        with tqdm(
            total=len(subject_inverse_indices),
            desc="Calculating final weights",
            ncols=80,
        ) as pbar:
            for start in range(0, len(subject_inverse_indices), chunk_size):
                end = min(start + chunk_size, len(bitScore))
                chunk_indices = subject_inverse_indices[start:end]
                chunk_scores = bitScore[start:end]
                result[start:end] = chunk_scores / total_weights[chunk_indices]
                pbar.update(end - start)
        # for start in range(0, len(bitScore), chunk_size):

        return result

    finally:
        # Cleanup
        try:
            if os.path.exists(total_weights_file):
                os.unlink(total_weights_file)
        except OSError:
            pass


def validate_probabilities(
    prob: np.ndarray, query_indices: np.ndarray, max_query: int
) -> bool:
    """Validate probability array for numerical stability."""
    if np.any(prob < 0):
        return False

    prob_sum = np.zeros(max_query + 1, dtype=np.float64)
    np.add.at(prob_sum, query_indices, prob)

    return not np.any(prob_sum == 0)


def chunked_fixed_point_map(
    input_prob: np.memmap,
    mask: np.memmap,
    slen: np.memmap,
    query_inverse_indices: np.memmap,
    max_query: int,
    mmap_folder: str,
    resource_manager: ResourceManager,
) -> np.memmap:
    """Process fixed point mapping using chunked processing."""
    # Get chunking strategy from resource manager
    arr_info = resource_manager.analyze_array(input_prob)
    strategy = resource_manager.calculate_chunk_size(arr_info)
    chunk_size = strategy.chunk_size

    new_prob_file = os.path.join(mmap_folder, "new_prob_temp.mmap")

    try:
        new_prob = np.memmap(
            new_prob_file, dtype=np.float64, mode="w+", shape=input_prob.shape
        )
        new_prob[:] = input_prob[:]

        masked_prob = input_prob[mask].copy()
        masked_slen = slen[mask].copy()
        masked_slen[masked_slen == 0] = np.finfo(np.float64).tiny

        s_w = masked_prob / masked_slen
        new_prob[mask] = masked_prob * s_w

        prob_sum = np.zeros(max_query + 1, dtype=np.float64)

        for start in range(0, len(mask), chunk_size):
            end = min(start + chunk_size, len(mask))
            chunk_mask = mask[start:end]
            if not np.any(chunk_mask):
                continue
            chunk_queries = query_inverse_indices[start:end][chunk_mask]
            chunk_probs = new_prob[start:end][chunk_mask]
            np.add.at(prob_sum, chunk_queries, chunk_probs)

        mask_sum = prob_sum[query_inverse_indices[mask]]
        mask_sum[mask_sum == 0] = np.finfo(np.float64).tiny
        new_prob[mask] = new_prob[mask] / mask_sum

        return new_prob

    finally:
        try:
            if os.path.exists(new_prob_file):
                os.unlink(new_prob_file)
        except OSError:
            pass


def chunked_squarem_step(
    prob: np.memmap,
    mask: np.memmap,
    slen: np.memmap,
    query_inverse_indices: np.memmap,
    max_query: int,
    mmap_folder: str,
    step_min: float = -1.0,
    step_max: float = -1.0,
    mstep: int = 4,
    resource_manager: Optional[
        ResourceManager
    ] = None,  # Make resource_manager last optional parameter
) -> np.ndarray:
    """SQUAREM implementation using chunked processing."""
    q = chunked_fixed_point_map(
        prob,
        mask,
        slen,
        query_inverse_indices,
        max_query,
        mmap_folder,
        resource_manager,
    )
    r = q - prob
    sr2 = (r**2).sum()

    if sr2 < 1e-10:
        return q

    q2 = chunked_fixed_point_map(
        q, mask, slen, query_inverse_indices, max_query, mmap_folder, resource_manager
    )
    r2 = q2 - q
    v = r2 - r
    sv2 = (v**2).sum()
    srv = (r * v).sum()

    if sv2 < 1e-10:
        return q2

    alpha = np.sqrt(sr2 / sv2)
    alpha = np.clip(alpha, step_min, step_max)

    p_new_file = os.path.join(mmap_folder, "p_new_temp.mmap")

    try:
        p_new = np.memmap(p_new_file, dtype=np.float64, mode="w+", shape=prob.shape)
        p_new[:] = prob + 2 * alpha * r + alpha**2 * v

        if validate_probabilities(p_new[mask], query_inverse_indices[mask], max_query):
            return chunked_fixed_point_map(
                p_new,
                mask,
                slen,
                query_inverse_indices,
                max_query,
                mmap_folder,
                resource_manager,
            )

        for m in range(mstep):
            alpha = alpha / 2
            p_new[:] = prob + 2 * alpha * r + alpha**2 * v
            if validate_probabilities(
                p_new[mask], query_inverse_indices[mask], max_query
            ):
                return chunked_fixed_point_map(
                    p_new,
                    mask,
                    slen,
                    query_inverse_indices,
                    max_query,
                    mmap_folder,
                    resource_manager,
                )

        return q2

    finally:
        try:
            if os.path.exists(p_new_file):
                os.unlink(p_new_file)
        except OSError:
            pass


def resolve_multimaps_return_indices(
    subject_inverse_indices: np.memmap,
    query_inverse_indices: np.memmap,
    prob: np.memmap,
    slen: np.memmap,
    iter_array: np.memmap,
    mmap_folder: str,
    iters: int = 10,
    step_min: float = -1.0,
    step_max: float = -1.0,
    mstep: int = 4,
    scale: float = 0.9,
    resource_manager: Optional[
        ResourceManager
    ] = None,  # Add resource_manager parameter
) -> np.ndarray:
    """Resolve multimapped reads using chunked processing."""
    mask = np.ones(subject_inverse_indices.shape, dtype=bool)
    total_reads = len(np.unique(query_inverse_indices))
    max_query = query_inverse_indices.max()
    current_iter = 0
    prev_num_alignments = np.inf

    log.info(
        f"Starting multimap resolution: {iters} iterations"
        if iters > 0
        else "Resolving multimaps until convergence"
    )
    log.info(f"Initial alignments: {mask.sum():,}")

    prob_working_file = os.path.join(mmap_folder, "prob_working.mmap")
    prob_working = np.memmap(
        prob_working_file, dtype=np.float64, mode="w+", shape=prob.shape
    )
    prob_working[:] = prob[:]

    try:
        while iters == 0 or current_iter < iters:
            n_alns = mask.sum()
            if n_alns == prev_num_alignments:
                log.info("Convergence reached - no more alignments removed")
                break

            prev_num_alignments = n_alns

            with tqdm(total=5, desc=f"Iteration {current_iter + 1}", ncols=80) as pbar:
                # SQUAREM update
                prob_working = chunked_squarem_step(
                    prob_working,
                    mask,
                    slen,
                    query_inverse_indices,
                    max_query,
                    mmap_folder,
                    step_min,
                    step_max,
                    mstep,
                    resource_manager=resource_manager,  # Pass resource_manager
                )
                pbar.update(1)

                # Calculate alignments per read
                n_aln = np.zeros(max_query + 1, dtype=np.int64)
                np.add.at(n_aln, query_inverse_indices[mask], 1)
                n_aln_per_read = n_aln[query_inverse_indices]
                pbar.update(1)

                # Rest of the implementation remains the same
                unique_mask = n_aln_per_read == 1
                non_unique_mask = n_aln_per_read > 1
                unique_mask &= mask
                non_unique_mask &= mask

                if unique_mask.all():
                    log.info("All reads uniquely mapped - stopping early")
                    break

                max_prob = np.zeros(max_query + 1, dtype=np.float64)
                np.maximum.at(max_prob, query_inverse_indices[mask], prob_working[mask])
                max_prob_scaled = max_prob[query_inverse_indices] * scale
                pbar.update(1)

                final_mask = (prob_working >= max_prob_scaled) & non_unique_mask
                pbar.update(1)

                iter_array[final_mask] = current_iter + 1
                mask &= unique_mask | final_mask
                pbar.update(1)

            global_uniques = unique_mask.sum()
            reads_to_process = total_reads - global_uniques
            log.info(
                f"Iteration {current_iter + 1}: Alignments={n_alns:,} | "
                f"Unique={global_uniques:,} | Remaining={reads_to_process:,}"
            )

            if mask.sum() == 0:
                log.info("All alignments processed - stopping")
                break

            current_iter += 1

        if iters > 0 and current_iter == iters:
            log.info(f"Reached maximum iterations ({iters})")

        return mask.copy()

    finally:
        try:
            if os.path.exists(prob_working_file):
                os.unlink(prob_working_file)
        except OSError:
            pass


def bucket_process_unique_values(
    input_array: np.memmap,
    mmap_folder: str,
    prefix: str,
    resource_manager: ResourceManager,
) -> Tuple[np.memmap, np.memmap]:
    """Process array to get unique values using process pool parallelism."""
    if len(input_array) == 0:
        return np.array([], dtype=input_array.dtype), np.array([], dtype=np.int64)

    os.makedirs(mmap_folder, exist_ok=True)

    # Calculate optimal process and thread distribution
    total_cores = resource_manager.max_threads
    num_processes = max(2, total_cores // 2)  # Use half cores for processes
    num_threads = 2  # Keep threads per process fixed at 2
    log.info(f"CPU cores available: {total_cores}")
    log.info(f"Using {num_processes} processes with {num_threads} threads each")

    # Initialize parameters
    num_buckets = min(2**8, len(input_array) // 1_000_000 + 1)
    if num_buckets < num_processes:
        num_buckets = num_processes
    bucket_mgr = BucketManager(num_buckets, mmap_folder)

    try:
        # Create shared memory for input array
        shm = shared_memory.SharedMemory(create=True, size=input_array.nbytes)
        shared_array = np.ndarray(
            input_array.shape, dtype=input_array.dtype, buffer=shm.buf
        )
        shared_array[:] = input_array[:]

        # Calculate chunk size
        chunk_size = min(1_000_000, len(input_array) // (num_processes * 2))
        chunks = [
            (i, min(i + chunk_size, len(input_array)))
            for i in range(0, len(input_array), chunk_size)
        ]

        # First pass: count bucket sizes using process pool
        bucket_sizes = np.zeros(num_buckets, dtype=np.int64)

        with Pool(processes=num_processes) as pool:
            with tqdm(total=len(input_array), desc="Counting", ncols=80) as pbar:
                worker_args = [
                    (start, end, shm.name, len(input_array), num_buckets, num_threads)
                    for start, end in chunks
                ]

                for indices, assignments, counts, start, end in pool.imap(
                    process_chunk_worker, worker_args
                ):
                    bucket_sizes += counts
                    pbar.update(end - start)

        # Create bucket arrays
        bucket_mgr.create_bucket_arrays(bucket_sizes)

        # Second pass: distribute elements using process pool
        with Pool(processes=num_processes) as pool:
            with tqdm(total=len(input_array), desc="Distributing", ncols=80) as pbar:
                worker_args = [
                    (start, end, shm.name, len(input_array), num_buckets, num_threads)
                    for start, end in chunks
                ]

                for indices, assignments, counts, start, end in pool.imap(
                    process_chunk_worker, worker_args
                ):
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
                    pbar.update(end - start)

        # Process buckets in parallel for finding uniques
        log.info("Finding unique elements with parallel processing...")
        unique_values_file = os.path.join(mmap_folder, f"{prefix}_unique_values.dat")
        result = np.memmap(
            unique_values_file,
            dtype=input_array.dtype,
            mode="w+",
            shape=(len(input_array),),
        )
        total_unique = 0

        # Get active buckets
        active_buckets = [
            (bucket_mgr.get_bucket_data(i), i)
            for i in range(num_buckets)
            if bucket_mgr.get_bucket_data(i) is not None
        ]

        # Process buckets in parallel using process pool
        with Pool(processes=num_processes) as pool:
            with tqdm(
                total=len(active_buckets), desc="Processing buckets", ncols=80
            ) as pbar:
                worker_args = [
                    (bucket_data, shm.name, len(input_array), num_threads)
                    for bucket_data, _ in active_buckets
                ]

                for unique_indices in pool.imap(find_bucket_unique_worker, worker_args):
                    if len(unique_indices) > 0:
                        result[total_unique : total_unique + len(unique_indices)] = (
                            unique_indices
                        )
                        total_unique += len(unique_indices)
                    pbar.update(1)

        # Cleanup shared memory and finalize
        shm.close()
        shm.unlink()

        # Create inverse indices array
        log.info("Creating inverse indices...")
        inverse_indices_file = os.path.join(
            mmap_folder, f"{prefix}_inverse_indices.dat"
        )
        inverse_indices = np.memmap(
            inverse_indices_file,
            dtype=np.int64,
            mode="w+",
            shape=(len(input_array),),
        )

        # Sort and create inverse indices
        result = result[:total_unique]
        result.sort()
        for i in range(len(input_array)):
            inverse_indices[i] = np.searchsorted(result, input_array[i])

        dedup_ratio = total_unique / len(input_array)
        log.info(f"\nFound {total_unique:,} unique elements ({dedup_ratio:.2%} unique)")

        return result, inverse_indices

    except Exception as e:
        log.error(f"Error in bucket_process_unique_values: {e}")
        raise
    finally:
        if "bucket_mgr" in locals():
            bucket_mgr.cleanup()
        if "shm" in locals():
            try:
                shm.close()
                shm.unlink()
            except:
                pass


def reassign(
    np_arrays: Dict[str, np.memmap],
    tmp_files: Dict[str, Any],
    iters: int = 25,
    step_min: float = -1.0,
    step_max: float = -1.0,
    mstep: int = 4,
    max_memory: Union[str, float, int] = "4G",
    num_threads: int = 1,
) -> pd.DataFrame:
    """Reassign multimapped reads using memory-efficient implementation."""
    # Initialize resource manager with parsed memory limit
    resource_manager = ResourceManager()
    if isinstance(max_memory, str):
        max_memory = resource_manager.parse_memory_limit(max_memory)
    resource_manager = ResourceManager(max_memory=max_memory, max_threads=num_threads)

    mmap_folder = tmp_files["mmap"]

    # Process unique values and get inverse indices
    # unique_subjects, subject_inverse_indices = bucket_process_unique_values(
    #     np_arrays["subject_numeric_id"], mmap_folder, "subject", resource_manager
    # )

    # unique_queries, query_inverse_indices = bucket_process_unique_values(
    #     np_arrays["query_numeric_id"], mmap_folder, "query", resource_manager
    # )
    log.info("Creating inverse subject mapping")
    subject_inverse_indices = initialize_mmap_array(
        total_positions=len(np_arrays["subject_numeric_id"]),
        dtype=np.int64,
        mmap_folder=mmap_folder,
        array_name="subject_inverse_indices",
    )

    subject_inverse_indices, unique_subjects = memory_efficient_factorize(
        np_arrays["subject_numeric_id"],
        mmap_folder=mmap_folder,
        max_memory=resource_manager.max_memory,
        inverse=subject_inverse_indices,
        num_threads=num_threads,
    )
    log.info("Starting factorization of reads")
    query_inverse_indices = initialize_mmap_array(
        total_positions=len(np_arrays["query_numeric_id"]),
        dtype=np.int64,
        mmap_folder=mmap_folder,
        array_name="reass_query_inverse_indices",
    )
    query_inverse_indices, unique_queries = memory_efficient_factorize(
        np_arrays["query_numeric_id"],
        mmap_folder=mmap_folder,
        max_memory=resource_manager.max_memory,
        inverse=query_inverse_indices,
        num_threads=num_threads,
    )

    log.info(f"Number of references: {len(unique_subjects):,}")
    log.info(f"Number of reads: {len(unique_queries):,}")

    with temp_memmap(
        os.path.join(mmap_folder, "iter_array.mmap"),
        dtype=np.int64,
        mode="w+",
        shape=(np_arrays["subject_numeric_id"].shape[0],),
    ) as iter_array:
        with temp_memmap(
            os.path.join(mmap_folder, "prob.dat"),
            dtype=np.float64,
            mode="w+",
            shape=(np_arrays["subject_numeric_id"].shape[0],),
        ) as prob:
            log.info("Initializing weights")
            prob[:] = chunked_initialize_weights(
                subject_inverse_indices,
                np_arrays["bitScore"],
                len(unique_subjects),
                mmap_folder,
                resource_manager,
            )

            log.info("Starting multimap resolution")
            final_mask = resolve_multimaps_return_indices(
                subject_inverse_indices=subject_inverse_indices,
                query_inverse_indices=query_inverse_indices,
                prob=prob,
                slen=np_arrays["slen"],
                iter_array=iter_array,
                mmap_folder=mmap_folder,
                iters=iters,
                step_min=step_min,
                step_max=step_max,
                mstep=mstep,
                resource_manager=resource_manager,  # Pass resource_manager
            )

            return pd.DataFrame(
                {
                    "query_numeric_id": np_arrays["query_numeric_id"][final_mask],
                    "subject_numeric_id": np_arrays["subject_numeric_id"][final_mask],
                    "bitScore": np_arrays["bitScore"][final_mask],
                    "alnLength": np_arrays["alnLength"][final_mask],
                    "subjectStart": np_arrays["subjectStart"][final_mask],
                    "subjectEnd": np_arrays["subjectEnd"][final_mask],
                    "percIdentity": np_arrays["percIdentity"][final_mask],
                    "row_hash": np_arrays["row_hash"][final_mask],
                }
            )
