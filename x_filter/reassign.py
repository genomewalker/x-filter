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
    calculate_parallel_distribution,
    process_chunk_worker,
    find_bucket_unique_worker,
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


@njit
def compute_p_new(
    prob_chunk: np.ndarray,
    r_chunk: np.ndarray,
    v_chunk: np.ndarray,
    two_alpha: float,
    alpha2: float,
) -> np.ndarray:
    """JIT-compiled function to compute p_new efficiently."""
    result = np.empty_like(prob_chunk)
    for i in range(len(prob_chunk)):
        result[i] = prob_chunk[i] + two_alpha * r_chunk[i] + alpha2 * v_chunk[i]
    return result


@njit
def compute_squared_sum(arr: np.ndarray) -> float:
    """JIT-compiled efficient sum of squares without temporary arrays."""
    result = 0.0
    for i in range(len(arr)):
        result += arr[i] * arr[i]
    return result


@njit
def compute_dot_product(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """JIT-compiled efficient dot product without temporary arrays."""
    result = 0.0
    for i in range(len(arr1)):
        result += arr1[i] * arr2[i]
    return result


def calculate_optimal_chunk_size(
    array_size: int,
    dtype_size: int,
    available_memory: int,
    overhead_factor: float = 0.6,
) -> int:
    """
    Calculate optimal chunk size based on array size and available memory,
    with stricter memory constraints.

    Args:
        array_size: Total number of elements in array
        dtype_size: Size of each element in bytes
        available_memory: Available memory in bytes
        overhead_factor: Factor to account for Python overhead (0.0-1.0)

    Returns:
        Optimal chunk size (number of elements)
    """
    # Calculate how much memory we can safely use (lower overhead factor for safety)
    usable_memory = int(available_memory * overhead_factor)

    # Enforce absolute maximum chunk size of 100M elements
    MAX_CHUNK_SIZE = 100_000_000

    # Calculate memory-based chunk size
    memory_based_size = usable_memory // (
        dtype_size * 3
    )  # Account for multiple array copies

    # For very large arrays, limit chunks based on number of elements
    if array_size > 1_000_000_000:  # More than 1 billion elements
        # Use fewer chunks for very large arrays, but still cap at MAX_CHUNK_SIZE
        min_chunks = 1000 if array_size > 10_000_000_000 else 500
        array_based_size = min(array_size // min_chunks, MAX_CHUNK_SIZE)

        # Use the smaller of the two sizes to ensure we don't use too much memory
        chunk_size = min(memory_based_size, array_based_size)
    else:
        # For smaller arrays, still respect memory constraints and max size
        chunk_size = min(memory_based_size, MAX_CHUNK_SIZE)

    # Ensure minimum reasonable size and don't exceed array size
    chunk_size = max(min(chunk_size, array_size), 100_000)

    # Log the calculation details
    memory_usage_gb = (chunk_size * dtype_size) / (1024 * 1024 * 1024)
    log.info(
        f"Calculated chunk size: {chunk_size:,} elements "
        f"({memory_usage_gb:.2f} GB) for {array_size:,} total elements"
    )
    log.info(f"This will process the array in {array_size/chunk_size:.1f} chunks")

    return chunk_size


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

        return result

    finally:
        # Cleanup
        try:
            if os.path.exists(total_weights_file):
                os.unlink(total_weights_file)
        except OSError:
            pass


def validate_probabilities(
    prob: np.ndarray, query_indices: np.ndarray, max_query: int, mmap_folder: str
) -> bool:
    """Validate probability array for numerical stability."""
    if np.any(prob < 0):
        return False

    # Use memmap for large temporary array
    prob_sum_file = os.path.join(mmap_folder, "prob_sum_temp.mmap")
    try:
        prob_sum = np.memmap(
            prob_sum_file, dtype=np.float64, mode="w+", shape=(max_query + 1,)
        )
        prob_sum.fill(0)

        # Process in chunks to reduce memory usage
        chunk_size = 1_000_000
        for i in range(0, len(query_indices), chunk_size):
            chunk_end = min(i + chunk_size, len(query_indices))
            np.add.at(prob_sum, query_indices[i:chunk_end], prob[i:chunk_end])

        result = not np.any(prob_sum == 0)
        return result
    finally:
        try:
            if os.path.exists(prob_sum_file):
                os.unlink(prob_sum_file)
        except OSError:
            pass


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
    prob_sum_file = os.path.join(mmap_folder, "prob_sum_temp.mmap")

    try:
        new_prob = np.memmap(
            new_prob_file, dtype=np.float64, mode="w+", shape=input_prob.shape
        )
        new_prob[:] = input_prob[:]

        # Update masked entries in chunks to avoid allocating full index arrays
        tiny = np.finfo(np.float64).tiny
        for cstart in range(0, len(mask), chunk_size):
            cend = min(cstart + chunk_size, len(mask))
            cm = mask[cstart:cend]
            if not np.any(cm):
                continue
            # Get indices just for this chunk, keeping memory use low
            offs = np.nonzero(cm)[0] + cstart
            mp = input_prob[offs]
            ms = slen[offs]
            ms[ms == 0] = tiny
            new_prob[offs] = mp * (mp / ms)

        # Create prob_sum as memory-mapped array
        prob_sum = np.memmap(
            prob_sum_file, dtype=np.float64, mode="w+", shape=(max_query + 1,)
        )
        prob_sum.fill(0)

        # Accumulate probabilities in chunks
        for start in range(0, len(mask), chunk_size):
            end = min(start + chunk_size, len(mask))
            chunk_mask = mask[start:end]
            if not np.any(chunk_mask):
                continue
            chunk_queries = query_inverse_indices[start:end][chunk_mask]
            chunk_probs = new_prob[start:end][chunk_mask]
            np.add.at(prob_sum, chunk_queries, chunk_probs)

        # Normalize masked entries in chunks
        for cstart in range(0, len(mask), chunk_size):
            cend = min(cstart + chunk_size, len(mask))
            cm = mask[cstart:cend]
            if not np.any(cm):
                continue
            # Get indices just for this chunk
            offs = np.nonzero(cm)[0] + cstart
            qs = query_inverse_indices[offs]
            sums = prob_sum[qs]
            sums[sums == 0] = tiny
            new_prob[offs] = new_prob[offs] / sums

        return new_prob

    finally:
        # Clean up temporary files
        try:
            if os.path.exists(new_prob_file):
                os.unlink(new_prob_file)
            if os.path.exists(prob_sum_file):
                os.unlink(prob_sum_file)
        except OSError:
            pass


def check_memory_requirements(
    prob_size_bytes: int, resource_manager: ResourceManager
) -> bool:
    """
    Check if there's enough memory to run the SQUAREM algorithm safely,
    taking into account our chunking strategy.
    """
    # Calculate max chunk size based on array size and dtype
    array_size = prob_size_bytes // 8  # Assuming float64 (8 bytes)
    dtype_size = 8  # float64

    # Get the optimal chunk size we would use
    chunk_size = calculate_optimal_chunk_size(
        array_size=array_size,
        dtype_size=dtype_size,
        available_memory=resource_manager.available_memory,
    )

    # Calculate memory needed for a single chunk processing
    # We need memory for several arrays: q, r, r2, v, p_new chunks plus overhead
    chunk_bytes = chunk_size * dtype_size
    required_mem_per_chunk = chunk_bytes * 5  # 5 arrays in memory

    # Add memory for other operations and Python overhead
    overhead_factor = 1.5
    total_required = required_mem_per_chunk * overhead_factor

    available_mem = resource_manager.available_memory
    safe_ratio = available_mem / total_required

    log.info(f"Memory check for chunked processing:")
    log.info(
        f"  - Chunk size: {chunk_size:,} elements ({chunk_bytes/(1024*1024):.2f} MB)"
    )
    log.info(
        f"  - Required per chunk: {required_mem_per_chunk/(1024*1024*1024):.2f} GB"
    )
    log.info(f"  - Available memory: {available_mem/(1024*1024*1024):.2f} GB")
    log.info(f"  - Safety ratio: {safe_ratio:.2f}")

    # We want at least 20% headroom
    if safe_ratio < 1.2:
        log.warning(
            f"Available memory ({available_mem/(1024**3):.2f} GB) may be insufficient "
            f"for SQUAREM algorithm with current chunk size."
        )
        log.warning(
            f"Consider reducing chunk size further or increasing available memory."
        )
        return False

    return True


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
    resource_manager: Optional[ResourceManager] = None,
) -> np.ndarray:
    """SQUAREM implementation using chunked processing with strict memory management."""
    if resource_manager is None:
        resource_manager = ResourceManager()

    # Get array size information
    array_size_gb = prob.nbytes / (1024**3)
    log.info(f"SQUAREM processing array of size: {array_size_gb:.2f} GB")

    # Calculate optimal chunk size based on array size and available memory
    # with maximum chunk size enforced
    chunk_size = calculate_optimal_chunk_size(
        array_size=len(prob),
        dtype_size=prob.dtype.itemsize,
        available_memory=resource_manager.available_memory,
    )

    # Use mega-chunks for bulk operations, but limit maximum size
    mega_chunk_factor = min(5, max(1, 100_000_000 // chunk_size))
    mega_chunk = min(chunk_size * mega_chunk_factor, 100_000_000)

    log.info(f"Using chunk size: {chunk_size:,}, mega chunk: {mega_chunk:,}")

    # Track and log memory usage
    available_mem_gb = resource_manager.available_memory / (1024**3)
    log.info(f"Available memory before SQUAREM: {available_mem_gb:.2f} GB")

    # First fixed point evaluation - initialize q to default value
    q = None  # Initialize q to ensure it's in scope
    try:
        log.info("Computing first fixed point map")
        q = chunked_fixed_point_map(
            prob,
            mask,
            slen,
            query_inverse_indices,
            max_query,
            mmap_folder,
            resource_manager,
        )
    except Exception as e:
        log.error(f"Failed to compute first fixed point map: {str(e)}")
        return prob

    if q is None:
        log.error("First fixed point calculation returned None")
        return prob

    # Process each step with careful memory management
    r_file = os.path.join(mmap_folder, "r_temp.mmap")
    r2_file = os.path.join(mmap_folder, "r2_temp.mmap")
    v_file = os.path.join(mmap_folder, "v_temp.mmap")
    p_new_file = os.path.join(mmap_folder, "p_new_temp.mmap")
    result_file = os.path.join(mmap_folder, "squarem_result.mmap")

    try:
        # Step 1: Calculate first difference r = q - prob
        log.info("Computing first difference vector r")
        r = np.memmap(r_file, dtype=np.float64, mode="w+", shape=prob.shape)

        # Process in mega-chunks for better I/O performance
        for start in range(0, len(prob), mega_chunk):
            end = min(start + mega_chunk, len(prob))
            r[start:end] = q[start:end] - prob[start:end]
            # Flush to disk less frequently
            if start % (mega_chunk * 2) == 0:
                r.flush()

        # Calculate sr2 using JIT-compiled function in mega-chunks
        sr2 = 0.0
        for start in range(0, len(r), mega_chunk):
            end = min(start + mega_chunk, len(r))
            sr2 += compute_squared_sum(r[start:end])

        # Early convergence check
        if sr2 < 1e-10:
            log.info("Early convergence detected")
            return q

        # Second fixed point evaluation
        log.info("Computing second fixed point map")
        q2 = chunked_fixed_point_map(
            q,
            mask,
            slen,
            query_inverse_indices,
            max_query,
            mmap_folder,
            resource_manager,
        )

        if q2 is None:
            log.error("Second fixed point calculation returned None")
            return q

        # Re-open r for reading only to save memory
        r = np.memmap(r_file, dtype=np.float64, mode="r", shape=prob.shape)

        # Calculate r2 = q2 - q and v = r2 - r in a single mega-chunk pass
        log.info("Computing difference vectors")
        r2 = np.memmap(r2_file, dtype=np.float64, mode="w+", shape=prob.shape)
        v = np.memmap(v_file, dtype=np.float64, mode="w+", shape=prob.shape)

        for start in range(0, len(prob), mega_chunk):
            end = min(start + mega_chunk, len(prob))
            # Calculate both in one pass to reduce memory operations
            r2_chunk = q2[start:end] - q[start:end]
            r2[start:end] = r2_chunk
            v[start:end] = r2_chunk - r[start:end]

            if start % (mega_chunk * 2) == 0:
                r2.flush()
                v.flush()

        # Free memory
        del r2_chunk

        # Calculate sv2 and srv using JIT-compiled functions
        log.info("Computing acceleration parameters")
        sv2 = 0.0
        srv = 0.0
        for start in range(0, len(v), mega_chunk):
            end = min(start + mega_chunk, len(v))
            v_chunk = v[start:end]
            r_chunk = r[start:end]
            sv2 += compute_squared_sum(v_chunk)
            srv += compute_dot_product(r_chunk, v_chunk)

        # Check stability
        if sv2 < 1e-10:
            log.info("Acceleration numerically unstable, returning second iterate")
            return q2

        # Calculate step length
        if step_min < 0:
            step_min = 0.001
        if step_max < step_min:
            step_max = 1.0

        alpha = np.sqrt(sr2 / sv2)
        alpha = np.clip(alpha, step_min, step_max)
        log.info(f"SQUAREM step length: alpha = {alpha:.6f}")

        # Pre-calculate coefficients for JIT function
        alpha2 = alpha * alpha
        two_alpha = 2 * alpha

        # Calculate p_new using JIT-compiled function
        log.info("Computing accelerated point")
        p_new = np.memmap(p_new_file, dtype=np.float64, mode="w+", shape=prob.shape)

        for start in range(0, len(prob), mega_chunk):
            end = min(start + mega_chunk, len(prob))
            p_new[start:end] = compute_p_new(
                prob[start:end], r[start:end], v[start:end], two_alpha, alpha2
            )

            # Flush less frequently
            if start % (mega_chunk * 2) == 0:
                p_new.flush()

        # Validate probabilities
        log.info("Validating accelerated point")
        valid = validate_probabilities(
            p_new[mask], query_inverse_indices[mask], max_query, mmap_folder
        )

        if valid:
            log.info("Computing fixed point of accelerated iterate")
            result = chunked_fixed_point_map(
                p_new,
                mask,
                slen,
                query_inverse_indices,
                max_query,
                mmap_folder,
                resource_manager,
            )

            # Free memory
            del p_new
            gc.collect()

            # Create final result array
            log.info("Creating final result")
            final_result = np.memmap(
                result_file, dtype=np.float64, mode="w+", shape=prob.shape
            )

            # Copy result in small chunks
            for start in range(0, len(result), chunk_size):
                end = min(start + chunk_size, len(result))
                final_result[start:end] = result[start:end]
                if start % (chunk_size * 5) == 0:
                    final_result.flush()
                    gc.collect()

            # Free memory
            del result
            gc.collect()

            # Validate final result
            valid_final = validate_probabilities(
                final_result[mask], query_inverse_indices[mask], max_query, mmap_folder
            )

            if valid_final:
                log.info("Final result validated successfully")
                return final_result
            else:
                log.warning("Final result validation failed, using second iterate")
                # Copy q2 to final_result
                for start in range(0, len(q2), chunk_size):
                    end = min(start + chunk_size, len(q2))
                    final_result[start:end] = q2[start:end]
                    if start % (chunk_size * 5) == 0:
                        final_result.flush()
                return final_result

        # If initial validation fails, try step halving
        log.info("Initial validation failed, attempting step halving")
        r = np.memmap(r_file, dtype=np.float64, mode="r", shape=prob.shape)
        v = np.memmap(v_file, dtype=np.float64, mode="r", shape=prob.shape)
        p_new = np.memmap(p_new_file, dtype=np.float64, mode="r+", shape=prob.shape)

        for m in range(mstep):
            alpha = alpha / 2
            log.info(f"Step halving iteration {m+1}, alpha={alpha:.6f}")

            # Update p_new in chunks
            for start in range(0, len(prob), chunk_size):
                end = min(start + chunk_size, len(prob))
                p_new[start:end] = (
                    prob[start:end]
                    + 2 * alpha * r[start:end]
                    + alpha * alpha * v[start:end]
                )
                if start % (chunk_size * 5) == 0:
                    p_new.flush()
                    gc.collect()

            valid = validate_probabilities(
                p_new[mask], query_inverse_indices[mask], max_query, mmap_folder
            )

            if valid:
                # Free memory before heavy computation
                del r, v
                gc.collect()

                log.info(f"Step halving succeeded with alpha={alpha:.6f}")
                result = chunked_fixed_point_map(
                    p_new,
                    mask,
                    slen,
                    query_inverse_indices,
                    max_query,
                    mmap_folder,
                    resource_manager,
                )

                # Free memory
                del p_new
                gc.collect()

                # Create final result
                final_result = np.memmap(
                    result_file, dtype=np.float64, mode="w+", shape=prob.shape
                )

                # Copy in small chunks
                for start in range(0, len(result), chunk_size):
                    end = min(start + chunk_size, len(result))
                    final_result[start:end] = result[start:end]
                    if start % (chunk_size * 5) == 0:
                        final_result.flush()
                        gc.collect()

                # Free memory
                del result
                gc.collect()

                valid_final = validate_probabilities(
                    final_result[mask],
                    query_inverse_indices[mask],
                    max_query,
                    mmap_folder,
                )

                if valid_final:
                    log.info("Step-halved result validated successfully")
                    return final_result

        # If all steps fail, use second iteration
        log.warning("All step halving attempts failed, using second iterate")
        final_result = np.memmap(
            result_file, dtype=np.float64, mode="w+", shape=prob.shape
        )

        # Copy q2 to final result
        for start in range(0, len(q2), chunk_size):
            end = min(start + chunk_size, len(q2))
            final_result[start:end] = q2[start:end]
            if start % (chunk_size * 5) == 0:
                final_result.flush()
                gc.collect()

        return final_result

    except Exception as e:
        log.error(f"Error in SQUAREM step: {str(e)}")
        # In case of error, try to return q2 if available, otherwise return q if available, finally return prob
        if "q2" in locals() and q2 is not None:
            return q2
        elif q is not None:
            return q
        return prob

    finally:
        # Clean up all temporary files
        for file_path in [r_file, r2_file, v_file, p_new_file]:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except OSError as e:
                log.warning(f"Error cleaning up {file_path}: {str(e)}")

        # Final garbage collection
        gc.collect()


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
    resource_manager: Optional[ResourceManager] = None,
) -> np.ndarray:
    """Resolve multimapped reads using chunked processing."""
    log.info(f"Multimap resolution using scale={scale}")
    # Add memory check before starting iterations
    if resource_manager is not None and not check_memory_requirements(
        prob.nbytes, resource_manager
    ):
        log.warning("Memory check failed. Proceeding with extra caution.")
        # Reduce chunk size further or take other measures

    # Create memory-mapped mask array instead of in-memory
    mask_file = os.path.join(mmap_folder, "mask.dat")
    mask = np.memmap(
        mask_file, dtype=np.bool_, mode="w+", shape=subject_inverse_indices.shape
    )
    mask.fill(1)  # Initialize all to True

    # Use memory-mapped array for total_reads calculation
    unique_queries_file = os.path.join(mmap_folder, "unique_queries_temp.dat")

    # Calculate unique queries in chunks to avoid memory issues
    max_query = query_inverse_indices.max()
    query_counts = np.memmap(
        unique_queries_file, dtype=np.int8, mode="w+", shape=(max_query + 1,)
    )
    query_counts.fill(0)

    # Count queries in chunks
    chunk_size = min(100_000_000, len(mask))
    for start in range(0, len(query_inverse_indices), chunk_size):
        end = min(start + chunk_size, len(query_inverse_indices))
        chunk_queries = query_inverse_indices[start:end]
        unique_indices = np.unique(chunk_queries)
        query_counts[unique_indices] = 1

    total_reads = np.sum(query_counts)
    del query_counts

    try:
        os.unlink(unique_queries_file)
    except OSError:
        pass

    current_iter = 0
    prev_num_alignments = np.inf

    log.info(
        f"Starting multimap resolution: {iters} iterations"
        if iters > 0
        else "Resolving multimaps until convergence"
    )

    # Calculate initial alignments using chunked processing
    total_alignments = 0
    for start in range(0, len(mask), chunk_size):
        end = min(start + chunk_size, len(mask))
        total_alignments += np.sum(mask[start:end])
    log.info(f"Initial alignments: {total_alignments:,}")

    prob_working_file = os.path.join(mmap_folder, "prob_working.mmap")
    prob_working = np.memmap(
        prob_working_file, dtype=np.float64, mode="w+", shape=prob.shape
    )
    prob_working[:] = prob[:]

    try:
        while iters == 0 or current_iter < iters:
            # Count alignments in chunks
            n_alns = 0
            for start in range(0, len(mask), chunk_size):
                end = min(start + chunk_size, len(mask))
                n_alns += np.sum(mask[start:end])

            if n_alns == prev_num_alignments:
                log.info("Convergence reached - no more alignments removed")
                break

            prev_num_alignments = n_alns
            gc.collect()  # Force garbage collection between iterations

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
                    resource_manager=resource_manager,
                )
                pbar.update(1)
                gc.collect()  # Force garbage collection after SQUAREM

                # Use memmap for large temporary arrays
                n_aln_file = os.path.join(
                    mmap_folder, f"n_aln_temp_{current_iter}.mmap"
                )
                max_prob_file = os.path.join(
                    mmap_folder, f"max_prob_temp_{current_iter}.mmap"
                )
                unique_mask_file = os.path.join(
                    mmap_folder, f"unique_mask_temp_{current_iter}.mmap"
                )
                non_unique_mask_file = os.path.join(
                    mmap_folder, f"non_unique_mask_temp_{current_iter}.mmap"
                )
                max_prob_scaled_file = os.path.join(
                    mmap_folder, f"max_prob_scaled_temp_{current_iter}.mmap"
                )
                final_mask_file = os.path.join(
                    mmap_folder, f"final_mask_temp_{current_iter}.mmap"
                )

                try:
                    # Create n_aln as memory-mapped
                    n_aln = np.memmap(
                        n_aln_file, dtype=np.int64, mode="w+", shape=(max_query + 1,)
                    )
                    n_aln.fill(0)

                    # Process in chunks
                    for start in range(0, len(mask), chunk_size):
                        end = min(start + chunk_size, len(mask))
                        chunk_mask = mask[start:end]
                        if not np.any(chunk_mask):
                            continue
                        chunk_queries = query_inverse_indices[start:end][chunk_mask]
                        np.add.at(n_aln, chunk_queries, 1)

                    # Create unique_mask and non_unique_mask as memory-mapped arrays
                    unique_mask = np.memmap(
                        unique_mask_file, dtype=np.bool_, mode="w+", shape=mask.shape
                    )
                    unique_mask.fill(False)

                    non_unique_mask = np.memmap(
                        non_unique_mask_file,
                        dtype=np.bool_,
                        mode="w+",
                        shape=mask.shape,
                    )
                    non_unique_mask.fill(False)

                    # Process in chunks to avoid memory issue during mask creation
                    for start in range(0, len(mask), chunk_size):
                        end = min(start + chunk_size, len(mask))
                        chunk_mask = mask[start:end]
                        chunk_queries = query_inverse_indices[start:end]
                        chunk_n_aln = n_aln[chunk_queries]

                        unique_mask[start:end] = (chunk_n_aln == 1) & chunk_mask
                        non_unique_mask[start:end] = (chunk_n_aln > 1) & chunk_mask

                    pbar.update(1)

                    if np.all(unique_mask):
                        log.info("All reads uniquely mapped - stopping early")
                        break

                    # Create max_prob as memory-mapped
                    max_prob = np.memmap(
                        max_prob_file,
                        dtype=np.float64,
                        mode="w+",
                        shape=(max_query + 1,),
                    )
                    max_prob.fill(0)

                    # Process in chunks for max_prob
                    for start in range(0, len(mask), chunk_size):
                        end = min(start + chunk_size, len(mask))
                        chunk_mask = mask[start:end]
                        if not np.any(chunk_mask):
                            continue
                        chunk_queries = query_inverse_indices[start:end][chunk_mask]
                        chunk_probs = prob_working[start:end][chunk_mask]
                        np.maximum.at(max_prob, chunk_queries, chunk_probs)

                    # Create max_prob_scaled as memory-mapped
                    log.info(f"Applying scale threshold: scale={scale}")
                    max_prob_scaled = np.memmap(
                        max_prob_scaled_file,
                        dtype=np.float64,
                        mode="w+",
                        shape=mask.shape,
                    )

                    # Process in chunks for scaling
                    for start in range(0, len(mask), chunk_size):
                        end = min(start + chunk_size, len(mask))
                        chunk_queries = query_inverse_indices[start:end]
                        # non-positive scale → exact max; positive scale → fraction of max
                        if scale <= 0:
                            max_prob_scaled[start:end] = max_prob[chunk_queries]
                        else:
                            max_prob_scaled[start:end] = max_prob[chunk_queries] * scale

                    # Log a random sample to verify scale is having an effect
                    sample_size = min(10, len(mask))
                    sample_indices = np.random.choice(
                        np.arange(len(mask))[mask], sample_size, replace=False
                    )
                    log.info(f"Sample probs vs thresholds:")
                    for idx in sample_indices:
                        query_idx = query_inverse_indices[idx]
                        log.info(
                            f"  Read {query_idx}: prob={prob_working[idx]:.6f}, max={max_prob[query_idx]:.6f}, threshold={max_prob_scaled[idx]:.6f}"
                        )

                    # Count alignments that would be kept with different scales
                    if current_iter == 0:  # Only on first iteration
                        test_scales = [0.0, 0.5, 0.9, 0.99, 1.0]
                        for test_scale in test_scales:
                            test_count = 0
                            for s in range(0, len(mask), chunk_size):
                                e = min(s + chunk_size, len(mask))
                                chunk_non_unique = non_unique_mask[s:e]
                                if not np.any(chunk_non_unique):
                                    continue
                                chunk_probs = prob_working[s:e]
                                chunk_queries = query_inverse_indices[s:e]
                                if test_scale <= 0:
                                    test_thresh = max_prob[chunk_queries]
                                else:
                                    test_thresh = max_prob[chunk_queries] * test_scale
                                test_count += np.sum(
                                    (chunk_probs >= test_thresh) & chunk_non_unique
                                )
                            log.info(
                                f"  Scale {test_scale} would keep {test_count:,} alignments"
                            )

                    pbar.update(1)

                    # Create final_mask as memory-mapped
                    final_mask = np.memmap(
                        final_mask_file, dtype=np.bool_, mode="w+", shape=mask.shape
                    )
                    final_mask.fill(False)

                    # Process in chunks for final mask calculation
                    for start in range(0, len(mask), chunk_size):
                        end = min(start + chunk_size, len(mask))
                        chunk_non_unique = non_unique_mask[start:end]
                        if not np.any(chunk_non_unique):
                            continue
                        chunk_probs = prob_working[start:end]
                        chunk_max_scaled = max_prob_scaled[start:end]
                        final_mask[start:end] = (
                            chunk_probs >= chunk_max_scaled
                        ) & chunk_non_unique

                    # Count final alignments kept
                    final_count = 0
                    for start in range(0, len(final_mask), chunk_size):
                        end = min(start + chunk_size, len(final_mask))
                        final_count += np.sum(final_mask[start:end])
                    log.info(
                        f"Scale {scale} kept {final_count:,} alignments after filtering"
                    )

                    pbar.update(1)

                    # Update iter_array in chunks
                    for start in range(0, len(mask), chunk_size):
                        end = min(start + chunk_size, len(mask))
                        chunk_final_mask = final_mask[start:end]
                        if np.any(chunk_final_mask):
                            iter_array[start:end][chunk_final_mask] = current_iter + 1

                    # Update mask in chunks
                    for start in range(0, len(mask), chunk_size):
                        end = min(start + chunk_size, len(mask))
                        chunk_unique = unique_mask[start:end]
                        chunk_final = final_mask[start:end]
                        mask[start:end] = chunk_unique | chunk_final

                    pbar.update(1)

                    # Calculate global statistics in chunks
                    global_uniques = 0
                    for start in range(0, len(unique_mask), chunk_size):
                        end = min(start + chunk_size, len(unique_mask))
                        global_uniques += np.sum(unique_mask[start:end])

                    reads_to_process = total_reads - global_uniques

                finally:
                    # Clean up temporary files after each iteration
                    for temp_file in [
                        n_aln_file,
                        max_prob_file,
                        unique_mask_file,
                        non_unique_mask_file,
                        max_prob_scaled_file,
                        final_mask_file,
                    ]:
                        try:
                            if os.path.exists(temp_file):
                                os.unlink(temp_file)
                        except OSError:
                            pass

                    # Force garbage collection
                    gc.collect()

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

        # Create a new memory-mapped array for the final result
        final_result_file = os.path.join(mmap_folder, "final_mask_result.dat")
        final_result = np.memmap(
            final_result_file, dtype=np.bool_, mode="w+", shape=mask.shape
        )

        # Copy the result in chunks
        for start in range(0, len(mask), chunk_size):
            end = min(start + chunk_size, len(mask))
            final_result[start:end] = mask[start:end]

        # Wait for any pending I/O and garbage collect
        final_result.flush()
        gc.collect()

        return final_result

    finally:
        # Clean up all temporary files
        for file_path in [mask_file, prob_working_file]:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except OSError:
                pass


def reassign(
    np_arrays: Dict[str, np.memmap],
    tmp_files: Dict[str, Any],
    iters: int = 25,
    step_min: float = -1.0,
    step_max: float = -1.0,
    mstep: int = 4,
    scale: float = 0.9,
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
                scale=scale,
                resource_manager=resource_manager,
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
