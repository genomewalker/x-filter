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
import pyarrow as pa
import pyarrow.parquet as pq
import glob

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


@njit(parallel=True)
def parallel_accumulate_weights(
    chunk_indices: np.ndarray, chunk_scores: np.ndarray, max_index: int
) -> np.ndarray:
    """
    Accumulate weights in parallel for a chunk and return the result.
    Uses thread-local storage to avoid race conditions.
    Args:
        chunk_indices: Indices within the chunk (relative to the original array structure, but values are global indices).
        chunk_scores: Scores corresponding to the indices.
        max_index: The maximum index value + 1 (size of the dimension to accumulate over).
    Returns:
        A numpy array of shape (max_index,) containing the sum of scores for each index within this chunk.
    """
    n_threads = numba.get_num_threads()
    # Each thread accumulates into its own local array
    local_outputs = np.zeros((n_threads, max_index), dtype=chunk_scores.dtype)

    # Parallel accumulation into thread-local arrays
    for i in prange(len(chunk_indices)):
        thread_id = numba.get_thread_id()
        idx = chunk_indices[i]
        if 0 <= idx < max_index:  # Bounds check for safety
            local_outputs[thread_id, idx] += chunk_scores[i]

    # Reduce the thread-local results into a single output array (in memory)
    final_chunk_output = np.zeros(max_index, dtype=chunk_scores.dtype)
    for i in range(n_threads):
        final_chunk_output += local_outputs[i]  # Sum results from all threads

    return final_chunk_output


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


def chunked_initialize_weights(
    subject_inverse_indices: np.memmap,
    bitScore: np.memmap,
    max_index: int,
    mmap_folder: str,
    resource_manager: ResourceManager,
    output_array: Optional[np.memmap] = None
) -> np.memmap:
    """Initialize weights using chunked processing with resource management."""
    arr_info = resource_manager.analyze_array(subject_inverse_indices)
    strategy = resource_manager.calculate_chunk_size(arr_info)
    chunk_size = strategy.chunk_size

    total_weights_file = os.path.join(mmap_folder, "total_weights.dat")
    total_weights = None

    try:
        # Use a memmap array for the final total weights across all chunks
        total_weights = np.memmap(
            total_weights_file, dtype=np.float64, mode="w+", shape=(max_index,)
        )
        total_weights.fill(0)
        total_weights.flush()

        log.info("Calculating total weights per subject (chunked)")
        with tqdm(
            total=len(subject_inverse_indices), desc="Calculating weights", ncols=80
        ) as pbar:
            for start in range(0, len(subject_inverse_indices), chunk_size):
                end = min(start + chunk_size, len(subject_inverse_indices))

                # Get chunk data
                chunk_indices = subject_inverse_indices[start:end]
                # Ensure scores are float64 for accumulation consistency
                chunk_scores = bitScore[start:end].astype(np.float64, copy=False)

                # Accumulate weights for the chunk in parallel (returns in-memory array)
                chunk_contribution = parallel_accumulate_weights(
                    chunk_indices, chunk_scores, max_index
                )

                # Add the chunk's contribution to the total_weights memmap array sequentially
                total_weights += chunk_contribution

                pbar.update(end - start)
                # Explicitly delete the temporary chunk result and collect garbage
                del chunk_contribution
                gc.collect()

        total_weights.flush()  # Ensure all additions are written

        # Handle zero weights exactly as original
        zero_weight_mask = total_weights == 0
        total_weights[zero_weight_mask] = np.finfo(np.float64).tiny
        total_weights.flush()  # Flush changes

        # Create result array or use provided one
        if output_array is None:
            result_file = os.path.join(mmap_folder, "weights_result.dat")
            result = np.memmap(
                result_file, dtype=np.float64, mode="w+", shape=bitScore.shape
            )
        else:
            result = output_array  # Use the provided memmap array

        # Calculate final weights in chunks, preserving original math
        log.info("Calculating final normalized weights (chunked)")
        with tqdm(
            total=len(subject_inverse_indices),
            desc="Calculating final weights",
            ncols=80,
        ) as pbar:
            for start in range(0, len(subject_inverse_indices), chunk_size):
                end = min(start + chunk_size, len(bitScore))
                chunk_indices = subject_inverse_indices[start:end]
                chunk_scores = bitScore[start:end]

                # Get corresponding total weights for the chunk indices
                weights_for_chunk = total_weights[chunk_indices]

                # Perform division, ensuring float64
                result[start:end] = chunk_scores.astype(np.float64, copy=False) / weights_for_chunk
                pbar.update(end - start)
        result.flush()
        gc.collect()

        return result

    finally:
        # Cleanup total_weights memmap
        if total_weights is not None:
            # Ensure data is flushed before deleting object
            try:
                total_weights.flush()
            except Exception as e:
                log.warning(f"Error flushing total_weights on cleanup: {e}")
            del total_weights
            gc.collect()
        # Attempt to delete the file
        try:
            if os.path.exists(total_weights_file):
                os.unlink(total_weights_file)
        except OSError as e:
            log.warning(f"Error deleting temporary file {total_weights_file}: {e}")

# The rest of the file remains unchanged.
