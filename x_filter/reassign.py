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


@njit
def compute_p_new(prob_chunk: np.ndarray, r_chunk: np.ndarray, v_chunk: np.ndarray, 
                  two_alpha: float, alpha2: float) -> np.ndarray:
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


def chunked_initialize_weights(
    subject_inverse_indices: np.memmap,
    bitScore: np.memmap,
    max_index: int,
    mmap_folder: str,
    resource_manager: ResourceManager,
    output_array: Optional[np.memmap] = None
) -> np.memmap:
    """Initialize weights using chunked processing with resource management."""
    # Estimate concurrent chunks needed: chunk_indices, chunk_scores, weights_chunk ~ 3
    num_concurrent = 3
    arr_info = resource_manager.analyze_array(subject_inverse_indices)
    strategy = resource_manager.calculate_chunk_size(arr_info, num_concurrent_chunks=num_concurrent)
    chunk_size = strategy.chunk_size

    # Create memory-mapped array for total weights
    total_weights_file = os.path.join(mmap_folder, "total_weights.dat")
    total_weights = np.memmap(
        total_weights_file, dtype=np.float64, mode="w+", shape=(max_index,)
    )
    total_weights.fill(0)

    try:
        # Process in chunks using np.add.at for accumulation
        log.info("Accumulating weights using np.add.at")
        with tqdm(
            total=len(subject_inverse_indices), desc="Calculating weights", ncols=80
        ) as pbar:
            for start in range(0, len(subject_inverse_indices), chunk_size):
                end = min(start + chunk_size, len(subject_inverse_indices))

                # Get chunk data directly into memory
                chunk_indices = subject_inverse_indices[start:end]
                chunk_scores = bitScore[start:end]

                # Use np.add.at for safe accumulation into the memmap array
                np.add.at(total_weights, chunk_indices, chunk_scores)

                pbar.update(end - start)
                del chunk_indices
                del chunk_scores
                gc.collect()

        # Flush accumulated weights to disk
        total_weights.flush()
        log.info("Weight accumulation finished, flushing total_weights.")

        # Handle zero weights exactly as original
        total_weights[total_weights == 0] = np.finfo(np.float64).tiny

        # Create or use the provided result array
        if output_array is None:
            result_file = os.path.join(mmap_folder, "weights_result.dat")
            result = np.memmap(
                result_file, dtype=np.float64, mode="w+", shape=bitScore.shape
            )
            log.info(f"Created new result memmap: {result_file}")
        else:
            result = output_array
            log.info("Using provided output_array for results.")

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
                weights_chunk = total_weights[chunk_indices]

                result[start:end] = chunk_scores / weights_chunk

                pbar.update(end - start)
                del chunk_indices
                del chunk_scores
                del weights_chunk
                gc.collect()

        result.flush()
        log.info("Final weight calculation finished, flushing result array.")
        return result

    finally:
        log.debug(f"Attempting to clean up {total_weights_file}")
        if 'total_weights' in locals() and isinstance(total_weights, np.memmap):
            try:
                total_weights.flush()
            except Exception as e:
                log.warning(f"Error flushing total_weights before deletion: {e}")
            del total_weights
            gc.collect()

        try:
            if os.path.exists(total_weights_file):
                os.unlink(total_weights_file)
                log.debug(f"Successfully deleted {total_weights_file}")
        except OSError as e:
            log.warning(f"Error deleting temporary file {total_weights_file}: {e}")


def validate_probabilities(
    prob: np.ndarray,  # Can be memmap or ndarray
    query_indices: np.ndarray,  # Can be memmap or ndarray
    max_query: int,
    mmap_folder: str,
    resource_manager: ResourceManager  # Add resource_manager parameter
) -> bool:
    """Validate probability array for numerical stability using dynamic chunking."""
    prob_sum_file = os.path.join(mmap_folder, "prob_sum_temp.mmap")
    prob_sum = None  # Initialize to None for finally block
    try:
        log.info("Validating probabilities...")
        prob_sum = np.memmap(
            prob_sum_file, dtype=np.float64, mode="w+", shape=(max_query + 1,)
        )
        prob_sum.fill(0)

        # Calculate chunk size dynamically
        # Estimate concurrent chunks: prob_chunk, indices_chunk ~ 2
        num_concurrent = 2
        arr_info = resource_manager.analyze_array(prob)
        strategy = resource_manager.calculate_chunk_size(arr_info, num_concurrent_chunks=num_concurrent)
        chunk_size = strategy.chunk_size
        log.info(f"Using chunk size {chunk_size:,} for probability validation.")

        negative_found = False
        # Process in chunks to reduce memory usage
        with tqdm(total=len(query_indices), desc="Validating probabilities", ncols=80) as pbar:
            for i in range(0, len(query_indices), chunk_size):
                chunk_end = min(i + chunk_size, len(query_indices))
                prob_chunk = prob[i:chunk_end]

                # Check for negative probabilities within the chunk
                if not negative_found and np.any(prob_chunk < 0):
                    log.error("Negative probabilities found during validation.")
                    negative_found = True

                # Accumulate sums using the chunk
                indices_chunk = query_indices[i:chunk_end]
                np.add.at(prob_sum, indices_chunk, prob_chunk)
                pbar.update(chunk_end - i)
                del prob_chunk, indices_chunk  # Help GC
                gc.collect()

        if negative_found:
            return False  # Return False if negative probabilities were detected

        prob_sum.flush()  # Ensure sums are written

        # Check if any query sum is zero (or close to zero)
        is_zero = np.isclose(prob_sum, 0.0)
        any_zero = np.any(is_zero)

        if any_zero:
            log.warning("Zero probability sums found for some queries during validation.")

        result = not any_zero
        log.info(f"Probability validation result: {'Valid' if result else 'Invalid (zero sums found)'}")
        return result
    finally:
        # Cleanup memmap
        if prob_sum is not None:
            try:
                prob_sum.flush()
            except Exception as e:
                log.warning(f"Error flushing prob_sum_temp.mmap: {e}")
            del prob_sum
            gc.collect()
        try:
            if os.path.exists(prob_sum_file):
                os.unlink(prob_sum_file)
        except OSError as e:
            log.warning(f"Error deleting temporary file {prob_sum_file}: {e}")


def check_memory_requirements(prob_size_bytes: int, resource_manager: ResourceManager) -> bool:
    """
    Check if there's enough memory to run the SQUAREM algorithm safely,
    taking into account our chunking strategy.
    """
    array_size = prob_size_bytes // 8  # Assuming float64 (8 bytes)
    dtype_size = 8  # float64

    # Create dummy ArrayInfo for calculation (or could analyze a dummy array)
    # This part is slightly awkward as we don't have the actual array here.
    # We might need to pass more info or make assumptions.
    # Assuming a large array scale for safety in estimation.
    dummy_arr_info = resource_manager.analyze_array(np.empty(array_size, dtype=np.float64))

    # Estimate concurrent chunks needed for SQUAREM (q, r, r2, v, p_new, etc.) - let's estimate 6
    num_concurrent_squarem = 6
    strategy = resource_manager.calculate_chunk_size(dummy_arr_info, num_concurrent_chunks=num_concurrent_squarem)
    chunk_size = strategy.chunk_size

    # Calculate memory needed for a single chunk processing based on the strategy
    chunk_bytes = chunk_size * dtype_size
    # Use the num_concurrent_squarem estimate directly
    required_mem_for_chunks = chunk_bytes * num_concurrent_squarem

    # Add overhead for non-chunk data, Python objects, etc.
    # Use a factor slightly > 1, e.g., 1.3 (30% overhead)
    overhead_factor = 1.3
    total_required = required_mem_for_chunks * overhead_factor

    available_mem = resource_manager.available_memory # Use current available
    safe_ratio = available_mem / total_required if total_required > 0 else float('inf')

    log.info(f"Memory check for SQUAREM chunked processing:")
    log.info(f"  - Array size: {array_size:,} elements ({prob_size_bytes / (1024**3):.2f} GB)")
    log.info(f"  - Calculated chunk size: {chunk_size:,} elements ({chunk_bytes/(1024**3):.3f} GB)")
    log.info(f"  - Estimated concurrent chunk memory needed: {required_mem_for_chunks/(1024**3):.3f} GB")
    log.info(f"  - Estimated total memory required (with overhead): {total_required/(1024**3):.3f} GB")
    log.info(f"  - Currently available memory: {available_mem/(1024**3):.3f} GB")
    log.info(f"  - Safety ratio: {safe_ratio:.2f}")

    # Target ratio (e.g., > 1.1 for 10% headroom)
    target_ratio = 1.1
    if safe_ratio < target_ratio:
        log.warning(f"Available memory ({available_mem/(1024**3):.2f} GB) may be insufficient "
                   f"for SQUAREM algorithm with current chunk size ({chunk_size:,}).")
        log.warning(f"Required: ~{total_required/(1024**3):.2f} GB. Ratio: {safe_ratio:.2f} (target > {target_ratio}).")
        log.warning(f"Consider increasing available memory or expect potential slowdowns.")
        # Depending on policy, could return False here
        # return False

    return True # Or return safe_ratio >= target_ratio


def chunked_fixed_point_map(
    input_prob: np.memmap,
    mask: np.memmap,
    slen: np.memmap,
    query_inverse_indices: np.memmap,
    max_query: int,
    mmap_folder: str,
    resource_manager: ResourceManager,
) -> np.memmap:
    """Process fixed point mapping using chunked processing, avoiding large RAM copies."""
    # Estimate concurrent chunks: chunk_mask, prob_chunk, slen_chunk, s_w_chunk,
    # indices_in_chunk, global_indices, chunk_queries, chunk_probs, mask_sum_chunk ~ 6-7
    num_concurrent = 6
    arr_info = resource_manager.analyze_array(input_prob)
    strategy = resource_manager.calculate_chunk_size(arr_info, num_concurrent_chunks=num_concurrent)
    chunk_size = strategy.chunk_size

    new_prob_file = os.path.join(mmap_folder, "new_prob_temp.mmap")
    prob_sum_file = os.path.join(mmap_folder, "prob_sum_temp.mmap")

    try:
        # Initialize new_prob directly from input_prob
        new_prob = np.memmap(
            new_prob_file, dtype=np.float64, mode="w+", shape=input_prob.shape
        )
        for start in range(0, len(input_prob), chunk_size):
            end = min(start + chunk_size, len(input_prob))
            new_prob[start:end] = input_prob[start:end]
        new_prob.flush()
        gc.collect()

        # Calculate s_w = prob / slen and update new_prob = prob * s_w = prob * (prob / slen) = prob^2 / slen
        with tqdm(total=len(mask), desc="Updating new_prob", ncols=80) as pbar:
            for start in range(0, len(mask), chunk_size):
                end = min(start + chunk_size, len(mask))
                chunk_mask = mask[start:end]
                
                if np.any(chunk_mask):
                    prob_chunk = new_prob[start:end]
                    slen_chunk = slen[start:end]
                    valid_slen_mask = (slen_chunk > np.finfo(np.float64).tiny) & chunk_mask
                    s_w_chunk = np.zeros_like(prob_chunk)
                    s_w_chunk[valid_slen_mask] = prob_chunk[valid_slen_mask] / slen_chunk[valid_slen_mask]
                    update_indices = np.where(chunk_mask)[0]
                    if len(update_indices) > 0:
                         new_prob[start + update_indices] = prob_chunk[update_indices] * s_w_chunk[update_indices]

                pbar.update(end - start)
        new_prob.flush()
        gc.collect()

        # Create prob_sum as memory-mapped array
        prob_sum = np.memmap(
            prob_sum_file, dtype=np.float64, mode="w+", shape=(max_query + 1,)
        )
        prob_sum.fill(0)

        # Accumulate probabilities in chunks where mask is True
        with tqdm(total=len(mask), desc="Accumulating prob_sum", ncols=80) as pbar:
            for start in range(0, len(mask), chunk_size):
                end = min(start + chunk_size, len(mask))
                chunk_mask = mask[start:end]
                
                if np.any(chunk_mask):
                    masked_indices_in_chunk = np.where(chunk_mask)[0]
                    global_indices = start + masked_indices_in_chunk
                    chunk_queries = query_inverse_indices[global_indices]
                    chunk_probs = new_prob[global_indices]
                    np.add.at(prob_sum, chunk_queries, chunk_probs)

                pbar.update(end - start)
        prob_sum.flush()
        gc.collect()

        # Normalize probabilities where mask is True
        with tqdm(total=len(mask), desc="Normalizing new_prob", ncols=80) as pbar:
            for start in range(0, len(mask), chunk_size):
                end = min(start + chunk_size, len(mask))
                chunk_mask = mask[start:end]

                if np.any(chunk_mask):
                    masked_indices_in_chunk = np.where(chunk_mask)[0]
                    global_indices = start + masked_indices_in_chunk
                    chunk_queries = query_inverse_indices[global_indices]
                    mask_sum_chunk = prob_sum[chunk_queries]
                    valid_sum_mask = mask_sum_chunk > np.finfo(np.float64).tiny
                    update_indices = global_indices[valid_sum_mask]
                    if len(update_indices) > 0:
                        new_prob[update_indices] = new_prob[update_indices] / mask_sum_chunk[valid_sum_mask]
                    zero_sum_indices = global_indices[~valid_sum_mask]
                    if len(zero_sum_indices) > 0:
                         new_prob[zero_sum_indices] = 0.0

                pbar.update(end - start)
        new_prob.flush()
        gc.collect()

        return new_prob

    finally:
        for file_path in [prob_sum_file]:
             try:
                 if os.path.exists(file_path):
                     os.unlink(file_path)
             except OSError as e:
                 log.warning(f"Error deleting temporary file {file_path}: {e}")


def reassign(
    np_arrays: Dict[str, np.memmap],
    tmp_files: Dict[str, Any],
    iters: int = 25,
    step_min: float = -1.0,
    step_max: float = -1.0,
    mstep: int = 4,
    max_memory: Union[str, float, int] = "4G",
    num_threads: int = 1,
) -> str:
    """Reassign multimapped reads using memory-efficient implementation. Returns path to results parquet file."""
    resource_manager = ResourceManager()
    if isinstance(max_memory, str):
        max_memory = resource_manager.parse_memory_limit(max_memory)
    resource_manager = ResourceManager(max_memory=max_memory, max_threads=num_threads)
    mmap_folder = tmp_files["mmap"]
    db_folder = tmp_files["db"]
    output_parquet_file = os.path.join(db_folder, "reassigned_results.parquet")

    log.info("Creating inverse subject mapping")
    subject_inverse_indices, unique_subjects = memory_efficient_factorize(
        np_arrays["subject_numeric_id"],
        mmap_folder=mmap_folder,
        max_memory=resource_manager.max_memory,
        num_threads=num_threads,
    )

    log.info("Starting factorization of reads")
    query_inverse_indices, unique_queries = memory_efficient_factorize(
        np_arrays["query_numeric_id"],
        mmap_folder=mmap_folder,
        max_memory=resource_manager.max_memory,
        num_threads=num_threads,
    )

    log.info(f"Number of references: {len(unique_subjects):,}")
    log.info(f"Number of reads: {len(unique_queries):,}")

    iter_array_file = os.path.join(mmap_folder, "iter_array.mmap")
    prob_file = os.path.join(mmap_folder, "prob.dat")
    final_mask = None

    try:
        iter_array = np.memmap(
            iter_array_file, dtype=np.int64, mode="w+", shape=(np_arrays["subject_numeric_id"].shape[0],)
        )
        iter_array[:] = 0
        iter_array.flush()

        prob = np.memmap(
            prob_file, dtype=np.float64, mode="w+", shape=(np_arrays["subject_numeric_id"].shape[0],)
        )

        log.info("Initializing weights")
        chunked_initialize_weights(
            subject_inverse_indices,
            np_arrays["bitScore"],
            len(unique_subjects),
            mmap_folder,
            resource_manager,
            output_array=prob
        )
        prob.flush()
        gc.collect()

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
            resource_manager=resource_manager,
        )

        log.info(f"Writing filtered results to {output_parquet_file}")
        schema_fields = []
        output_columns = [
            "query_numeric_id", "subject_numeric_id", "bitScore", "alnLength",
            "subjectStart", "subjectEnd", "percIdentity", "row_hash"
        ]
        for col in output_columns:
             if col in np_arrays:
                 pa_type = pa.from_numpy_dtype(np_arrays[col].dtype)
                 schema_fields.append(pa.field(col, pa_type))
             else:
                 log.warning(f"Column {col} not found in input arrays, skipping.")
        
        schema = pa.schema(schema_fields)
        write_chunk_size = min(10_000_000, len(final_mask) // 10)
        write_chunk_size = max(write_chunk_size, 1_000_000)
        
        total_written = 0
        with pq.ParquetWriter(output_parquet_file, schema, compression='snappy') as writer:
            with tqdm(total=len(final_mask), desc="Writing results", ncols=80) as pbar:
                for start in range(0, len(final_mask), write_chunk_size):
                    end = min(start + write_chunk_size, len(final_mask))
                    mask_chunk = final_mask[start:end]
                    true_indices_in_chunk = np.where(mask_chunk)[0]
                    
                    if len(true_indices_in_chunk) > 0:
                        global_indices = start + true_indices_in_chunk
                        chunk_data = []
                        for col in output_columns:
                             if col in np_arrays:
                                 data_slice = np_arrays[col][global_indices]
                                 chunk_data.append(pa.array(data_slice))

                        if chunk_data:
                             table = pa.Table.from_arrays(chunk_data, schema=schema)
                             writer.write_table(table)
                             total_written += len(table)
                    
                    pbar.update(end - start)
                    gc.collect()

        log.info(f"Successfully wrote {total_written:,} rows to {output_parquet_file}")
        return output_parquet_file

    finally:
        log.debug("Cleaning up reassign temporary files and memmaps")
        if 'subject_inverse_indices' in locals() and subject_inverse_indices._mmap is not None: del subject_inverse_indices
        if 'unique_subjects' in locals(): del unique_subjects
        if 'query_inverse_indices' in locals() and query_inverse_indices._mmap is not None: del query_inverse_indices
        if 'unique_queries' in locals(): del unique_queries
        if 'iter_array' in locals() and iter_array._mmap is not None: del iter_array
        if 'prob' in locals() and prob._mmap is not None: del prob
        if final_mask is not None and isinstance(final_mask, np.memmap) and final_mask._mmap is not None: del final_mask
        gc.collect()

        for pattern in ["subject_*.dat", "query_*.dat", "iter_array.mmap", "prob.dat", "mask.dat", "*_temp.mmap", "*_temp.dat", "total_weights.dat", "weights_result.dat"]:
            for filename in glob.glob(os.path.join(mmap_folder, pattern)):
                try:
                    if os.path.exists(filename):
                        os.unlink(filename)
                except OSError as e:
                    log.warning(f"Error deleting temporary file {filename}: {e}")
        gc.collect()
