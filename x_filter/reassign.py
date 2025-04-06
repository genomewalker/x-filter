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
    if len(sorted_arr) <= 1:
        return sorted_arr

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
        new_array = np.memmap(new_file, dtype=dtype, mode="w+", shape=(new_size,))
        new_array[: len(data)] = data
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
    arr_info = resource_manager.analyze_array(input_array)
    strategy = resource_manager.calculate_chunk_size(arr_info)
    chunk_size = strategy.chunk_size

    unique_values_file = os.path.join(mmap_folder, f"{prefix}_unique_values.dat")
    inverse_indices_file = os.path.join(mmap_folder, f"{prefix}_inverse_indices.dat")
    temp_merge_file = os.path.join(mmap_folder, f"{prefix}_temp_merge.dat")

    try:
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

        unique_values = np.memmap(
            unique_values_file,
            dtype=input_array.dtype,
            mode="w+",
            shape=(initial_size,),
        )

        current_unique_count = 0
        with tqdm(
            total=len(input_array), desc=f"Processing {prefix}", ncols=80
        ) as pbar:
            for start in range(0, len(input_array), chunk_size):
                end = min(start + chunk_size, len(input_array))
                chunk = input_array[start:end]
                chunk_uniques = parallel_unique_sort(chunk)

                if current_unique_count > 0:
                    existing_uniques = unique_values[:current_unique_count]
                    merged = merge_sorted_unique(existing_uniques, chunk_uniques)

                    if len(merged) > len(unique_values):
                        growth_factor = 1.5
                        new_size = min(
                            int(len(merged) * growth_factor), len(input_array)
                        )

                        log.info(f"Resizing {prefix} array to {new_size:,} elements")

                        unique_values = resize_memmap(
                            unique_values_file,
                            temp_merge_file,
                            input_array.dtype,
                            new_size,
                            merged,
                        )

                        os.rename(temp_merge_file, unique_values_file)

                    else:
                        unique_values[: len(merged)] = merged

                    current_unique_count = len(merged)
                else:
                    if len(chunk_uniques) > len(unique_values):
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

        actual_size = current_unique_count
        log.info(f"Final unique {prefix} count: {actual_size:,}")

        os.truncate(unique_values_file, actual_size * unique_values.dtype.itemsize)

        unique_values = np.memmap(
            unique_values_file, dtype=input_array.dtype, mode="r+", shape=(actual_size,)
        )

        inverse_indices = np.memmap(
            inverse_indices_file, dtype=np.int64, mode="w+", shape=(len(input_array),)
        )

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
    n_threads = numba.get_num_threads()
    local_outputs = np.zeros((n_threads, len(output)), dtype=output.dtype)

    for i in prange(len(chunk_indices)):
        thread_id = numba.get_thread_id()
        idx = chunk_indices[i]
        local_outputs[thread_id, idx] += chunk_scores[i]

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
    """Initialize weights using chunked processing with resource management."""
    arr_info = resource_manager.analyze_array(subject_inverse_indices)
    strategy = resource_manager.calculate_chunk_size(arr_info)
    chunk_size = strategy.chunk_size

    total_weights_file = os.path.join(mmap_folder, "total_weights.dat")
    total_weights = np.memmap(
        total_weights_file, dtype=np.float64, mode="w+", shape=(max_index,)
    )
    total_weights.fill(0)

    try:
        with tqdm(
            total=len(subject_inverse_indices), desc="Calculating weights", ncols=80
        ) as pbar:
            for start in range(0, len(subject_inverse_indices), chunk_size):
                end = min(start + chunk_size, len(subject_inverse_indices))
                chunk_indices = subject_inverse_indices[start:end]
                chunk_scores = bitScore[start:end]
                parallel_accumulate_weights(chunk_indices, chunk_scores, total_weights)
                pbar.update(end - start)

        total_weights[total_weights == 0] = np.finfo(np.float64).tiny

        result_file = os.path.join(mmap_folder, "weights_result.dat")
        result = np.memmap(
            result_file, dtype=np.float64, mode="w+", shape=bitScore.shape
        )

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

    prob_sum_file = os.path.join(mmap_folder, "prob_sum_temp.mmap")
    try:
        prob_sum = np.memmap(
            prob_sum_file, dtype=np.float64, mode="w+", shape=(max_query + 1,)
        )
        prob_sum.fill(0)

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

        prob_sum_file = os.path.join(mmap_folder, "prob_sum_fixed_point.mmap")
        prob_sum = np.memmap(
            prob_sum_file, dtype=np.float64, mode="w+", shape=(max_query + 1,)
        )
        prob_sum.fill(0)

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
            if os.path.exists(prob_sum_file):
                os.unlink(prob_sum_file)
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
    resource_manager: Optional[ResourceManager] = None,
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

    r_file = os.path.join(mmap_folder, "r_temp.mmap")
    r = np.memmap(r_file, dtype=np.float64, mode="w+", shape=prob.shape)

    chunk_size = min(100_000_000, len(prob))
    for start in range(0, len(prob), chunk_size):
        end = min(start + chunk_size, len(prob))
        r[start:end] = q[start:end] - prob[start:end]

    sr2 = 0.0
    for start in range(0, len(r), chunk_size):
        end = min(start + chunk_size, len(r))
        sr2 += np.sum(r[start:end] ** 2)

    if sr2 < 1e-10:
        try:
            os.unlink(r_file)
        except OSError:
            pass
        return q

    q2 = chunked_fixed_point_map(
        q, mask, slen, query_inverse_indices, max_query, mmap_folder, resource_manager
    )

    r2_file = os.path.join(mmap_folder, "r2_temp.mmap")
    v_file = os.path.join(mmap_folder, "v_temp.mmap")

    r2 = np.memmap(r2_file, dtype=np.float64, mode="w+", shape=prob.shape)
    v = np.memmap(v_file, dtype=np.float64, mode="w+", shape=prob.shape)

    for start in range(0, len(q2), chunk_size):
        end = min(start + chunk_size, len(q2))
        r2[start:end] = q2[start:end] - q[start:end]
        v[start:end] = r2[start:end] - r[start:end]

    srv_file = os.path.join(mmap_folder, "squarem_srv_temp.mmap")
    sv2_file = os.path.join(mmap_folder, "squarem_sv2_temp.mmap")

    try:
        srv_array = np.memmap(srv_file, dtype=np.float64, mode="w+", shape=(1,))
        sv2_array = np.memmap(sv2_file, dtype=np.float64, mode="w+", shape=(1,))
        srv_array[0] = 0.0
        sv2_array[0] = 0.0

        for start in range(0, len(v), chunk_size):
            end = min(start + chunk_size, len(v))
            sv2_array[0] += np.sum(v[start:end] ** 2)
            srv_array[0] += np.sum(r[start:end] * v[start:end])

        sv2 = sv2_array[0]
        srv = srv_array[0]

        if sv2 < 1e-10:
            for file_path in [r_file, r2_file, v_file]:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except OSError:
                    pass
            return q2

        if step_min < 0:
            step_min = 0.001
        if step_max < step_min:
            step_max = 1.0

        alpha = np.sqrt(sr2 / sv2)
        alpha = np.clip(alpha, step_min, step_max)

        p_new_file = os.path.join(mmap_folder, "p_new_temp.mmap")
        result_file = os.path.join(mmap_folder, "squarem_result.mmap")

        try:
            p_new = np.memmap(p_new_file, dtype=np.float64, mode="w+", shape=prob.shape)

            for start in range(0, len(prob), chunk_size):
                end = min(start + chunk_size, len(prob))
                p_new[start:end] = prob[start:end] + 2 * alpha * r[start:end] + alpha * alpha * v[start:end]

            if validate_probabilities(
                p_new[mask], query_inverse_indices[mask], max_query, mmap_folder
            ):
                result = chunked_fixed_point_map(
                    p_new,
                    mask,
                    slen,
                    query_inverse_indices,
                    max_query,
                    mmap_folder,
                    resource_manager,
                )

                final_result = np.memmap(
                    result_file, dtype=np.float64, mode="w+", shape=prob.shape
                )

                for start in range(0, len(result), chunk_size):
                    end = min(start + chunk_size, len(result))
                    final_result[start:end] = result[start:end]

                del result
                gc.collect()

                if validate_probabilities(
                    final_result[mask], query_inverse_indices[mask], max_query, mmap_folder
                ):
                    return final_result

            for m in range(mstep):
                alpha = alpha / 2
                for start in range(0, len(prob), chunk_size):
                    end = min(start + chunk_size, len(prob))
                    p_new[start:end] = prob[start:end] + 2 * alpha * r[start:end] + alpha * alpha * v[start:end]

                if validate_probabilities(
                    p_new[mask], query_inverse_indices[mask], max_query, mmap_folder
                ):
                    result = chunked_fixed_point_map(
                        p_new,
                        mask,
                        slen,
                        query_inverse_indices,
                        max_query,
                        mmap_folder,
                        resource_manager,
                    )

                    final_result = np.memmap(
                        result_file, dtype=np.float64, mode="w+", shape=prob.shape
                    )

                    for start in range(0, len(result), chunk_size):
                        end = min(start + chunk_size, len(result))
                        final_result[start:end] = result[start:end]

                    del result
                    gc.collect()

                    if validate_probabilities(
                        final_result[mask], query_inverse_indices[mask], max_query, mmap_folder
                    ):
                        return final_result

            final_result = np.memmap(
                result_file, dtype=np.float64, mode="w+", shape=prob.shape
            )

            for start in range(0, len(q2), chunk_size):
                end = min(start + chunk_size, len(q2))
                final_result[start:end] = q2[start:end]

            return final_result

        finally:
            for file_path in [r_file, r2_file, v_file, p_new_file, srv_file, sv2_file]:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except OSError:
                    pass

            gc.collect()

    finally:
        for file_path in [srv_file, sv2_file]:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except OSError:
                pass
