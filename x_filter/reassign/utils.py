import logging
import os
import time
import uuid
import gc
import numpy as np
import psutil
import tqdm
from numba import njit, prange, set_num_threads 
from typing import Tuple, Optional

log = logging.getLogger("my_logger")



@njit(fastmath=True)
def fast_sum_by_index(indices, values, out):
    """
    Numba-accelerated sum by index (like np.add.at).
    """
    for i in range(len(indices)):
        out[indices[i]] += values[i]

@njit(fastmath=True, parallel=True)
def fast_uniform_subject_weights(subject_indices, subject_weights_out):
    """
    Extremely fast uniform weight initialization using Numba.
    """
    max_subject = np.max(subject_indices)
    uniform_weight = 1.0 / (max_subject + 1)
    
    for i in prange(len(subject_indices)):
        subject_weights_out[i] = uniform_weight

def initialize_subject_weights(data, mmap_dir=None, max_memory=None):
    """
    Ultra-fast uniform initialization using Numba for maximum speed.
    
    Args:
        data: Structured array with fields: source, subject, var, slen, etc.
        mmap_dir: Directory for memory-mapped files (unused)
        max_memory: Maximum memory to use (unused)
        
    Returns:
        Updated structured array with initialized s_W field (prob will be set by EM)
    """
    if data.shape[0] == 0:
        log.warning("Empty data array - cannot initialize weights")
        return data
    
    log.info("Using ultra-fast uniform subject weight initialization")
    
    # Use Numba for maximum speed
    fast_uniform_subject_weights(data["subject"], data["s_W"])
    
    # Initialize other fields
    data["prob"].fill(0.0)  # Will be computed by EM E-step
    data["iter"].fill(0)
    data["n_aln"].fill(0)
    data["max_prob"].fill(0.0)
    
    log.info("Ultra-fast uniform subject weights initialized")
    return data

@njit(parallel=True)
def parallel_unique_sort(arr: np.ndarray) -> np.ndarray:
    """Find unique values using parallel sorting."""
    sorted_arr = np.sort(arr)
    mask = np.ones(len(sorted_arr), dtype=np.bool_)

    for i in prange(1, len(sorted_arr)):
        mask[i] = sorted_arr[i] != sorted_arr[i - 1]

    return sorted_arr[mask]

def process_chunk(chunk_data):
    """Process a single chunk to find unique values."""
    chunk_sorted = np.sort(chunk_data)
    return chunk_sorted[np.concatenate(([True], chunk_sorted[1:] != chunk_sorted[:-1]))]

@njit(parallel=True, cache=False)
def create_inverse_chunk(chunk_data, unique_vals, output, num_threads=1):
    """Numba-accelerated chunk processing for finding indices efficiently."""
    # Remove the invalid import - numba.set_num_threads should be called outside
    chunk_size = (len(chunk_data) + num_threads - 1) // num_threads
    unique_len = len(unique_vals)
    unique_min = unique_vals[0]
    unique_max = unique_vals[-1]
    
    for thread_id in prange(num_threads):
        start_idx = thread_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(chunk_data))
        
        for i in range(start_idx, end_idx):
            target = chunk_data[i]
            
            # Expanded fast path for common cases
            if target == unique_max:
                output[i] = unique_len - 1
                continue
            elif target > unique_max:
                output[i] = unique_len - 1
                continue
            elif target < unique_min:
                output[i] = 0
                continue
                
            # Use binary search for the general case
            left, right = 0, unique_len - 1
            
            while left <= right:
                mid = (left + right) >> 1
                mid_val = unique_vals[mid]
                
                if mid_val == target:
                    output[i] = mid
                    break
                elif mid_val < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            # If we didn't find an exact match
            if left > right:
                output[i] = left if left < unique_len else right

# Helper function to set threads before calling create_inverse_chunk
def prepare_and_create_inverse_chunk(chunk_data, unique_vals, output, num_threads=1):
    """Wrapper to set number of threads before calling create_inverse_chunk"""
    # Set number of threads outside the JIT-compiled function
    set_num_threads(num_threads)
    # Now call the JIT-compiled function
    create_inverse_chunk(chunk_data, unique_vals, output, num_threads)

def memory_efficient_factorize(
    arr: np.ndarray,
    mmap_dir: str,
    threads: int = 1,
    max_memory: Optional[int] = None,  # Maximum memory to use (in bytes)
    chunk_size: Optional[int] = None,  # Size of chunks to process (if None, calculated automatically)
    min_chunk_size: int = 1_000_000,  # Minimum chunk size of 1M
    max_chunk_size: Optional[int] = None,  # No maximum chunk size - use all available memory
    return_mmap: bool = True  # New parameter to control return type
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient factorization using chunked processing and memory mapping.
    
    Args:
        arr: Input array to factorize
        mmap_dir: Directory for temporary memory-mapped files
        threads: Number of threads to use
        chunk_size: Size of chunks to process (if None, calculated automatically)
        return_mmap: If True, return memory-mapped arrays (caller must manage cleanup)
    
    Returns:
        Tuple of (inverse mapping array, unique values array)
    """
    from x_filter.resource_management import ResourceManager
    
    # Create resource manager for optimal chunk sizing
    resource_manager = ResourceManager(max_threads=threads, max_memory=max_memory, mmap_folder=mmap_dir)
    
    if chunk_size is None:
        # Calculate optimal chunk size with no artificial limits
        chunk_size = resource_manager.calculate_optimal_chunk_size(
            total_elements=len(arr),
            element_size=arr.dtype.itemsize,
            operation_overhead=3.0,  # Account for sorting and uniqueness operations
            min_chunk_size=min_chunk_size,  # 1M minimum
            max_chunk_size=max_chunk_size  # No maximum - use all available memory
        )
    
    log.info(f"Factorizing {len(arr):,} elements using chunk size: {chunk_size:,}")
    
    # Create memory-mapped arrays for intermediate storage
    inverse_temp_path = os.path.join(mmap_dir, f"factorize_inverse_{id(arr)}.mmap")
    unique_temp_path = os.path.join(mmap_dir, f"factorize_unique_{id(arr)}.mmap")
    
    try:
        # Process in chunks to find all unique values
        unique_values_list = []
        
        if len(arr) <= chunk_size:
            # Small enough to process in one chunk
            unique_values = np.unique(arr)
        else:
            # Process in chunks and merge unique values
            for start_idx in range(0, len(arr), chunk_size):
                end_idx = min(start_idx + chunk_size, len(arr))
                chunk = arr[start_idx:end_idx]
                chunk_unique = np.unique(chunk)
                unique_values_list.append(chunk_unique)
            
            # Merge all unique values and get final unique set
            if unique_values_list:
                all_unique = np.concatenate(unique_values_list)
                unique_values = np.unique(all_unique)
            else:
                unique_values = np.array([], dtype=arr.dtype)  # Fixed: use arr.dtype instead of array_data.dtype
        
        log.info(f"Found {len(unique_values):,} unique values")
        
        # Create memory-mapped arrays for output
        inverse_mmap = np.memmap(
            inverse_temp_path,
            dtype=np.int64,  # Always use int64 for indices
            mode='w+',
            shape=(len(arr),)
        )
        
        unique_mmap = np.memmap(
            unique_temp_path,
            dtype=arr.dtype,  # Fixed: use arr.dtype instead of array_data.dtype
            mode='w+',
            shape=(len(unique_values),)
        )
        
        # Store unique values
        unique_mmap[:] = unique_values
        unique_mmap.flush()
        
        # Create inverse mapping in chunks
        for start_idx in range(0, len(arr), chunk_size):
            end_idx = min(start_idx + chunk_size, len(arr))
            chunk = arr[start_idx:end_idx]
            
            # Find indices for this chunk using searchsorted
            chunk_inverse = np.searchsorted(unique_values, chunk)
            inverse_mmap[start_idx:end_idx] = chunk_inverse
        
        inverse_mmap.flush()
        
        if return_mmap:
            # Return memory-mapped arrays directly - caller manages cleanup
            log.info("Returning memory-mapped arrays - caller responsible for cleanup")
            return inverse_mmap, unique_mmap
        else:
            # Convert to regular arrays for standard compatibility and automatic cleanup
            # This ensures the function works with all numpy operations and the temp files are cleaned up
            inverse_result = np.array(inverse_mmap, dtype=np.int64)
            unique_result = np.array(unique_mmap, dtype=arr.dtype)
            
            return inverse_result, unique_result
        
    finally:
        # Clean up temporary files only if not returning mmap arrays
        if not return_mmap:
            try:
                if os.path.exists(inverse_temp_path):
                    os.remove(inverse_temp_path)
                if os.path.exists(unique_temp_path):
                    os.remove(unique_temp_path)
            except Exception as e:
                log.debug(f"Could not remove temporary factorization files: {e}")


@njit(parallel=True)
def compute_best_assignments(source_indices, subject_indices, probabilities, max_source, output_array):
    """Find best assignment for each read based on maximum probability."""
    # Initialize best probability tracking arrays
    best_probs = np.full(max_source + 1, -1.0)
    
    # Find best assignment for each source in parallel
    for i in prange(len(source_indices)):
        source_id = source_indices[i]
        subject_id = subject_indices[i]
        prob = probabilities[i]
        
        # Use atomic updates to track best probability for each source
        if prob > best_probs[source_id]:
            best_probs[source_id] = prob
            output_array[source_id] = subject_id

@njit(parallel=True)
def create_inverse_chunk(chunk_data, unique_vals, output, num_threads=1):
    """Numba-accelerated chunk processing for finding indices efficiently."""
    set_num_threads(num_threads)
    
    chunk_size = (len(chunk_data) + num_threads - 1) // num_threads
    unique_len = len(unique_vals)
    unique_min = unique_vals[0]
    unique_max = unique_vals[-1]
    
    for thread_id in prange(num_threads):
        start_idx = thread_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(chunk_data))
        
        for i in range(start_idx, end_idx):
            target = chunk_data[i]
            
            # Expanded fast path for common cases
            if target == unique_max:
                output[i] = unique_len - 1
                continue
            elif target > unique_max:
                output[i] = unique_len - 1
                continue
            elif target < unique_min:
                output[i] = 0
                continue
                
            # Use binary search for the general case
            left, right = 0, unique_len - 1
            
            while left <= right:
                mid = (left + right) >> 1
                mid_val = unique_vals[mid]
                
                if mid_val == target:
                    output[i] = mid
                    break
                elif mid_val < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            # If we didn't find an exact match
            if left > right:
                output[i] = left if left < unique_len else right

def cleanup_temp_files(file_paths):
    """Clean up temporary memory-mapped files."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            log.warning(f"Could not remove temporary file {path}: {e}")
    gc.collect()
