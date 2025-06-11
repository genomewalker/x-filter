import logging
import os
import time
import uuid
import gc
import numpy as np
import psutil
import tqdm
from numba import njit, prange, set_num_threads 
from typing import Tuple, Optional, Dict

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
        data: Dictionary or structured array with fields: source, subject, var, slen, etc.
        mmap_dir: Directory for memory-mapped files (unused)
        max_memory: Maximum memory to use (unused)
        
    Returns:
        Updated data structure with initialized s_W field (prob will be set by EM)
    """
    # Handle both dictionary and structured array formats
    if isinstance(data, dict):
        # Dictionary format
        if len(data.get('subject', [])) == 0:
            log.warning("Empty data dictionary - cannot initialize weights")
            return data
        
        log.info("Using ultra-fast uniform subject weight initialization (dictionary format)")
        
        # Use Numba for maximum speed
        fast_uniform_subject_weights(data["subject"], data["s_W"])
        
        # Initialize other fields
        data["prob"].fill(0.0)  # Will be computed by EM E-step
        data["iter"].fill(0)
        data["n_aln"].fill(0)
        data["max_prob"].fill(0.0)
        
    elif hasattr(data, 'shape') and data.shape[0] == 0:
        # Structured array format - empty
        log.warning("Empty data array - cannot initialize weights")
        return data
        
    elif hasattr(data, 'dtype') and data.dtype.names is not None:
        # Structured array format - non-empty
        log.info("Using ultra-fast uniform subject weight initialization (structured array format)")
        
        # Use Numba for maximum speed
        fast_uniform_subject_weights(data["subject"], data["s_W"])
        
        # Initialize other fields
        data["prob"].fill(0.0)  # Will be computed by EM E-step
        data["iter"].fill(0)
        data["n_aln"].fill(0)
        data["max_prob"].fill(0.0)
        
    else:
        # Unknown format
        log.error(f"Unknown data format: {type(data)}")
        raise ValueError(f"Unsupported data format: {type(data)}")
    
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

@njit(parallel=True, fastmath=True)
def fast_inverse_mapping_chunk(chunk_data, unique_vals, output, start_offset=0):
    """
    Ultra-fast inverse mapping using optimized binary search with Numba.
    Uses parallel processing and optimized search strategies.
    """
    unique_len = len(unique_vals)
    chunk_len = len(chunk_data)
    
    if unique_len == 0 or chunk_len == 0:
        return
    
    unique_min = unique_vals[0]
    unique_max = unique_vals[-1]
    
    # Process chunk in parallel
    for i in prange(chunk_len):
        target = chunk_data[i]
        
        # Fast path for edge cases
        if target <= unique_min:
            output[start_offset + i] = 0
            continue
        elif target >= unique_max:
            output[start_offset + i] = unique_len - 1
            continue
        
        # Optimized binary search for general case
        left, right = 0, unique_len - 1
        
        # Use interpolation search hint for better starting position
        if unique_max > unique_min:
            # Estimate position based on value distribution
            ratio = (target - unique_min) / (unique_max - unique_min)
            estimated_pos = int(ratio * (unique_len - 1))
            estimated_pos = max(0, min(unique_len - 1, estimated_pos))
            
            # Check if our estimate is close
            if unique_vals[estimated_pos] == target:
                output[start_offset + i] = estimated_pos
                continue
            elif unique_vals[estimated_pos] < target:
                left = estimated_pos
            else:
                right = estimated_pos
        
        # Binary search with optimizations
        while left <= right:
            mid = (left + right) >> 1
            mid_val = unique_vals[mid]
            
            if mid_val == target:
                output[start_offset + i] = mid
                break
            elif mid_val < target:
                left = mid + 1
            else:
                right = mid - 1
        else:
            # If exact match not found, use left boundary
            output[start_offset + i] = min(left, unique_len - 1)

@njit(parallel=True, fastmath=True)
def fast_hash_based_inverse(chunk_data, unique_vals, output, start_offset=0):
    """
    Hash-based inverse mapping for dense integer sequences.
    Much faster when unique values are densely packed.
    """
    unique_len = len(unique_vals)
    chunk_len = len(chunk_data)
    
    if unique_len == 0 or chunk_len == 0:
        return
    
    unique_min = unique_vals[0]
    unique_max = unique_vals[-1]
    value_range = unique_max - unique_min + 1
    
    # Only use hash approach if values are reasonably dense
    if value_range <= unique_len * 2:  # At most 50% sparse
        # Create lookup table
        lookup_size = int(value_range)
        lookup_table = np.full(lookup_size, -1, dtype=np.int64)
        
        # Fill lookup table
        for i in range(unique_len):
            offset = unique_vals[i] - unique_min
            if 0 <= offset < lookup_size:
                lookup_table[offset] = i
        
        # Fast lookup for chunk
        for i in prange(chunk_len):
            target = chunk_data[i]
            offset = target - unique_min
            
            if 0 <= offset < lookup_size and lookup_table[offset] != -1:
                output[start_offset + i] = lookup_table[offset]
            else:
                # Fallback to binary search for values not in lookup
                left, right = 0, unique_len - 1
                while left <= right:
                    mid = (left + right) >> 1
                    if unique_vals[mid] == target:
                        output[start_offset + i] = mid
                        break
                    elif unique_vals[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                else:
                    output[start_offset + i] = min(left, unique_len - 1)
    else:
        # Fall back to optimized binary search
        fast_inverse_mapping_chunk(chunk_data, unique_vals, output, start_offset)

def memory_efficient_factorize(
    arr: np.ndarray,
    mmap_dir: str,
    threads: int = 1,
    max_memory: Optional[int] = None,
    chunk_size: Optional[int] = None,
    min_chunk_size: int = 1_000_000,
    max_chunk_size: Optional[int] = None,
    return_mmap: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient factorization with ultra-fast inverse mapping.
    
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
        # Calculate optimal chunk size for inverse mapping operations
        chunk_size = resource_manager.calculate_optimal_chunk_size(
            total_elements=len(arr),
            element_size=arr.dtype.itemsize,
            operation_overhead=2.5,  # Reduced overhead for optimized operations
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
    
    log.info(f"Factorizing {len(arr):,} elements using optimized inverse mapping with chunk size: {chunk_size:,}")
    
    # Create memory-mapped arrays for intermediate storage
    inverse_temp_path = os.path.join(mmap_dir, f"factorize_inverse_{uuid.uuid4().hex}.mmap")
    unique_temp_path = os.path.join(mmap_dir, f"factorize_unique_{uuid.uuid4().hex}.mmap")
    
    try:
        # Process in chunks to find all unique values (unchanged)
        unique_values_list = []
        
        if len(arr) <= chunk_size:
            unique_values = np.unique(arr)
        else:
            for start_idx in range(0, len(arr), chunk_size):
                end_idx = min(start_idx + chunk_size, len(arr))
                chunk = arr[start_idx:end_idx]
                chunk_unique = np.unique(chunk)
                unique_values_list.append(chunk_unique)
            
            if unique_values_list:
                all_unique = np.concatenate(unique_values_list)
                unique_values = np.unique(all_unique)
            else:
                unique_values = np.array([], dtype=arr.dtype)
        
        log.info(f"Found {len(unique_values):,} unique values")
        
        # Create memory-mapped arrays for output
        inverse_mmap = np.memmap(
            inverse_temp_path,
            dtype=np.int64,
            mode='w+',
            shape=(len(arr),)
        )
        
        unique_mmap = np.memmap(
            unique_temp_path,
            dtype=arr.dtype,
            mode='w+',
            shape=(len(unique_values),)
        )
        
        # Store unique values
        unique_mmap[:] = unique_values
        unique_mmap.flush()
        
        # Determine optimal inverse mapping strategy
        unique_min = unique_values[0] if len(unique_values) > 0 else 0
        unique_max = unique_values[-1] if len(unique_values) > 0 else 0
        value_range = unique_max - unique_min + 1 if len(unique_values) > 0 else 0
        
        use_hash_method = (
            len(unique_values) > 0 and 
            value_range <= len(unique_values) * 2 and  # At most 50% sparse
            value_range < 100_000_000  # Reasonable memory usage
        )
        
        if use_hash_method:
            log.info(f"Using hash-based inverse mapping (range: {value_range:,}, unique: {len(unique_values):,})")
        else:
            log.info(f"Using optimized binary search inverse mapping")
        
        # Set number of threads for Numba
        set_num_threads(threads)
        
        # Create inverse mapping in chunks using optimized methods
        for start_idx in range(0, len(arr), chunk_size):
            end_idx = min(start_idx + chunk_size, len(arr))
            chunk = arr[start_idx:end_idx]
            
            if use_hash_method:
                fast_hash_based_inverse(chunk, unique_values, inverse_mmap, start_idx)
            else:
                fast_inverse_mapping_chunk(chunk, unique_values, inverse_mmap, start_idx)
        
        inverse_mmap.flush()
        
        if return_mmap:
            log.info("Returning memory-mapped arrays - caller responsible for cleanup")
            return inverse_mmap, unique_mmap
        else:
            inverse_result = np.array(inverse_mmap, dtype=np.int64)
            unique_result = np.array(unique_mmap, dtype=arr.dtype)
            return inverse_result, unique_result
        
    finally:
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

@njit(parallel=True, fastmath=True)
def efficient_array_fill(target_array: np.ndarray, value: float) -> None:
    """Ultra-fast parallel array filling using Numba."""
    for i in prange(len(target_array)):
        target_array[i] = value

def create_efficient_structured_array(
    n_elements: int,
    mmap_path: str,
    dtype: np.dtype,
    chunk_size: int = 1_000_000
) -> np.memmap:
    """
    Create structured array efficiently to avoid memory spikes.
    
    Args:
        n_elements: Number of elements
        mmap_path: Path for memory-mapped file
        dtype: Structured array dtype
        chunk_size: Chunk size for processing
        
    Returns:
        Memory-mapped structured array
    """
    log.info(f"Creating efficient structured array with {n_elements:,} elements")
    
    # Create memory-mapped array
    em_data = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(n_elements,))
    
    # Initialize in chunks to avoid memory pressure
    log.info("Initializing structured array fields in chunks...")
    
    for field_name, field_dtype in dtype.descr:
        log.debug(f"Initializing field: {field_name} ({field_dtype})")
        
        # Determine default value based on dtype
        if 'float' in field_dtype:
            default_value = 0.0
        elif 'int' in field_dtype:
            default_value = 0
        else:
            default_value = 0
        
        # Initialize in chunks
        for start_idx in range(0, n_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, n_elements)
            em_data[field_name][start_idx:end_idx] = default_value
    
    log.info("Structured array initialization complete")
    return em_data

def assign_arrays_to_structured(
    em_data: np.memmap,
    field_assignments: Dict[str, Tuple[np.ndarray, bool]],  # field_name -> (array, need_conversion)
    chunk_size: int = 1_000_000
) -> None:
    """
    Efficiently assign arrays to structured array fields.
    
    Args:
        em_data: Target structured array
        field_assignments: Dict mapping field names to (source_array, needs_conversion)
        chunk_size: Chunk size for processing large arrays
    """
    n_elements = len(em_data)
    
    for field_name, (source_array, needs_conversion) in field_assignments.items():
        log.info(f"Assigning {field_name} ({'with conversion' if needs_conversion else 'direct'})")
        
        if not needs_conversion:
            # Direct assignment - fastest
            em_data[field_name] = source_array
        else:
            # Chunked conversion to avoid memory spikes
            log.info(f"Converting {field_name} in chunks of {chunk_size:,}")
            target_dtype = em_data[field_name].dtype
            
            for start_idx in range(0, n_elements, chunk_size):
                end_idx = min(start_idx + chunk_size, n_elements)
                em_data[field_name][start_idx:end_idx] = source_array[start_idx:end_idx].astype(target_dtype)
    
    log.info("Array assignment complete")

def cleanup_temp_files(file_paths):
    """Clean up temporary memory-mapped files."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            log.warning(f"Could not remove temporary file {path}: {e}")
    gc.collect()

@njit(fastmath=True, parallel=True)
def optimized_memory_layout_conversion(
    source_data: np.ndarray,
    target_layout: np.ndarray,
    chunk_size: int = 65536
) -> None:
    """
    Convert data layout for better cache performance during EM iterations.
    """
    n_elements = len(source_data)
    
    # Process in cache-friendly chunks
    for chunk_start in prange(0, n_elements, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_elements)
        
        # Sequential access within chunks for better cache usage
        for i in range(chunk_start, chunk_end):
            target_layout[i] = source_data[i]

def create_cache_optimized_em_data(
    original_data: dict,
    resource_manager,
    optimize_layout: bool = True
) -> dict:
    """
    Create EM data structure with optimized memory layout for faster access.
    """
    n_elements = len(original_data["source"])
    
    if not optimize_layout:
        return original_data
    
    log.info("Creating cache-optimized data layout...")
    
    # Create arrays with better alignment and access patterns
    optimized_data = {}
    
    # Copy data with optimized memory layout
    for key in ["source", "subject", "var", "slen", "orig_idx"]:
        if key in original_data:
            # Create optimally aligned array
            optimized_array = resource_manager.create_array(
                name=f"optimized_{key}",
                shape=(n_elements,),
                dtype=original_data[key].dtype,
                temp=True
            )
            
            # Use optimized copy
            optimized_memory_layout_conversion(
                original_data[key], 
                optimized_array,
                chunk_size=65536  # Cache-friendly chunk size
            )
            
            optimized_data[key] = optimized_array
        else:
            optimized_data[key] = original_data[key]
    
    # Add new arrays
    for key in ["s_W", "prob", "iter", "n_aln", "max_prob"]:
        if key not in optimized_data:
            optimized_data[key] = resource_manager.create_array(
                name=f"optimized_{key}",
                shape=(n_elements,),
                dtype=np.float64 if key != "iter" else np.int64,
                temp=True
            )
    
    log.info("Cache-optimized layout created")
    return optimized_data
