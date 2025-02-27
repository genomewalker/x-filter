# x_filter/db/mmap.py

import os
import gc
import numpy as np
from typing import Dict, List, Set, Tuple, Union, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import numba
from numba import njit, prange

from x_filter.utils.logging import get_logger
from x_filter.utils.memory import track_memory

log = get_logger(__name__)


class MemoryMappedArray:
    """Wrapper for numpy memory-mapped arrays with enhanced functionality."""

    def __init__(
        self,
        filename: str,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        mode: str = "r+",
        create: bool = False,
    ):
        """Initialize a memory-mapped array.

        Args:
            filename: Path to the memory-mapped file
            dtype: Data type of the array
            shape: Shape of the array
            mode: File mode ('r+' for read/write, 'r' for read-only, 'w+' for create/overwrite)
            create: Whether to create the directory if it doesn't exist
        """
        self.filename = filename
        self.dtype = dtype
        self.shape = shape
        self.mode = mode

        if create:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        self.array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __len__(self):
        return len(self.array)

    def flush(self):
        """Flush changes to disk."""
        self.array.flush()

    def close(self):
        """Close the memory-mapped array."""
        del self.array
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def initialize_mmap_array(
    total_positions: int,
    dtype: np.dtype,
    mmap_folder: str,
    array_name: str = "flattened_coverage",
) -> np.ndarray:
    """
    Initialize memory-mapped array.

    Args:
        total_positions: Total number of elements in the array
        dtype: Data type of the array
        mmap_folder: Path to memory-mapped folder
        array_name: Name of the array

    Returns:
        Memory-mapped array
    """
    os.makedirs(mmap_folder, exist_ok=True)

    coverage_path = os.path.join(mmap_folder, f"{array_name}.dat")
    log.debug(f"Creating array of size {total_positions:,} elements")

    return np.memmap(
        coverage_path, dtype=dtype, mode="w+", shape=(total_positions,), order="C"
    )


@track_memory(name="filter_mmap_arrays", detailed=True)
def filter_arrays_by_subjects(
    arrays: Dict[str, np.ndarray],
    target_subjects: Set[int],
    output_dir: str,
    subject_key: str = "subject_numeric_id",
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Filter memory-mapped arrays based on subject IDs.

    Args:
        arrays: Dictionary of numpy arrays to filter
        target_subjects: Set of subject IDs to keep
        output_dir: Directory for output memory-mapped files
        subject_key: Key for the subject ID array
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process
        progress: Whether to show progress bars

    Returns:
        Dictionary of filtered arrays
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert target_subjects to a numpy array for faster lookups
    target_array = np.array(list(target_subjects))

    # First pass: count matches
    log.debug(f"Counting matches across {len(arrays[subject_key]):,} elements")
    total_matches = _count_matching_elements(
        arrays[subject_key],
        target_array,
        num_threads=num_threads,
        chunk_size=chunk_size,
        progress=progress,
    )

    log.info(
        f"Found {total_matches:,} matching elements ({total_matches/len(arrays[subject_key]):.2%})"
    )

    if total_matches == 0:
        log.warning("No elements match the filtering criteria")
        return {}

    # Create output arrays
    result_arrays = {}
    for key, arr in arrays.items():
        result_arrays[key] = np.memmap(
            os.path.join(output_dir, f"{key}.dat"),
            dtype=arr.dtype,
            mode="w+",
            shape=(total_matches,),
        )

    # Second pass: copy matching elements
    log.debug("Copying matching elements to filtered arrays")
    _copy_matching_elements(
        arrays,
        result_arrays,
        arrays[subject_key],
        target_array,
        num_threads=num_threads,
        chunk_size=chunk_size,
        progress=progress,
    )

    # Flush changes to disk
    for arr in result_arrays.values():
        arr.flush()

    return result_arrays


def _count_matching_elements(
    subject_array: np.ndarray,
    target_array: np.ndarray,
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
    progress: bool = True,
) -> int:
    """
    Count elements matching the target array.

    Args:
        subject_array: Array of subject IDs
        target_array: Array of target subject IDs
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process
        progress: Whether to show progress bars

    Returns:
        Number of matching elements
    """
    total_chunks = (len(subject_array) + chunk_size - 1) // chunk_size
    counts = np.zeros(total_chunks, dtype=np.int64)

    def count_chunk(chunk_idx: int) -> int:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(subject_array))

        # Use numpy's vectorized operations
        chunk = subject_array[start:end]
        mask = np.isin(chunk, target_array)
        return np.sum(mask)

    # Use ThreadPoolExecutor for parallelization
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(count_chunk, i) for i in range(total_chunks)]

        # Use tqdm for progress tracking if requested
        if progress:
            with tqdm(total=total_chunks, desc="Counting matches") as pbar:
                for i, future in enumerate(as_completed(futures)):
                    counts[i] = future.result()
                    pbar.update(1)
        else:
            for i, future in enumerate(as_completed(futures)):
                counts[i] = future.result()

    return np.sum(counts)


def _copy_matching_elements(
    source_arrays: Dict[str, np.ndarray],
    target_arrays: Dict[str, np.ndarray],
    subject_array: np.ndarray,
    target_array: np.ndarray,
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
    progress: bool = True,
) -> None:
    """
    Copy matching elements from source arrays to target arrays.

    Args:
        source_arrays: Dictionary of source arrays
        target_arrays: Dictionary of target arrays
        subject_array: Array of subject IDs
        target_array: Array of target subject IDs
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process
        progress: Whether to show progress bars
    """
    total_chunks = (len(subject_array) + chunk_size - 1) // chunk_size

    # Use a shared counter for the write position
    counter = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()

    def process_chunk(chunk_idx: int) -> None:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(subject_array))

        # Create mask for matching elements
        chunk = subject_array[start:end]
        mask = np.isin(chunk, target_array)
        matches = np.sum(mask)

        if matches > 0:
            # Get write position
            with lock:
                pos = counter.value
                counter.value += matches

            # Copy matching elements for each array
            for key in source_arrays:
                target_arrays[key][pos : pos + matches] = source_arrays[key][start:end][
                    mask
                ]

    # Use ThreadPoolExecutor for parallelization
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, i) for i in range(total_chunks)]

        # Use tqdm for progress tracking if requested
        if progress:
            with tqdm(total=total_chunks, desc="Copying elements") as pbar:
                for future in as_completed(futures):
                    future.result()  # This will raise any exceptions
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                future.result()


def slice_arrays(
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    output_dir: str,
    num_threads: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Slice multiple arrays using the same indices.

    Args:
        arrays: Dictionary of numpy arrays
        indices: Indices to keep
        output_dir: Directory for output memory-mapped files
        num_threads: Number of threads to use

    Returns:
        Dictionary of sliced arrays
    """
    os.makedirs(output_dir, exist_ok=True)

    result_arrays = {}
    for key, arr in arrays.items():
        result_arrays[key] = np.memmap(
            os.path.join(output_dir, f"{key}.dat"),
            dtype=arr.dtype,
            mode="w+",
            shape=(len(indices),),
        )
        result_arrays[key][:] = arr[indices]
        result_arrays[key].flush()

    return result_arrays


def cleanup_mmap_files(directory: str) -> None:
    """
    Clean up memory-mapped files in a directory.

    Args:
        directory: Directory containing memory-mapped files
    """
    if not os.path.exists(directory):
        return

    for filename in os.listdir(directory):
        if filename.endswith(".dat"):
            try:
                os.unlink(os.path.join(directory, filename))
            except (OSError, IOError) as e:
                log.warning(f"Failed to delete {filename}: {e}")


@track_memory(name="get_representative_indices", detailed=True)
def get_representative_indices(
    row_hashes: np.ndarray,
    num_threads: int = 1,
    max_memory: Union[str, int, float] = "4G",
    mmap_folder: Optional[str] = None,
    output_filename: str = "final_indices.dat",
) -> np.ndarray:
    """
    Get representative indices by removing duplicates.

    Args:
        row_hashes: Array of row hashes
        num_threads: Number of threads to use
        max_memory: Maximum memory to use
        mmap_folder: Path to memory-mapped folder
        output_filename: Name of output file

    Returns:
        Array of representative indices
    """
    if mmap_folder is None:
        mmap_folder = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(mmap_folder, exist_ok=True)

    # Create dictionary of unique values with their first occurrence index
    chunk_size = 10_000_000
    seen_values = {}
    representative_indices = []

    # Process in chunks to conserve memory
    for chunk_idx in range(0, len(row_hashes), chunk_size):
        chunk_end = min(chunk_idx + chunk_size, len(row_hashes))

        for i in range(chunk_idx, chunk_end):
            hash_val = row_hashes[i]
            if hash_val not in seen_values:
                seen_values[hash_val] = i
                representative_indices.append(i)

        # Periodically clear memory
        if chunk_idx % (chunk_size * 10) == 0 and chunk_idx > 0:
            gc.collect()

    # Create memory-mapped array for result
    output_path = os.path.join(mmap_folder, output_filename)
    result = np.memmap(
        output_path, dtype=np.int64, mode="w+", shape=(len(representative_indices),)
    )

    # Copy indices to mmap array
    result[:] = representative_indices
    result.flush()

    log.info(f"Found {len(representative_indices):,} unique elements")

    return result


@track_memory(name="memory_efficient_factorize", detailed=True)
def memory_efficient_factorize(
    array: np.ndarray,
    mmap_folder: str,
    max_memory: Union[str, int, float] = "4G",
    inverse: Optional[np.ndarray] = None,
    num_threads: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient implementation of array factorization.

    Args:
        array: Input array
        mmap_folder: Path to memory-mapped folder
        max_memory: Maximum memory to use
        inverse: Optional existing inverse array
        num_threads: Number of threads to use

    Returns:
        Tuple of (inverse_indices, unique_values)
    """
    os.makedirs(mmap_folder, exist_ok=True)

    # Get unique values
    log.debug(f"Finding unique values in array of length {len(array):,}")
    unique_values = np.unique(array)

    log.info(f"Found {len(unique_values):,} unique values")

    # Create or use existing inverse array
    if inverse is None:
        inverse = np.memmap(
            os.path.join(mmap_folder, "inverse.dat"),
            dtype=np.int64,
            mode="w+",
            shape=(len(array),),
        )

    # Process in chunks
    chunk_size = 10_000_000
    for chunk_start in tqdm(
        range(0, len(array), chunk_size),
        desc="Creating inverse indices",
        ncols=80,
    ):
        chunk_end = min(chunk_start + chunk_size, len(array))

        # Find indices for this chunk
        for i in range(chunk_start, chunk_end):
            inverse[i] = np.searchsorted(unique_values, array[i])

        # Flush every 10 chunks
        if (chunk_start // chunk_size) % 10 == 0:
            inverse.flush()

    # Final flush
    inverse.flush()

    return inverse, unique_values
