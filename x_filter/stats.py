import os
import gc
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional
from x_filter.logging_setup import get_logger
from x_filter.utils import is_debug
from x_filter.memory_tracker import track_memory
from x_filter.resource_management import ResourceManager
from x_filter.core_processing import (
    update_coverage_array,
    perform_cumulative_sum,
    compute_alignment_statistics,
    compute_coverage_statistics,
    compute_global_coverage_statistics,
    trim_coverage_by_subject,
    initialize_mmap_array,
    initialize_mmap_arrays,
    get_representative_indices,
    memory_efficient_factorize,
)
from tqdm import tqdm

log = get_logger()


from numba import njit, prange
import numpy as np
from typing import Tuple
import os


@njit(parallel=True, fastmath=True)
def _parallel_max_lengths(
    max_lengths: np.ndarray,
    inverse_indices: np.ndarray,
    subject_lengths: np.ndarray,
    chunk_size: int = 10_000_000,
) -> None:
    """Parallel implementation of maximum length calculation"""
    n_elements = len(inverse_indices)
    n_chunks = (n_elements + chunk_size - 1) // chunk_size

    # Process in chunks to avoid memory issues
    for chunk_id in prange(n_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, n_elements)

        # Process each element in the chunk
        for i in range(start_idx, end_idx):
            idx = inverse_indices[i]
            val = subject_lengths[i]
            # Atomic maximum operation
            current = max_lengths[idx]
            while val > current:
                if max_lengths[idx] == current:
                    max_lengths[idx] = val
                    break
                current = max_lengths[idx]


@track_memory(name="calculate_max_subject_lengths", detailed=True)
def calculate_max_subject_lengths(
    unique_subjects: np.ndarray,
    inverse_indices: np.ndarray,
    subject_lengths: np.ndarray,
    mmap_folder: str,
) -> Tuple[np.memmap, np.int64]:
    """Calculate maximum subject lengths using memory mapping and parallel processing."""
    log.debug("Calculating maximum subject lengths")
    n_unique = len(unique_subjects)

    max_lengths_path = os.path.join(mmap_folder, "max_subject_lengths.dat")
    max_subject_lengths = np.memmap(
        max_lengths_path, dtype=subject_lengths.dtype, mode="w+", shape=(n_unique,)
    )

    try:
        # Initialize with zeros
        max_subject_lengths[:] = 0
        max_subject_lengths.flush()

        # Calculate chunk size based on array size
        chunk_size = min(10_000_000, len(inverse_indices) // (os.cpu_count() * 2))

        # Run parallel computation
        _parallel_max_lengths(
            max_subject_lengths, inverse_indices, subject_lengths, chunk_size
        )
        max_subject_lengths.flush()

        total_positions = np.sum(max_subject_lengths)
        log.debug(f"Total positions: {total_positions:,}")

        return max_subject_lengths, total_positions

    except Exception as e:
        log.error(f"Error calculating max lengths: {e}")
        if os.path.exists(max_lengths_path):
            os.remove(max_lengths_path)
        raise


def cleanup_memmaps(active_memmaps: Dict[str, np.memmap]) -> None:
    """Clean up active memory-mapped arrays."""
    for name, mmap_array in active_memmaps.items():
        try:
            del mmap_array
        except Exception as e:
            log.warning(f"Error cleaning up memmap {name}: {e}")

    active_memmaps.clear()
    gc.collect()


from concurrent.futures import ThreadPoolExecutor


def chunk_bincount(arr, n, chunk_size=10_000_000, num_threads=1):
    """Process bincount in chunks using multiple threads"""
    result = np.zeros(n, dtype=np.intp)

    def process_range(start_idx):
        end_idx = min(start_idx + chunk_size, len(arr))
        chunk = arr[start_idx:end_idx]
        return np.bincount(chunk, minlength=n)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        # Only submit chunk_size number of tasks at a time
        for start_idx in range(0, len(arr), chunk_size):
            futures.append(executor.submit(process_range, start_idx))

            # Process completed futures to free memory
            if len(futures) >= num_threads * 2:
                for future in futures:
                    result += future.result()
                futures = []

        # Process any remaining futures
        for future in futures:
            result += future.result()

    return result


@track_memory(name="calculate_statistics", detailed=True)
def calculate_statistics(
    numpy_arrays: Dict[str, np.ndarray],
    temp_files: Dict[str, str],
    num_threads: int = 1,
    max_memory: Union[str, float, int] = "4G",
    rm_dups: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculate comprehensive coverage statistics for input arrays.

    Args:
        numpy_arrays: Dictionary of input arrays
        temp_files: Dictionary of temporary file paths
        num_threads: Number of threads for parallel processing
        max_memory: Maximum memory limit
        rm_dups: Whether to remove duplicates

    Returns:
        Tuple containing:
        - statistics DataFrame
        - unique subjects array
        - inverse indices array
        - processed numpy arrays
    """
    mmap_folder = temp_files["mmap"]
    os.makedirs(mmap_folder, exist_ok=True)
    active_memmaps = {}

    try:
        # Setup progress bar - main progress bar at position 0
        total_steps = 7  # Total number of major processing steps
        pbar = tqdm(
            total=total_steps, desc="Computing statistics", leave=False, ncols=80
        )

        # Initialize resource manager
        resource_mgr = ResourceManager(max_memory=max_memory, max_threads=num_threads)

        # Extract arrays
        subject_ids = numpy_arrays["subject_numeric_id"]
        subject_start_positions = numpy_arrays["subjectStart"]
        subject_end_positions = numpy_arrays["subjectEnd"]
        alignment_lengths = numpy_arrays["alnLength"]
        query_lengths = numpy_arrays["qlen"]
        percent_identity = numpy_arrays["percIdentity"]
        subject_lengths = numpy_arrays["slen"]

        # Track active memmaps
        active_memmaps.update(
            {
                "subject_ids": subject_ids,
                "subject_start_positions": subject_start_positions,
                "subject_end_positions": subject_end_positions,
                "alignment_lengths": alignment_lengths,
                "query_lengths": query_lengths,
                "percent_identity": percent_identity,
                "subject_lengths": subject_lengths,
            }
        )

        # Handle deduplication with nested progress bars
        if rm_dups:
            pbar.set_description("Deduplicating arrays")
            log.debug("Deduplicating arrays")
            row_hashes = numpy_arrays["row_hash"]
            active_memmaps["row_hashes"] = row_hashes

            # Get representative indices - creates nested progress bars
            representative_indices = get_representative_indices(
                row_hashes,
                num_threads=num_threads,
                max_memory=max_memory,
                mmap_folder=mmap_folder,
            )

            # Print a newline for proper spacing after nested progress bars
            print("")

            # Slice arrays using representative indices
            numpy_arrays = resource_mgr.slice_arrays(
                numpy_arrays, representative_indices, mmap_folder
            )

            # Update array references after deduplication
            subject_ids = numpy_arrays["subject_numeric_id"]
            subject_start_positions = numpy_arrays["subjectStart"]
            subject_end_positions = numpy_arrays["subjectEnd"]
            alignment_lengths = numpy_arrays["alnLength"]
            query_lengths = numpy_arrays["qlen"]
            percent_identity = numpy_arrays["percIdentity"]
            subject_lengths = numpy_arrays["slen"]
            pbar.update(1)

        # Create inverse mapping
        log.debug("Creating inverse subject mapping")
        inverse_indices = initialize_mmap_array(
            total_positions=len(subject_ids),
            dtype=np.int64,
            mmap_folder=mmap_folder,
            array_name="inverse_indices",
        )

        # This function will create nested progress bars
        inverse_indices, unique_subjects = memory_efficient_factorize(
            subject_ids,
            mmap_folder=mmap_folder,
            max_memory=max_memory,
            inverse=inverse_indices,
            num_threads=num_threads,
        )
        active_memmaps.update(
            {"inverse_indices": inverse_indices, "unique_subjects": unique_subjects}
        )

        # Get chunk strategy for coverage processing
        # Estimate concurrent chunks needed for coverage update: indices, starts, ends, coverage ~ 4
        num_concurrent_coverage = 4
        arr_info = resource_mgr.analyze_array(subject_ids)
        strategy = resource_mgr.calculate_chunk_size(arr_info, num_concurrent_chunks=num_concurrent_coverage)

        # Calculate maximum subject lengths
        pbar.set_description("Calculating maximum subject lengths")
        log.debug("Calculating maximum subject lengths")
        max_subject_lengths, total_positions = calculate_max_subject_lengths(
            unique_subjects, inverse_indices, subject_lengths, mmap_folder
        )
        active_memmaps["max_subject_lengths"] = max_subject_lengths
        pbar.update(1)

        # Initialize coverage arrays
        pbar.set_description("Initializing coverage arrays")
        log.debug("Initializing coverage arrays")

        array_size = total_positions * np.dtype(np.int32).itemsize
        if array_size < resource_mgr.max_memory:
            log.debug(f"Using memory for flattened_coverage: {array_size:,} bytes")
            flattened_coverage = np.zeros(total_positions, dtype=np.int32)
        else:
            flattened_coverage = np.memmap(
                os.path.join(mmap_folder, "flattened_coverage.dat"),
                dtype=np.int32,
                mode="w+",
                shape=(total_positions,),
            )
            active_memmaps["flattened_coverage"] = flattened_coverage

        start_positions, subject_lengths_mmap = initialize_mmap_arrays(
            mmap_folder, unique_subjects, max_subject_lengths
        )
        active_memmaps.update(
            {
                "start_positions": start_positions,
                "subject_lengths_mmap": subject_lengths_mmap,
            }
        )
        pbar.update(1)

        # Update coverage
        log.debug(f"Computing coverage with {num_threads} threads")
        pbar.set_description(f"Computing coverage with {num_threads} threads")
        update_coverage_array(
            flattened_coverage,
            inverse_indices,
            subject_start_positions,
            subject_end_positions,
            start_positions,
            subject_lengths_mmap,
            n_partitions=num_threads,
            num_threads=num_threads,
        )
        pbar.update(1)

        # Perform cumulative sum
        pbar.set_description("Computing cumulative sums")
        log.debug("Computing cumulative sums")
        perform_cumulative_sum(
            flattened_coverage, start_positions, subject_lengths_mmap
        )
        pbar.update(1)

        # Calculate alignment statistics
        pbar.set_description("Computing alignment statistics")
        log.debug("Computing alignment statistics")
        n_subjects = len(unique_subjects)
        alignment_stats = compute_alignment_statistics(
            alignment_lengths,
            query_lengths,
            percent_identity,
            inverse_indices,
            n_subjects,
            num_threads=strategy.total_threads,
        )

        # Count alignments per subject
        log.debug("Counting alignments per subject")
        num_alignments = chunk_bincount(
            inverse_indices, n_subjects, chunk_size=10_000_000, num_threads=num_threads
        )

        # Trim coverage
        pbar.set_description("Trimming coverage")
        log.debug("Trimming coverage")
        trim_coverage_by_subject(
            flattened_coverage,
            start_positions,
            subject_lengths_mmap,
            alignment_stats["avg_aln_len"],
        )
        pbar.update(1)

        # Calculate coverage statistics
        pbar.set_description("Computing coverage statistics")
        log.debug("Computing coverage statistics")
        total_coverage = initialize_mmap_array(
            total_positions=len(start_positions),
            dtype=np.int64,
            mmap_folder=mmap_folder,
            array_name="total_coverage",
        )
        nonzero_coverage_counts = initialize_mmap_array(
            total_positions=len(start_positions),
            dtype=np.int32,
            mmap_folder=mmap_folder,
            array_name="nonzero_coverage_counts",
        )

        compute_global_coverage_statistics(
            flattened_coverage,
            start_positions,
            subject_lengths_mmap,
            total_coverage,
            nonzero_coverage_counts,
        )
        active_memmaps.update(
            {
                "total_coverage": total_coverage,
                "nonzero_coverage_counts": nonzero_coverage_counts,
            }
        )
        pbar.update(1)

        # Calculate mean coverage and depth statistics
        pbar.set_description("Computing depth statistics")
        log.debug("Computing depth statistics")
        mean_coverage, std_coverage = compute_coverage_statistics(
            flattened_coverage, start_positions, subject_lengths_mmap
        )
        if isinstance(flattened_coverage, np.memmap):
            del flattened_coverage
            os.remove(os.path.join(mmap_folder, "flattened_coverage.dat"))
            if "flattened_coverage" in active_memmaps:
                del active_memmaps["flattened_coverage"]
        else:
            del flattened_coverage
        gc.collect()
        # Calculate depth evenness
        depth_evenness = np.zeros_like(mean_coverage, dtype=np.float32)
        valid_coverage_mask = (mean_coverage > 0) & (~np.isnan(mean_coverage))
        depth_evenness[valid_coverage_mask] = (
            std_coverage[valid_coverage_mask] / mean_coverage[valid_coverage_mask]
        )
        depth_evenness[~valid_coverage_mask] = np.nan

        # Create final statistics DataFrame
        pbar.set_description("Creating final statistics DataFrame")
        log.debug("Creating final statistics DataFrame")
        final_stats = pd.DataFrame(
            {
                "subject_numeric_id": unique_subjects,
                "avg_alignment_length": alignment_stats["avg_aln_len"],
                "num_alignments": num_alignments,
                "avg_read_length": alignment_stats["avg_read_len"],
                "std_read_length": alignment_stats["std_read_len"],
                "avg_identity": alignment_stats["avg_identity"],
                "std_identity": alignment_stats["std_identity"],
                "total_covered_bases": nonzero_coverage_counts,
                "total_depth": total_coverage,
                "subject_length": max_subject_lengths,
                "breadth": nonzero_coverage_counts / max_subject_lengths,
                "depth_mean": mean_coverage,
                "depth_std": std_coverage,
                "depth_evenness": depth_evenness,
            }
        )

        # Ensure the final update happens AFTER the DataFrame creation is complete
        pbar.update(1)

        # Add a small delay to ensure the progress bar updates visually
        import time

        time.sleep(0.1)

        # Force the progress bar to show 100%
        pbar.refresh()

        # Close the progress bar after the final update
        pbar.close()

        # Add a debug message to confirm completion
        log.debug("Statistics calculation completed successfully")

        return final_stats, unique_subjects, inverse_indices, numpy_arrays

    except Exception as e:
        # Close the progress bar in case of error
        if "pbar" in locals():
            pbar.close()
        log.error(f"Error in calculate_statistics: {e}")
        raise

    finally:
        # Make sure the progress bar is closed if it exists
        if "pbar" in locals():
            try:
                pbar.close()
            except:
                pass
        # Clean up memory-mapped arrays
        cleanup_memmaps(active_memmaps)
        gc.collect()
