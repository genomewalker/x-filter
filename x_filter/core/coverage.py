# x_filter/core/coverage.py

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import njit, prange

from x_filter.utils.logging import get_logger, LogContext
from x_filter.utils.memory import track_memory

log = get_logger(__name__)


@track_memory(name="calculate_coverage_statistics", detailed=True)
def calculate_coverage_statistics(
    numpy_arrays: Dict[str, np.ndarray],
    tmp_files: Dict[str, str],
    num_threads: int = 1,
    max_memory: Union[str, float, int] = "4G",
    rm_dups: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculate comprehensive coverage statistics for input arrays.

    Args:
        numpy_arrays: Dictionary of input arrays
        tmp_files: Dictionary of temporary file paths
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
    with LogContext(log, "Calculating coverage statistics"):
        mmap_folder = tmp_files["mmap"]
        os.makedirs(mmap_folder, exist_ok=True)
        active_memmaps = {}

        try:
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

            # Handle deduplication
            if rm_dups and "row_hash" in numpy_arrays:
                log.info("Deduplicating arrays")
                row_hashes = numpy_arrays["row_hash"]
                active_memmaps["row_hashes"] = row_hashes

                # Get representative indices
                from x_filter.db.mmap import get_representative_indices

                representative_indices = get_representative_indices(
                    row_hashes,
                    num_threads=num_threads,
                    max_memory=max_memory,
                    mmap_folder=mmap_folder,
                )

                # Slice arrays using representative indices
                from x_filter.db.mmap import slice_arrays

                numpy_arrays = slice_arrays(
                    numpy_arrays,
                    representative_indices,
                    mmap_folder,
                    num_threads=num_threads,
                )

                # Update array references after deduplication
                subject_ids = numpy_arrays["subject_numeric_id"]
                subject_start_positions = numpy_arrays["subjectStart"]
                subject_end_positions = numpy_arrays["subjectEnd"]
                alignment_lengths = numpy_arrays["alnLength"]
                query_lengths = numpy_arrays["qlen"]
                percent_identity = numpy_arrays["percIdentity"]
                subject_lengths = numpy_arrays["slen"]

            # Create inverse mapping
            log.info("Creating inverse subject mapping")

            from x_filter.db.mmap import (
                initialize_mmap_array,
                memory_efficient_factorize,
            )

            inverse_indices = initialize_mmap_array(
                total_positions=len(subject_ids),
                dtype=np.int64,
                mmap_folder=mmap_folder,
                array_name="inverse_indices",
            )

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

            # Calculate maximum subject lengths
            log.info("Calculating maximum subject lengths")

            max_subject_lengths, total_positions = calculate_max_subject_lengths(
                unique_subjects,
                inverse_indices,
                subject_lengths,
                mmap_folder,
                num_threads=num_threads,
            )

            active_memmaps["max_subject_lengths"] = max_subject_lengths

            # Initialize coverage arrays
            log.info("Initializing coverage arrays")

            flattened_coverage = initialize_mmap_array(
                total_positions=total_positions,
                dtype=np.int32,
                mmap_folder=mmap_folder,
                array_name="flattened_coverage",
            )

            active_memmaps["flattened_coverage"] = flattened_coverage

            start_positions, subject_lengths_mmap = initialize_subject_arrays(
                mmap_folder, unique_subjects, max_subject_lengths
            )

            active_memmaps.update(
                {
                    "start_positions": start_positions,
                    "subject_lengths_mmap": subject_lengths_mmap,
                }
            )

            # Update coverage
            log.info(f"Computing coverage with {num_threads} threads")

            update_coverage_array(
                flattened_coverage,
                inverse_indices,
                subject_start_positions,
                subject_end_positions,
                start_positions,
                subject_lengths_mmap,
                num_threads=num_threads,
            )

            # Perform cumulative sum
            log.info("Computing cumulative sums")

            perform_cumulative_sum(
                flattened_coverage,
                start_positions,
                subject_lengths_mmap,
                num_threads=num_threads,
            )

            # Calculate alignment statistics
            log.info("Computing alignment statistics")

            n_subjects = len(unique_subjects)
            alignment_stats = compute_alignment_statistics(
                alignment_lengths,
                query_lengths,
                percent_identity,
                inverse_indices,
                n_subjects,
                num_threads=num_threads,
            )

            # Count alignments per subject
            log.info("Counting alignments per subject")

            num_alignments = count_alignments_per_subject(
                inverse_indices, n_subjects, num_threads=num_threads
            )

            # Trim coverage
            log.info("Trimming coverage")

            trim_coverage_by_subject(
                flattened_coverage,
                start_positions,
                subject_lengths_mmap,
                alignment_stats["avg_aln_len"],
            )

            # Calculate coverage statistics
            log.info("Computing coverage statistics")

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

            # Calculate mean coverage and depth statistics
            log.info("Computing depth statistics")

            mean_coverage, std_coverage = compute_coverage_statistics(
                flattened_coverage,
                start_positions,
                subject_lengths_mmap,
                num_threads=num_threads,
            )

            # Clean up flattened coverage now that we're done with it
            if "flattened_coverage" in active_memmaps:
                del active_memmaps["flattened_coverage"]
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
            log.info("Creating final statistics DataFrame")

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

            # Calculate breadth expected ratio
            final_stats["breadth_expected"] = 1.0 - np.exp(-final_stats["depth_mean"])
            final_stats["breadth_expected_ratio"] = (
                final_stats["breadth"] / final_stats["breadth_expected"]
            )

            return final_stats, unique_subjects, inverse_indices, numpy_arrays

        except Exception as e:
            log.error(f"Error in calculate_coverage_statistics: {e}")
            raise

        finally:
            # Clean up memory-mapped arrays
            cleanup_memmaps(active_memmaps)
            gc.collect()


@njit(parallel=True, fastmath=True)
def calculate_max_subject_lengths(
    unique_subjects: np.ndarray,
    inverse_indices: np.ndarray,
    subject_lengths: np.ndarray,
    mmap_folder: str,
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
) -> Tuple[np.ndarray, np.int64]:
    """
    Calculate maximum subject lengths using parallel processing.

    Args:
        unique_subjects: Array of unique subjects
        inverse_indices: Inverse mapping indices
        subject_lengths: Array of subject lengths
        mmap_folder: Path to memory-mapped folder
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process

    Returns:
        Tuple of (max_subject_lengths, total_positions)
    """
    numba.set_num_threads(num_threads)
    n_unique = len(unique_subjects)

    # Create output array
    max_lengths = np.zeros(n_unique, dtype=subject_lengths.dtype)

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

    # Calculate total positions for coverage array
    total_positions = np.sum(max_lengths)

    return max_lengths, total_positions


def initialize_subject_arrays(
    mmap_folder: str, unique_subjects: np.ndarray, max_subject_lengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize arrays for subject positions and lengths.

    Args:
        mmap_folder: Path to memory-mapped folder
        unique_subjects: Array of unique subjects
        max_subject_lengths: Array of maximum subject lengths

    Returns:
        Tuple of (start_positions, subject_lengths)
    """
    # Calculate cumulative sum for start positions
    cumsum = np.concatenate(([0], np.cumsum(max_subject_lengths[:-1])))

    # Create memory-mapped arrays
    start_positions_file = os.path.join(mmap_folder, "start_positions.dat")
    subject_lengths_file = os.path.join(mmap_folder, "subject_lengths.dat")

    start_positions = np.memmap(
        start_positions_file, dtype=np.int64, mode="w+", shape=(len(unique_subjects),)
    )

    subject_lengths_mmap = np.memmap(
        subject_lengths_file, dtype=np.int32, mode="w+", shape=(len(unique_subjects),)
    )

    # Write data
    start_positions[:] = cumsum
    subject_lengths_mmap[:] = max_subject_lengths

    # Flush to disk
    start_positions.flush()
    subject_lengths_mmap.flush()

    return start_positions, subject_lengths_mmap


@njit(parallel=True, fastmath=True)
def update_coverage_array(
    flattened_coverage: np.ndarray,
    inverse_subject_indices: np.ndarray,
    subject_start_positions: np.ndarray,
    subject_end_positions: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
    num_threads: int = 1,
    n_partitions: int = 1024,
) -> None:
    """
    Update coverage array in parallel based on alignments.

    Args:
        flattened_coverage: Array to store coverage values
        inverse_subject_indices: Array mapping to subject indices
        subject_start_positions: Start positions within subjects
        subject_end_positions: End positions within subjects
        start_positions: Global start positions
        subject_lengths: Subject lengths
        num_threads: Number of threads to use
        n_partitions: Number of partitions for parallel processing
    """
    numba.set_num_threads(num_threads)
    n_intervals = len(inverse_subject_indices)
    total_length = len(flattened_coverage)
    partition_size = (total_length + n_partitions - 1) // n_partitions

    # Process partitions in parallel
    for p_id in prange(n_partitions):
        p_start = p_id * partition_size
        p_end = min(p_start + partition_size, total_length)

        # Use a fixed small buffer for events per partition
        buffer_size = min(65536, partition_size)

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


@njit(parallel=True, fastmath=True)
def perform_cumulative_sum(
    flattened_coverage: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
    num_threads: int = 1,
) -> None:
    """
    Perform cumulative sum on coverage array in parallel by subjects.

    Args:
        flattened_coverage: Array containing coverage values
        start_positions: Start positions for each subject
        subject_lengths: Length of each subject
        num_threads: Number of threads to use
    """
    numba.set_num_threads(num_threads)

    for subject_id in prange(len(start_positions)):
        start_idx = start_positions[subject_id]
        end_idx = start_positions[subject_id] + subject_lengths[subject_id]

        cumsum = 0
        for i in range(start_idx, end_idx):
            cumsum += flattened_coverage[i]
            flattened_coverage[i] = cumsum


@njit(parallel=True, fastmath=True)
def compute_alignment_statistics(
    alignment_lengths: np.ndarray,
    query_lengths: np.ndarray,
    percent_identity: np.ndarray,
    inverse_indices: np.ndarray,
    n_subjects: int,
    num_threads: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute alignment statistics using parallel processing.

    Args:
        alignment_lengths: Array of alignment lengths
        query_lengths: Array of query lengths
        percent_identity: Array of percent identity values
        inverse_indices: Inverse mapping indices
        n_subjects: Number of unique subjects
        num_threads: Number of threads to use

    Returns:
        Dictionary of statistical arrays
    """
    numba.set_num_threads(num_threads)
    chunk_size = (len(inverse_indices) + num_threads - 1) // num_threads

    # Allocate results array (thread_id, subject_id, statistic)
    # Statistics: sum_aln_len, sum_aln_len_sq, sum_read_len, sum_read_len_sq,
    #            sum_identity, sum_identity_sq, count
    results = np.zeros((num_threads, n_subjects, 7), dtype=np.float64)

    # Process in parallel
    for thread_id in prange(num_threads):
        start = thread_id * chunk_size
        end = min(start + chunk_size, len(inverse_indices))

        for i in range(start, end):
            subject_idx = inverse_indices[i]
            aln_len = alignment_lengths[i]
            read_len = query_lengths[i]
            identity = percent_identity[i]

            # Accumulate statistics
            results[thread_id, subject_idx, 0] += aln_len
            results[thread_id, subject_idx, 1] += aln_len * aln_len
            results[thread_id, subject_idx, 2] += read_len
            results[thread_id, subject_idx, 3] += read_len * read_len
            results[thread_id, subject_idx, 4] += identity
            results[thread_id, subject_idx, 5] += identity * identity
            results[thread_id, subject_idx, 6] += 1

    # Combine results from all threads
    final_results = results.sum(axis=0)

    # Calculate statistics
    stats = {}
    mask = final_results[:, 6] > 0

    # Process each statistic
    for idx, name in [(0, "aln_len"), (2, "read_len"), (4, "identity")]:
        mean = np.zeros(n_subjects, dtype=np.float64)
        std = np.zeros(n_subjects, dtype=np.float64)

        # Calculate mean
        mean[mask] = final_results[mask, idx] / final_results[mask, 6]

        # Calculate standard deviation
        variance = np.maximum(
            final_results[mask, idx + 1] / final_results[mask, 6] - mean[mask] ** 2, 0
        )
        std[mask] = np.sqrt(variance)

        stats[f"avg_{name}"] = mean
        stats[f"std_{name}"] = std

    return stats


def count_alignments_per_subject(
    inverse_indices: np.ndarray,
    n_subjects: int,
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
) -> np.ndarray:
    """
    Count alignments per subject using chunked processing.

    Args:
        inverse_indices: Inverse mapping indices
        n_subjects: Number of unique subjects
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process

    Returns:
        Array of alignment counts per subject
    """
    result = np.zeros(n_subjects, dtype=np.int64)

    def process_range(start_idx: int, end_idx: int) -> np.ndarray:
        """Process a range of indices."""
        chunk = inverse_indices[start_idx:end_idx]
        return np.bincount(chunk, minlength=n_subjects)

    # Process in parallel using thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        # Submit chunk jobs
        for start_idx in range(0, len(inverse_indices), chunk_size):
            end_idx = min(start_idx + chunk_size, len(inverse_indices))
            futures.append(executor.submit(process_range, start_idx, end_idx))

        # Collect results
        for future in as_completed(futures):
            result += future.result()

    return result


@njit(parallel=True, fastmath=True)
def compute_global_coverage_statistics(
    flattened_coverage: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
    total_coverage: np.ndarray,
    nonzero_coverage_counts: np.ndarray,
    num_threads: int = 1,
) -> None:
    """
    Compute global coverage statistics using parallel processing.

    Args:
        flattened_coverage: Array containing coverage values
        start_positions: Start positions for each subject
        subject_lengths: Length of each subject
        total_coverage: Output array for total coverage values
        nonzero_coverage_counts: Output array for nonzero coverage counts
        num_threads: Number of threads to use
    """
    numba.set_num_threads(num_threads)

    for i in prange(len(start_positions)):
        start_idx = start_positions[i]
        end_idx = start_idx + subject_lengths[i]

        # Get segment for this subject
        segment = flattened_coverage[start_idx:end_idx]

        # Calculate statistics
        total_coverage[i] = np.sum(segment)
        nonzero_coverage_counts[i] = np.count_nonzero(segment)


@njit(parallel=True, fastmath=True)
def compute_coverage_statistics(
    flattened_coverage: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
    num_threads: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coverage statistics using parallel processing.

    Args:
        flattened_coverage: Array containing coverage values
        start_positions: Start positions for each subject
        subject_lengths: Length of each subject
        num_threads: Number of threads to use

    Returns:
        Tuple of (mean_coverage, std_coverage)
    """
    numba.set_num_threads(num_threads)
    n_subjects = len(start_positions)

    mean_coverage = np.empty(n_subjects, dtype=np.float32)
    std_coverage = np.empty(n_subjects, dtype=np.float32)

    for i in prange(n_subjects):
        start = start_positions[i]
        length = subject_lengths[i]
        end = start + length

        if length == 0:
            mean_coverage[i] = 0.0
            std_coverage[i] = 0.0
            continue

        # Calculate mean and variance
        sum_values = 0.0
        sum_squares = 0.0

        for j in range(start, end):
            value = flattened_coverage[j]
            sum_values += value
            sum_squares += value * value

        mean = sum_values / length
        variance = (sum_squares / length) - (mean * mean)

        mean_coverage[i] = mean
        std_coverage[i] = np.sqrt(max(0.0, variance))

    return mean_coverage, std_coverage


@njit(fastmath=True)
def trim_coverage_by_subject(
    flattened_coverage: np.ndarray,
    start_positions: np.ndarray,
    subject_lengths: np.ndarray,
    avg_alignment_lengths: np.ndarray,
    trim_multiplier: float = 2.0,
    trim_offset: int = 10,
) -> None:
    """
    Trim coverage at subject boundaries.

    Args:
        flattened_coverage: Array containing coverage values
        start_positions: Start positions for each subject
        subject_lengths: Length of each subject
        avg_alignment_lengths: Average alignment length per subject
        trim_multiplier: Multiplier for trim length
        trim_offset: Offset for trim length
    """
    for i in range(len(start_positions)):
        # Calculate trim length
        trim_length = (
            int(np.ceil(avg_alignment_lengths[i] / 2 * trim_multiplier)) + trim_offset
        )

        if trim_length >= subject_lengths[i]:
            continue

        start_idx = start_positions[i]
        end_idx = start_idx + subject_lengths[i]

        # Set coverage to 0 at boundaries
        flattened_coverage[start_idx : start_idx + trim_length] = 0
        flattened_coverage[end_idx - trim_length : end_idx] = 0


def cleanup_memmaps(active_memmaps: Dict[str, np.ndarray]) -> None:
    """
    Clean up active memory-mapped arrays.

    Args:
        active_memmaps: Dictionary of active memory-mapped arrays
    """
    for name, mmap_array in active_memmaps.items():
        try:
            del mmap_array
        except Exception as e:
            log.warning(f"Error cleaning up memmap {name}: {e}")

    active_memmaps.clear()
    gc.collect()


def analyze_alignments(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract arrays from alignments DataFrame for coverage analysis.

    Args:
        df: DataFrame with alignment data

    Returns:
        Dictionary of numpy arrays
    """
    query_counts = df["query_numeric_id"].value_counts()
    single_alignment_queries = query_counts[query_counts == 1].index
    multi_alignment_queries = query_counts[query_counts > 1].index

    log.info(f"Number of unique reads: {len(query_counts):,}")
    log.info(f"Number of single alignment: {len(single_alignment_queries):,}")
    log.info(f"Number of multi-alignment: {len(multi_alignment_queries):,}")

    return {
        "subject_numeric_id": df["subject_numeric_id"].to_numpy(),
        "subjectStart": df["subjectStart"].to_numpy(),
        "subjectEnd": df["subjectEnd"].to_numpy(),
        "alnLength": df["alnLength"].to_numpy(),
        "qlen": df["qlen"].to_numpy(),
        "percIdentity": df["percIdentity"].to_numpy(),
        "slen": df["slen"].to_numpy(),
    }
