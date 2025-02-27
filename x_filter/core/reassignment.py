# x_filter/core/reassignment.py

import os
import gc
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, Any
from numba import njit, prange
import threading
from concurrent.futures import ThreadPoolExecutor

from x_filter.utils.logging import get_logger, LogContext
from x_filter.utils.memory import track_memory
from x_filter.db.mmap import initialize_mmap_array

log = get_logger(__name__)


@track_memory(name="reassign_reads", detailed=True)
def reassign_reads(
    numpy_arrays: Dict[str, np.ndarray],
    tmp_files: Dict[str, str],
    iters: int = 25,
    step_min: float = -1.0,
    step_max: float = -1.0,
    mstep: int = 4,
    scale: float = 0.9,
    max_memory: Union[str, float, int] = "4G",
    num_threads: int = 1,
) -> pd.DataFrame:
    """
    Reassign multi-mapped reads using an iterative algorithm.

    Args:
        numpy_arrays: Dictionary of input arrays
        tmp_files: Dictionary of temporary file paths
        iters: Number of iterations
        step_min: Minimum step size
        step_max: Maximum step size
        mstep: Maximum step attempts
        scale: Scale factor for filtering
        max_memory: Maximum memory to use
        num_threads: Number of threads to use

    Returns:
        DataFrame with reassigned alignments
    """
    with LogContext(log, "Reassigning multi-mapped reads"):
        mmap_folder = tmp_files["mmap"]

        # First create inverse indices for subjects and queries
        log.info("Creating inverse subject mapping")
        subject_inverse_indices = initialize_mmap_array(
            total_positions=len(numpy_arrays["subject_numeric_id"]),
            dtype=np.int64,
            mmap_folder=mmap_folder,
            array_name="subject_inverse_indices",
        )

        from x_filter.db.mmap import memory_efficient_factorize

        subject_inverse_indices, unique_subjects = memory_efficient_factorize(
            numpy_arrays["subject_numeric_id"],
            mmap_folder=mmap_folder,
            max_memory=max_memory,
            inverse=subject_inverse_indices,
            num_threads=num_threads,
        )

        log.info("Creating inverse query mapping")
        query_inverse_indices = initialize_mmap_array(
            total_positions=len(numpy_arrays["query_numeric_id"]),
            dtype=np.int64,
            mmap_folder=mmap_folder,
            array_name="reass_query_inverse_indices",
        )

        query_inverse_indices, unique_queries = memory_efficient_factorize(
            numpy_arrays["query_numeric_id"],
            mmap_folder=mmap_folder,
            max_memory=max_memory,
            inverse=query_inverse_indices,
            num_threads=num_threads,
        )

        log.info(f"Number of references: {len(unique_subjects):,}")
        log.info(f"Number of reads: {len(unique_queries):,}")

        # Initialize arrays for tracking
        iter_array = initialize_mmap_array(
            total_positions=numpy_arrays["subject_numeric_id"].shape[0],
            dtype=np.int64,
            mmap_folder=mmap_folder,
            array_name="iter_array",
        )

        prob = initialize_mmap_array(
            total_positions=numpy_arrays["subject_numeric_id"].shape[0],
            dtype=np.float64,
            mmap_folder=mmap_folder,
            array_name="prob",
        )

        # Initialize weights
        log.info("Initializing weights")
        prob[:] = chunked_initialize_weights(
            subject_inverse_indices,
            numpy_arrays["bitScore"],
            len(unique_subjects),
            mmap_folder,
            num_threads=num_threads,
        )

        # Resolve multimappings
        log.info("Resolving multi-mapped reads")
        final_mask = resolve_multimaps_return_indices(
            subject_inverse_indices=subject_inverse_indices,
            query_inverse_indices=query_inverse_indices,
            prob=prob,
            slen=numpy_arrays["slen"],
            iter_array=iter_array,
            mmap_folder=mmap_folder,
            iters=iters,
            step_min=step_min,
            step_max=step_max,
            mstep=mstep,
            scale=scale,
            num_threads=num_threads,
        )

        # Create DataFrame with filtered results
        log.info("Creating filtered DataFrame")
        return pd.DataFrame(
            {
                "query_numeric_id": numpy_arrays["query_numeric_id"][final_mask],
                "subject_numeric_id": numpy_arrays["subject_numeric_id"][final_mask],
                "bitScore": numpy_arrays["bitScore"][final_mask],
                "alnLength": numpy_arrays["alnLength"][final_mask],
                "subjectStart": numpy_arrays["subjectStart"][final_mask],
                "subjectEnd": numpy_arrays["subjectEnd"][final_mask],
                "percIdentity": numpy_arrays["percIdentity"][final_mask],
                "row_hash": numpy_arrays["row_hash"][final_mask],
            }
        )


@njit(parallel=True, fastmath=True)
def parallel_accumulate_weights(
    chunk_indices: np.ndarray,
    chunk_scores: np.ndarray,
    output: np.ndarray,
    num_threads: int = 1,
) -> None:
    """
    Accumulate weights in parallel, equivalent to np.add.at.

    Args:
        chunk_indices: Array of indices
        chunk_scores: Array of scores
        output: Output array
        num_threads: Number of threads to use
    """
    numba.set_num_threads(num_threads)
    n_threads = numba.get_num_threads()

    # Create thread-local accumulators to avoid race conditions
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
    subject_inverse_indices: np.ndarray,
    bitScore: np.ndarray,
    max_index: int,
    mmap_folder: str,
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
) -> np.ndarray:
    """
    Initialize weights using chunked processing with parallel accumulation.

    Args:
        subject_inverse_indices: Inverse mapping indices for subjects
        bitScore: Array of bit scores
        max_index: Maximum index
        mmap_folder: Path to memory-mapped folder
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process

    Returns:
        Array of weights
    """
    # Create memory-mapped array for total weights
    total_weights_file = os.path.join(mmap_folder, "total_weights.dat")
    total_weights = np.memmap(
        total_weights_file, dtype=np.float64, mode="w+", shape=(max_index,)
    )
    total_weights.fill(0)

    try:
        # Process in chunks
        from tqdm import tqdm

        with tqdm(
            total=len(subject_inverse_indices), desc="Calculating weights", ncols=80
        ) as pbar:
            for start in range(0, len(subject_inverse_indices), chunk_size):
                end = min(start + chunk_size, len(subject_inverse_indices))

                # Get chunk data
                chunk_indices = subject_inverse_indices[start:end]
                chunk_scores = bitScore[start:end]

                # Accumulate weights in parallel
                parallel_accumulate_weights(
                    chunk_indices, chunk_scores, total_weights, num_threads=num_threads
                )

                pbar.update(end - start)

        # Handle zero weights (avoid division by zero)
        total_weights[total_weights == 0] = np.finfo(np.float64).tiny

        # Create result array
        result_file = os.path.join(mmap_folder, "weights_result.dat")
        result = np.memmap(
            result_file, dtype=np.float64, mode="w+", shape=bitScore.shape
        )

        # Calculate final weights in chunks
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
    prob: np.ndarray, query_indices: np.ndarray, max_query: int
) -> bool:
    """
    Validate probability array for numerical stability.

    Args:
        prob: Array of probabilities
        query_indices: Array of query indices
        max_query: Maximum query index

    Returns:
        Whether probabilities are valid
    """
    if np.any(prob < 0):
        return False

    prob_sum = np.zeros(max_query + 1, dtype=np.float64)
    np.add.at(prob_sum, query_indices, prob)

    return not np.any(prob_sum == 0)


def chunked_fixed_point_map(
    input_prob: np.ndarray,
    mask: np.ndarray,
    slen: np.ndarray,
    query_inverse_indices: np.ndarray,
    max_query: int,
    mmap_folder: str,
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
) -> np.ndarray:
    """
    Process fixed point mapping using chunked processing.

    Args:
        input_prob: Input probability array
        mask: Mask array
        slen: Array of subject lengths
        query_inverse_indices: Inverse mapping indices for queries
        max_query: Maximum query index
        mmap_folder: Path to memory-mapped folder
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process

    Returns:
        Updated probability array
    """
    new_prob_file = os.path.join(mmap_folder, "new_prob_temp.mmap")

    try:
        # Create memory-mapped array for new probabilities
        new_prob = np.memmap(
            new_prob_file, dtype=np.float64, mode="w+", shape=input_prob.shape
        )
        new_prob[:] = input_prob[:]

        # Calculate masked values
        masked_prob = input_prob[mask].copy()
        masked_slen = slen[mask].copy()

        # Avoid division by zero
        masked_slen[masked_slen == 0] = np.finfo(np.float64).tiny

        # Calculate weight
        s_w = masked_prob / masked_slen
        new_prob[mask] = masked_prob * s_w

        # Calculate probability sum per query
        prob_sum = np.zeros(max_query + 1, dtype=np.float64)

        for start in range(0, len(mask), chunk_size):
            end = min(start + chunk_size, len(mask))
            chunk_mask = mask[start:end]

            if not np.any(chunk_mask):
                continue

            chunk_queries = query_inverse_indices[start:end][chunk_mask]
            chunk_probs = new_prob[start:end][chunk_mask]

            np.add.at(prob_sum, chunk_queries, chunk_probs)

        # Normalize probabilities
        mask_sum = prob_sum[query_inverse_indices[mask]]
        mask_sum[mask_sum == 0] = np.finfo(np.float64).tiny
        new_prob[mask] = new_prob[mask] / mask_sum

        return new_prob

    finally:
        # Cleanup
        try:
            if os.path.exists(new_prob_file):
                os.unlink(new_prob_file)
        except OSError:
            pass


def chunked_squarem_step(
    prob: np.ndarray,
    mask: np.ndarray,
    slen: np.ndarray,
    query_inverse_indices: np.ndarray,
    max_query: int,
    mmap_folder: str,
    step_min: float = -1.0,
    step_max: float = -1.0,
    mstep: int = 4,
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
) -> np.ndarray:
    """
    SQUAREM implementation using chunked processing.

    Args:
        prob: Probability array
        mask: Mask array
        slen: Array of subject lengths
        query_inverse_indices: Inverse mapping indices for queries
        max_query: Maximum query index
        mmap_folder: Path to memory-mapped folder
        step_min: Minimum step size
        step_max: Maximum step size
        mstep: Maximum step attempts
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process

    Returns:
        Updated probability array
    """
    # First fixed-point mapping
    q = chunked_fixed_point_map(
        prob,
        mask,
        slen,
        query_inverse_indices,
        max_query,
        mmap_folder,
        num_threads=num_threads,
        chunk_size=chunk_size,
    )

    r = q - prob
    sr2 = (r**2).sum()

    if sr2 < 1e-10:
        return q

    # Second fixed-point mapping
    q2 = chunked_fixed_point_map(
        q,
        mask,
        slen,
        query_inverse_indices,
        max_query,
        mmap_folder,
        num_threads=num_threads,
        chunk_size=chunk_size,
    )

    r2 = q2 - q
    v = r2 - r
    sv2 = (v**2).sum()
    srv = (r * v).sum()

    if sv2 < 1e-10:
        return q2

    # Calculate step size
    alpha = np.sqrt(sr2 / sv2)
    alpha = np.clip(alpha, step_min, step_max)

    # Create temporary array for new probabilities
    p_new_file = os.path.join(mmap_folder, "p_new_temp.mmap")

    try:
        p_new = np.memmap(p_new_file, dtype=np.float64, mode="w+", shape=prob.shape)
        p_new[:] = prob + 2 * alpha * r + alpha**2 * v

        # Validate probabilities
        if validate_probabilities(p_new[mask], query_inverse_indices[mask], max_query):
            return chunked_fixed_point_map(
                p_new,
                mask,
                slen,
                query_inverse_indices,
                max_query,
                mmap_folder,
                num_threads=num_threads,
                chunk_size=chunk_size,
            )

        # Try with smaller steps
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
                    num_threads=num_threads,
                    chunk_size=chunk_size,
                )

        return q2

    finally:
        # Cleanup
        try:
            if os.path.exists(p_new_file):
                os.unlink(p_new_file)
        except OSError:
            pass


def resolve_multimaps_return_indices(
    subject_inverse_indices: np.ndarray,
    query_inverse_indices: np.ndarray,
    prob: np.ndarray,
    slen: np.ndarray,
    iter_array: np.ndarray,
    mmap_folder: str,
    iters: int = 10,
    step_min: float = -1.0,
    step_max: float = -1.0,
    mstep: int = 4,
    scale: float = 0.9,
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
) -> np.ndarray:
    """
    Resolve multimapped reads using SQUAREM acceleration.

    Args:
        subject_inverse_indices: Inverse mapping indices for subjects
        query_inverse_indices: Inverse mapping indices for queries
        prob: Probability array
        slen: Array of subject lengths
        iter_array: Array for tracking iterations
        mmap_folder: Path to memory-mapped folder
        iters: Number of iterations
        step_min: Minimum step size
        step_max: Maximum step size
        mstep: Maximum step attempts
        scale: Scale factor for filtering
        num_threads: Number of threads to use
        chunk_size: Size of chunks to process

    Returns:
        Mask array for filtered alignments
    """
    # Initialize mask
    mask = np.ones(subject_inverse_indices.shape, dtype=bool)

    # Get total reads and maximum query ID
    total_reads = len(np.unique(query_inverse_indices))
    max_query = query_inverse_indices.max()

    # Initialize iteration variables
    current_iter = 0
    prev_num_alignments = np.inf

    log.info(
        f"Starting multimap resolution: {iters} iterations"
        if iters > 0
        else "Resolving multimaps until convergence"
    )
    log.info(f"Initial alignments: {mask.sum():,}")

    # Create working probability array
    prob_working_file = os.path.join(mmap_folder, "prob_working.mmap")
    prob_working = np.memmap(
        prob_working_file, dtype=np.float64, mode="w+", shape=prob.shape
    )
    prob_working[:] = prob[:]

    try:
        from tqdm import tqdm

        while iters == 0 or current_iter < iters:
            n_alns = mask.sum()

            # Check for convergence
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
                    num_threads=num_threads,
                    chunk_size=chunk_size,
                )
                pbar.update(1)

                # Calculate alignments per read
                n_aln = np.zeros(max_query + 1, dtype=np.int64)
                np.add.at(n_aln, query_inverse_indices[mask], 1)
                n_aln_per_read = n_aln[query_inverse_indices]
                pbar.update(1)

                # Create masks
                unique_mask = n_aln_per_read == 1
                non_unique_mask = n_aln_per_read > 1
                unique_mask &= mask
                non_unique_mask &= mask

                if unique_mask.all():
                    log.info("All reads uniquely mapped - stopping early")
                    break

                # Find maximum probability per read
                max_prob = np.zeros(max_query + 1, dtype=np.float64)
                np.maximum.at(max_prob, query_inverse_indices[mask], prob_working[mask])
                max_prob_scaled = max_prob[query_inverse_indices] * scale
                pbar.update(1)

                # Create final mask for this iteration
                final_mask = (prob_working >= max_prob_scaled) & non_unique_mask
                pbar.update(1)

                # Update iteration tracking
                iter_array[final_mask] = current_iter + 1
                mask &= unique_mask | final_mask
                pbar.update(1)

            # Log progress
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

        # Log final status
        if iters > 0 and current_iter == iters:
            log.info(f"Reached maximum iterations ({iters})")

        return mask.copy()

    finally:
        # Cleanup
        try:
            if os.path.exists(prob_working_file):
                os.unlink(prob_working_file)
        except OSError:
            pass
