import numpy as np
import pandas as pd
import logging
import os
from contextlib import contextmanager
from typing import List, Tuple, Dict, Any, Generator
from tqdm import tqdm


log = logging.getLogger("my_logger")
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@contextmanager
def temp_memmap(
    filename: str, dtype: np.dtype, mode: str, shape: Tuple[int, ...]
) -> Generator[np.memmap, None, None]:
    """Create a temporary memory-mapped array."""
    try:
        memmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        yield memmap_array
    finally:
        if "memmap_array" in locals():
            del memmap_array
        if os.path.exists(filename):
            os.remove(filename)



def initialize_subject_weights(
    subject_inverse_indices: np.ndarray, bitScore: np.ndarray
) -> np.ndarray:
    """Initialize subject weights based on bitScore."""
    total_weights = np.bincount(subject_inverse_indices, weights=bitScore)
    return bitScore / total_weights[subject_inverse_indices]



def resolve_multimaps_return_indices(
    subject_inverse_indices: np.ndarray,
    query_inverse_indices: np.ndarray,
    prob: np.ndarray,
    slen: np.ndarray,
    iter_array: np.ndarray,
    scale: float = 0.9,
    iters: int = 10,
) -> np.ndarray:
    """Resolve multimaps and return indices."""
    mask = np.ones(subject_inverse_indices.shape, dtype=bool)
    total_reads = len(np.unique(query_inverse_indices))
    max_query = query_inverse_indices.max()

    steps = [
        "Calculate weights",
        "Update probs",
        "Calc alignments",
        "Max prob",
        "Remove low-prob",
        "Update mask",
    ]
    total_steps = len(steps)

    prev_num_alignments = np.inf

    for current_iter in range(iters):
        n_alns = mask.sum()
        # log.info(f"Iter: {current_iter + 1} - Total alignments: {n_alns:,}")

        if n_alns == prev_num_alignments:
            log.info("No more alignments removed. Stopping iterations.")
            break

        prev_num_alignments = n_alns

        with tqdm(
            total=total_steps,
            desc=f"Iteration {current_iter + 1}",
            unit="step",
            ncols=80,
            leave=False,
        ) as pbar:
            # Step 1: Calculate subject weights
            s_W = prob[mask] / slen[mask]
            pbar.update(1)

            # Step 2: Update probabilities
            new_prob = prob[mask] * s_W
            prob_sum_array = np.zeros(max_query + 1, dtype=np.float64)
            np.add.at(prob_sum_array, query_inverse_indices[mask], new_prob)
            prob[mask] = new_prob / prob_sum_array[query_inverse_indices[mask]]
            pbar.update(1)

            # Step 3: Calculate number of alignments per query
            n_aln = np.zeros(max_query + 1, dtype=np.int64)
            np.add.at(n_aln, query_inverse_indices[mask], 1)
            n_aln = n_aln[query_inverse_indices]
            pbar.update(1)

            unique_mask = n_aln == 1
            non_unique_mask = n_aln > 1
            unique_mask &= mask
            non_unique_mask &= mask

            if unique_mask.all():
                log.info("No more multimapping reads. Early stopping.")
                break

            # Step 4: Calculate max_prob
            max_prob = np.zeros(max_query + 1, dtype=np.float64)
            np.maximum.at(max_prob, query_inverse_indices[mask], prob[mask])
            max_prob_scaled = max_prob[query_inverse_indices] * scale
            pbar.update(1)

            # Step 5: Remove low-probability alignments
            final_mask = (prob >= max_prob_scaled) & non_unique_mask
            pbar.update(1)

            # Step 6: Update mask
            iter_array[final_mask] = current_iter + 1
            mask &= unique_mask | final_mask
            pbar.update(1)

        # Calculate and print summary
        global_uniques = unique_mask.sum()
        # new_uniques = global_uniques - (total_reads - prev_num_alignments)
        reads_to_process = total_reads - global_uniques
        log.info(
            f"Iter: {current_iter + 1} - Alignments: {n_alns:,} | "
            f"Total uniques: {global_uniques:,} | Reads to process: {reads_to_process:,}"
        )

        if mask.sum() == 0:
            log.info("No more alignments to remove. Stopping.")
            break

    return mask



def reassign(
    np_arrays: Dict[str, np.ndarray], tmp_files: Dict[str, Any]
) -> pd.DataFrame:
    """Reassign and filter alignments."""
    mmap_folder = tmp_files["mmap"]

    filtered_subjectStart = np_arrays["subjectStart"]
    filtered_subjectEnd = np_arrays["subjectEnd"]
    filtered_alnLength = np_arrays["alnLength"]
    filtered_qlen = np_arrays["qlen"]
    filtered_percIdentity = np_arrays["percIdentity"]
    filtered_slen = np_arrays["slen"]
    filtered_subject_numeric_id = np_arrays["subject_numeric_id"]
    filtered_query_numeric_id = np_arrays["query_numeric_id"]
    filtered_bitScore = np_arrays["bitScore"]
    filtered_row_hash = np_arrays["row_hash"]

    log.info("Factorizing subject and query IDs")
    subject_inverse_indices, unique_subjects = pd.factorize(
        filtered_subject_numeric_id, sort=False
    )
    query_inverse_indices, unique_queries = pd.factorize(
        filtered_query_numeric_id, sort=False
    )

    log.info(f"Number of references: {len(unique_subjects):,}")
    log.info(f"Number of reads: {len(unique_queries):,}")

    with temp_memmap(
        f"{mmap_folder}/iter_array.mmap",
        dtype="int64",
        mode="w+",
        shape=(filtered_subject_numeric_id.shape[0],),
    ) as iter_array:
        prob = initialize_subject_weights(subject_inverse_indices, filtered_bitScore)
        final_mask = resolve_multimaps_return_indices(
            subject_inverse_indices,
            query_inverse_indices,
            prob,
            filtered_slen,
            iter_array,
            scale=0.9,
            iters=25,
        )

    filtered_ids_df = pd.DataFrame(
        {
            "query_numeric_id": filtered_query_numeric_id[final_mask],
            "subject_numeric_id": filtered_subject_numeric_id[final_mask],
            "bitScore": filtered_bitScore[final_mask],
            "alnLength": filtered_alnLength[final_mask],
            "subjectStart": filtered_subjectStart[final_mask],
            "subjectEnd": filtered_subjectEnd[final_mask],
            "percIdentity": filtered_percIdentity[final_mask],
            "row_hash": filtered_row_hash[final_mask],
        }
    )

    return filtered_ids_df
