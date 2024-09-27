import numpy as np
import pandas as pd
import logging
import os
import tqdm
from contextlib import contextmanager
from memory_profiler import profile

log = logging.getLogger("my_logger")
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@contextmanager
def temp_memmap(filename, dtype, mode, shape):
    try:
        memmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        yield memmap_array
    finally:
        if "memmap_array" in locals():
            del memmap_array
        if os.path.exists(filename):
            os.remove(filename)


def initialize_subject_weights(subject_inverse_indices, bitScore, mmap_dir):
    prob_filename = os.path.join(mmap_dir, "prob.mmap")
    prob = np.memmap(
        prob_filename, dtype="float64", mode="w+", shape=subject_inverse_indices.shape
    )
    total_weights = np.bincount(subject_inverse_indices, weights=bitScore)
    prob[:] = bitScore / total_weights[subject_inverse_indices]
    return prob


def cleanup_iter_files(mmap_dir, current_iter):
    for filename in os.listdir(mmap_dir):
        if filename.endswith(f"_{current_iter}.mmap"):
            os.remove(os.path.join(mmap_dir, filename))


def resolve_multimaps_return_indices(
    subject_inverse_indices,
    query_inverse_indices,
    prob,
    slen,
    iter_array,
    scale=0.9,
    iters=10,
    mmap_dir=None,
):
    try:
        current_iter = 0
        prev_num_alignments = np.inf
        mask = np.ones(subject_inverse_indices.shape, dtype=bool)
        global_uniques = 0
        total_reads = len(np.unique(query_inverse_indices))

        steps = [
            "Calculate subject weights",
            "Update probabilities",
            "Calculate alignments per query",
            "Calculate max_prob",
            "Remove low-probability alignments",
            "Update mask",
        ]
        total_steps = len(steps)

        while current_iter < iters:
            n_alns = np.sum(mask)
            log.info(f"Iter: {current_iter + 1} - Total alignments: {n_alns:,}")

            if n_alns == prev_num_alignments:
                log.info("No more alignments removed. Stopping iterations.")
                break

            prev_num_alignments = n_alns

            with tqdm.tqdm(
                total=total_steps,
                desc=f"Iteration {current_iter + 1}",
                unit="step",
                ncols=80,
                leave=False,
            ) as pbar:
                # Step 1: Calculate subject weights
                with temp_memmap(
                    os.path.join(mmap_dir, f"subject_weights_{current_iter}.mmap"),
                    dtype="float64",
                    mode="w+",
                    shape=subject_inverse_indices.shape,
                ) as subject_weights:
                    subject_weights[mask] = prob[mask]
                    s_W = subject_weights[mask] / slen[mask]
                pbar.update(1)

                # Step 2: Update probabilities
                new_prob = prob[mask] * s_W
                max_query = np.max(query_inverse_indices[mask])
                with temp_memmap(
                    os.path.join(mmap_dir, f"prob_sum_array_{current_iter}.mmap"),
                    dtype="float64",
                    mode="w+",
                    shape=(max_query + 1,),
                ) as prob_sum_array:
                    np.add.at(prob_sum_array, query_inverse_indices[mask], new_prob)
                    prob[mask] = new_prob / prob_sum_array[query_inverse_indices[mask]]
                pbar.update(1)

                # Step 3: Calculate number of alignments per query
                with temp_memmap(
                    os.path.join(mmap_dir, f"query_counts_array_{current_iter}.mmap"),
                    dtype="int64",
                    mode="w+",
                    shape=(max_query + 1,),
                ) as query_counts:
                    np.add.at(query_counts, query_inverse_indices[mask], 1)
                    n_aln = query_counts[query_inverse_indices]
                pbar.update(1)

                unique_mask = n_aln == 1
                non_unique_mask = n_aln > 1
                unique_mask = unique_mask & mask
                non_unique_mask = non_unique_mask & mask

                if np.all(unique_mask):
                    log.info("No more multimapping reads. Early stopping.")
                    break

                # Step 4: Calculate max_prob
                with temp_memmap(
                    os.path.join(mmap_dir, f"max_prob_{current_iter}.mmap"),
                    dtype="float64",
                    mode="w+",
                    shape=(max_query + 1,),
                ) as max_prob:
                    np.maximum.at(max_prob, query_inverse_indices[mask], prob[mask])
                    max_prob_scaled = max_prob[query_inverse_indices] * scale
                pbar.update(1)

                # Step 5: Remove low-probability alignments
                final_mask = (prob >= max_prob_scaled) & non_unique_mask
                to_remove = np.sum(mask) - np.sum(final_mask)
                total_n_unique = np.sum(unique_mask)
                pbar.update(1)

                # Step 6: Update mask
                iter_array[final_mask] = current_iter + 1
                mask = mask & (unique_mask | final_mask)
                pbar.update(1)

            # Calculate and print summary
            new_uniques = np.sum(unique_mask) - global_uniques
            global_uniques = np.sum(unique_mask)
            reads_to_process = total_reads - global_uniques

            log.info(
                f"Iter: {current_iter + 1} - New uniques: {new_uniques:,}|Total uniques: {global_uniques:,}|Reads to process: {reads_to_process:,}"
            )

            if np.sum(mask) == 0:
                log.info("No more alignments to remove. Stopping.")
                break

            current_iter += 1
            cleanup_iter_files(mmap_dir, current_iter)

        return mask
    finally:
        prob_filename = prob.filename
        del prob
        if os.path.exists(prob_filename):
            os.remove(prob_filename)


def reassign(np_arrays, tmp_files):
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

    # log.info("Factorizing subject and query IDs")
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
        prob = initialize_subject_weights(
            subject_inverse_indices, filtered_bitScore, mmap_dir=mmap_folder
        )
        try:
            final_mask = resolve_multimaps_return_indices(
                subject_inverse_indices,
                query_inverse_indices,
                prob,
                filtered_slen,
                iter_array,
                scale=0.9,
                iters=25,
                mmap_dir=mmap_folder,
            )
        finally:
            prob_filename = prob.filename
            del prob
            if os.path.exists(prob_filename):
                os.remove(prob_filename)

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

    # Clean up any remaining temporary files
    for filename in os.listdir(mmap_folder):
        if filename.endswith(".mmap"):
            os.remove(os.path.join(mmap_folder, filename))

    return filtered_ids_df
