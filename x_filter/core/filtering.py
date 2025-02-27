# x_filter/core/filtering.py

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Union, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

from x_filter.utils.logging import get_logger, LogContext
from x_filter.utils.memory import track_memory
from x_filter.db.mmap import filter_arrays_by_subjects
from x_filter.core.reassignment import reassign_reads

log = get_logger(__name__)


@track_memory(name="filter_arrays", detailed=True)
def filter_arrays(
    stats_df: pd.DataFrame,
    unique_subjects: np.ndarray,
    inverse_indices: np.ndarray,
    numpy_arrays: Dict[str, np.ndarray],
    tmp_files: Dict[str, str],
    skip_reassign: bool = False,
    iters: int = 25,
    max_memory: Union[str, float, int] = "4G",
    num_threads: int = 1,
) -> pd.DataFrame:
    """
    Filter arrays based on coverage statistics and optionally reassign multi-mapped reads.

    Args:
        stats_df: DataFrame with coverage statistics
        unique_subjects: Array of unique subject IDs
        inverse_indices: Array mapping to subject indices
        numpy_arrays: Dictionary of numpy arrays to filter
        tmp_files: Dictionary of temporary file paths
        skip_reassign: Whether to skip read reassignment
        iters: Number of iterations for reassignment
        max_memory: Maximum memory to use
        num_threads: Number of threads to use

    Returns:
        DataFrame with filtered and optionally reassigned alignments
    """
    with LogContext(log, "Filtering arrays"):
        # If no filtering criteria provided, use all alignments
        if stats_df.empty:
            log.info("No filtering criteria - using all alignments")

            if skip_reassign:
                log.info("Skipping multi-mapping resolution")
                return pd.DataFrame(
                    {
                        key: numpy_arrays[key]
                        for key in [
                            "query_numeric_id",
                            "subject_numeric_id",
                            "bitScore",
                            "alnLength",
                            "subjectStart",
                            "subjectEnd",
                            "percIdentity",
                            "row_hash",
                        ]
                    }
                )

            log.info("Resolving multi-mapped reads")
            return reassign_reads(
                numpy_arrays,
                tmp_files,
                iters=iters,
                max_memory=max_memory,
                num_threads=num_threads,
            )

        # Extract subject IDs to keep
        target_subjects = set(stats_df["subject_numeric_id"].values)

        log.info(
            f"Filtering to {len(target_subjects):,} subjects "
            f"({len(target_subjects)/len(unique_subjects):.2%} of total)"
        )

        # Filter arrays based on target subjects
        filtered_arrays = filter_arrays_by_subjects(
            numpy_arrays,
            target_subjects,
            tmp_files["mmap"],
            subject_key="subject_numeric_id",
            num_threads=num_threads,
        )

        # If no alignments pass the filter
        if not filtered_arrays:
            log.warning("No alignments passed the filtering criteria")
            return pd.DataFrame()

        # Return filtered results directly if skipping reassignment
        if skip_reassign:
            log.info("Skipping multi-mapping resolution")
            return pd.DataFrame(
                {
                    key: filtered_arrays[key]
                    for key in [
                        "query_numeric_id",
                        "subject_numeric_id",
                        "bitScore",
                        "alnLength",
                        "subjectStart",
                        "subjectEnd",
                        "percIdentity",
                        "row_hash",
                    ]
                }
            )

        # Reassign multi-mapped reads
        log.info("Resolving multi-mapped reads")
        return reassign_reads(
            filtered_arrays,
            tmp_files,
            iters=iters,
            max_memory=max_memory,
            num_threads=num_threads,
        )


def apply_filters(
    stats_df: pd.DataFrame, filters: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Apply multiple filters to coverage statistics.

    Args:
        stats_df: DataFrame with coverage statistics
        filters: List of filter dictionaries, each with 'name' and 'threshold' keys

    Returns:
        Filtered DataFrame
    """
    if not filters:
        return stats_df

    log.info(f"Applying {len(filters)} filters to {len(stats_df)} references")
    original_count = len(stats_df)

    filtered_df = stats_df.copy()

    for filter_spec in filters:
        filter_type = filter_spec.get("name", "breadth")
        threshold = filter_spec.get("threshold", 0.5)

        log.info(f"Applying {filter_type} filter with threshold {threshold}")

        if filter_type == "breadth":
            filtered_df = filtered_df[filtered_df["breadth"] >= threshold]

        elif filter_type == "depth":
            filtered_df = filtered_df[filtered_df["depth_mean"] >= threshold]

        elif filter_type == "depth_evenness":
            filtered_df = filtered_df[filtered_df["depth_evenness"] <= threshold]

        elif filter_type == "breadth_expected_ratio":
            # Calculate expected breadth where not already present
            if "breadth_expected" not in filtered_df.columns:
                filtered_df["breadth_expected"] = 1 - np.exp(-filtered_df["depth_mean"])

            filtered_df["breadth_expected_ratio"] = (
                filtered_df["breadth"] / filtered_df["breadth_expected"]
            )
            filtered_df = filtered_df[
                filtered_df["breadth_expected_ratio"] >= threshold
            ]

        log.info(
            f"After {filter_type} filter: {len(filtered_df)} references "
            f"({len(filtered_df)/original_count:.2%} of original)"
        )

        if len(filtered_df) == 0:
            log.warning(f"No references passed the {filter_type} filter")
            break

    return filtered_df
