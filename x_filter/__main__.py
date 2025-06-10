from x_filter.logging_setup import setup_logging, get_logger
from x_filter.utils import get_arguments, apply_filters, create_output_files
from x_filter.ops import (
    setup_temporary_directory,
    process_input_data,
    set_memory_limit,
    ColumnInfo,
    export_to_parquet,
    process_parquet_to_memmap,
)
from x_filter.stats import calculate_statistics
# Import the reassign function from the correct module
import importlib.util
import sys
import os

# Load the reassign.py module directly
reassign_module_path = os.path.join(os.path.dirname(__file__), 'reassign.py')
spec = importlib.util.spec_from_file_location("reassign_module", reassign_module_path)
reassign_module = importlib.util.module_from_spec(spec)
sys.modules["reassign_module"] = reassign_module
spec.loader.exec_module(reassign_module)

from x_filter.aggregate import aggregate_gene_abundances, convert_to_anvio
from x_filter.db_manager import DatabaseManager
from x_filter.core_processing import (
    initialize_mmap_array,
    create_filtered_mmap_arrays,
    apply_mask_parallel,
)
from numba import njit, prange

# Standard library imports
import os
import gc
import time
import glob
import tempfile
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Get logger after configuration
log = get_logger()


@njit(parallel=True, fastmath=True)
def _create_subject_mask_ultra_fast(
    alignment_subject_ids: np.ndarray,
    target_subjects: np.ndarray,
    output_mask: np.ndarray,
) -> None:
    """
    Ultra-fast subject mask creation using optimized hash-like lookup.
    """
    n_alignments = len(alignment_subject_ids)
    n_targets = len(target_subjects)
    
    # For small target sets, use linear search
    if n_targets < 100:
        for i in prange(n_alignments):
            subject_id = alignment_subject_ids[i]
            found = False
            for j in range(n_targets):
                if target_subjects[j] == subject_id:
                    found = True
                    break
            output_mask[i] = found
    else:
        # For larger sets, use sorted array with binary search
        sorted_targets = np.sort(target_subjects)
        
        for i in prange(n_alignments):
            subject_id = alignment_subject_ids[i]
            
            # Optimized binary search
            left = 0
            right = n_targets - 1
            found = False
            
            while left <= right:
                mid = left + ((right - left) >> 1)  # Faster division by 2
                mid_val = sorted_targets[mid]
                
                if mid_val == subject_id:
                    found = True
                    break
                elif mid_val < subject_id:
                    left = mid + 1
                else:
                    right = mid - 1
            
            output_mask[i] = found


@njit(parallel=True, fastmath=True)
def _apply_filters_vectorized(
    filter_mask: np.ndarray,
    stat_values: np.ndarray,
    threshold: float,
    operator: int,  # 0 for >=, 1 for <=
) -> None:
    """Ultra-fast vectorized filter application."""
    if operator == 0:  # >=
        for i in prange(len(filter_mask)):
            if filter_mask[i]:
                filter_mask[i] = stat_values[i] >= threshold
    else:  # <=
        for i in prange(len(filter_mask)):
            if filter_mask[i]:
                filter_mask[i] = stat_values[i] <= threshold


def apply_filters_to_mmap(
    stats_arrays: Dict[str, np.memmap],
    filters: List[Dict[str, Any]],
    mmap_folder: str,
) -> np.memmap:
    """
    Ultra-fast filter application using vectorized operations.
    """
    from x_filter.utils import USER_FRIENDLY_FILTER_MAPPING
    
    n_subjects = len(stats_arrays['subject_numeric_id'])
    
    # Create filter mask
    filter_mask = initialize_mmap_array(
        total_positions=n_subjects,
        dtype=np.bool_,
        mmap_folder=mmap_folder,
        array_name="filter_mask",
    )
    
    # Initialize all as True
    filter_mask[:] = True
    
    # Apply each filter using ultra-fast vectorized functions
    for f in filters:
        filter_name = f["filter_name"]
        value = f["value"]
        
        if filter_name not in USER_FRIENDLY_FILTER_MAPPING:
            continue
            
        stat_column, operator = USER_FRIENDLY_FILTER_MAPPING[filter_name]
        
        if stat_column in stats_arrays:
            stat_values = stats_arrays[stat_column]
            
            # Convert operator to integer for numba
            op_int = 0 if operator == ">=" else 1
            
            _apply_filters_vectorized(filter_mask, stat_values, value, op_int)
    
    filter_mask.flush()
    return filter_mask


def process_data(
    args: Any,
    filters: List[Dict[str, Any]],
    tmp_dir: Any,
    tmp_files: Dict[str, str],
    output_files: Dict[str, str] = {},
    enable_initial_filtering: bool = False,
) -> Tuple[Dict[str, np.memmap], np.ndarray, np.memmap, Dict[str, np.memmap], str]:
    """Process data using only memory-mapped arrays."""
    np_arrays, parquet_file = process_input_data(
        args.input,
        (tmp_dir, tmp_files),
        num_threads=args.threads,
        evalue_threshold=args.evalue,
        bitscore_threshold=args.bitscore,
        percent_identity_threshold=args.percent_identity,
        max_memory=args.max_memory,
        keep_db=args.keep_db,
        output_files=output_files,
        mmap_folder_dir=args.mmap_folder_dir,
    )

    if enable_initial_filtering:
        log.info("Getting initial coverage statistics for filtering")
        stats_df, unique_subjects, inverse_indices, numpy_arrays = (
            calculate_statistics(
                np_arrays,
                tmp_files,
                num_threads=args.threads,
                max_memory=args.max_memory,
            )
        )
        
        if filters:
            log.info("Applying initial filters to memory-mapped arrays")
            
            # Convert DataFrame to memmap arrays for filtering
            stats_arrays = {}
            for col in stats_df.columns:
                col_data = stats_df[col].values
                col_mmap = initialize_mmap_array(
                    total_positions=len(col_data),
                    dtype=col_data.dtype,
                    mmap_folder=tmp_files["mmap"],
                    array_name=f"initial_stats_{col}",
                )
                col_mmap[:] = col_data
                col_mmap.flush()
                stats_arrays[col] = col_mmap
            
            filter_mask = apply_filters_to_mmap(
                stats_arrays=stats_arrays,
                filters=filters,
                mmap_folder=tmp_files["mmap"],
            )
            
            # Count filtered subjects
            n_kept = np.sum(filter_mask)
            log.info(f"Filters kept {n_kept:,} out of {len(unique_subjects):,} subjects")
            
            # Return the memmap arrays instead of DataFrame
            return stats_arrays, unique_subjects, inverse_indices, numpy_arrays, parquet_file
        else:
            # No filters, but convert DataFrame to memmap arrays for consistency
            stats_arrays = {}
            for col in stats_df.columns:
                col_data = stats_df[col].values
                col_mmap = initialize_mmap_array(
                    total_positions=len(col_data),
                    dtype=col_data.dtype,
                    mmap_folder=tmp_files["mmap"],
                    array_name=f"initial_stats_{col}",
                )
                col_mmap[:] = col_data
                col_mmap.flush()
                stats_arrays[col] = col_mmap
            
            return stats_arrays, unique_subjects, inverse_indices, numpy_arrays, parquet_file
    else:
        # Skip statistics calculation and filtering when disabled
        stats_arrays = {}
        unique_subjects = np.array([])
        inverse_indices = initialize_mmap_array(
            total_positions=1,
            dtype=np.int64,
            mmap_folder=tmp_files["mmap"],
            array_name="dummy_inverse",
        )
        numpy_arrays = np_arrays

    return stats_arrays, unique_subjects, inverse_indices, numpy_arrays, parquet_file


def efficient_filter_arrays(
    stats_arrays: Dict[str, np.memmap],
    numpy_arrays: Dict[str, np.memmap], 
    tmp_files: Dict[str, str],
    args: Any,
    max_chunk_size: int = 100_000_000,
) -> str:
    """Ultra-fast array filtering using optimized operations."""
    
    if not stats_arrays:
        log.info("No filtering criteria - using all alignments")
        
        if args.skip_reassign:
            # Create a simple ID file with all rowids
            filtered_ids_file = os.path.join(tmp_files["db"], "filtered_ids.parquet")
            
            # Export rowids directly from memory-mapped array
            with DatabaseManager(
                temp_dir=tmp_files['db'],
                threads=args.threads,
                memory_limit=args.max_memory,  # Pass as string
                max_memory_pct=60,
                enable_progress=True
            ) as db_manager:
                # Register the rowid array as a table
                import pyarrow as pa
                rowid_table = pa.table({"rowid": numpy_arrays["rowid"]})
                db_manager.con.register("rowids", rowid_table)
                
                db_manager.execute(
                    f"COPY rowids TO '{filtered_ids_file}' (FORMAT 'parquet')"
                )
            
            return filtered_ids_file
        else:
            log.info("Running reassignment without scale threshold")
            
            # Run reassignment on full arrays with all parameters from CLI
            reassigned_rowids = reassign_module.reassign(
                args=type('Args', (), {
                    'filtered_arrays': numpy_arrays,
                    'mmap_dir': tmp_files["mmap"],
                    'threads': args.threads,
                    'n_iters': args.n_iters,
                    'min_improvement': getattr(args, 'min_improvement', 1e-4),
                    'adaptive_convergence': getattr(args, 'adaptive_convergence', False),
                    'max_memory': args.max_memory,  # Pass as string
                    'selection_mode': getattr(args, 'selection_mode', 'hard_cutoff'),
                    'reference_bias': getattr(args, 'reference_bias', 0.0),
                    'handle_ties': getattr(args, 'handle_ties', 'keep_all'),
                    'min_assignment_confidence': getattr(args, 'min_assignment_confidence', 0.01),
                    'min_confidence_margin': getattr(args, 'min_confidence_margin', 0.0),
                    'acceleration_method': getattr(args, 'acceleration_method', 'hybrid'),
                    'anderson_memory': getattr(args, 'anderson_memory', 10),
                    'lbfgs_memory': getattr(args, 'lbfgs_memory', 10),
                })()
            )
            
            # Handle case where reassignment returns None or empty results
            if reassigned_rowids is None:
                log.error("Reassignment returned None - using all rowids as fallback")
                reassigned_rowids = numpy_arrays["rowid"].tolist()
            elif len(reassigned_rowids) == 0:
                log.warning("Reassignment returned empty list - no alignments selected")
                return None
            
            # Export reassigned rowids - reassigned_rowids is now a numpy array
            reassigned_file = os.path.join(tmp_files["db"], "reassigned_ids.parquet")
            with DatabaseManager(
                temp_dir=tmp_files['db'],
                threads=args.threads,
                memory_limit=args.max_memory,  # Pass as string
                max_memory_pct=60,
                enable_progress=True
            ) as db_manager:
                # Create PyArrow table from the numpy array of row IDs
                import pyarrow as pa
                rowid_table = pa.table({"rowid": pa.array(reassigned_rowids, type=pa.int64())})
                
                # Register and export
                db_manager.con.register("reassigned_rowids", rowid_table)
                db_manager.execute(f"""
                    COPY reassigned_rowids TO '{reassigned_file}' (FORMAT 'parquet')
                """)
            
            return reassigned_file

    # Filtering case - use ultra-fast operations
    filter_mask_file = os.path.join(tmp_files["mmap"], "filter_mask.dat")
    if not os.path.exists(filter_mask_file):
        log.error("Filter mask file not found - filtering may have failed")
        return None
        
    # Load the filter mask
    filter_mask = np.memmap(
        filter_mask_file,
        dtype=np.bool_,
        mode="r",
        shape=(len(stats_arrays['subject_numeric_id']),)
    )
    
    # Ultra-fast subject extraction
    target_subjects = _extract_filtered_subjects(
        stats_arrays['subject_numeric_id'], 
        filter_mask,
        tmp_files["mmap"]
    )
    
    log.info(f"Filtering to {len(target_subjects):,} subjects (out of {len(stats_arrays['subject_numeric_id']):,} total)")

    # Create boolean mask for alignments using ultra-fast method
    alignment_mask = initialize_mmap_array(
        total_positions=len(numpy_arrays["subject_numeric_id"]),
        dtype=np.bool_,
        mmap_folder=tmp_files["mmap"],
        array_name="alignment_filter_mask",
    )
    
    # Use ultra-fast subject mask creation
    _create_subject_mask_ultra_fast(
        numpy_arrays["subject_numeric_id"],
        target_subjects,
        alignment_mask,
    )
    
    n_kept = np.sum(alignment_mask)
    log.info(f"Found {n_kept:,} alignments matching filtered subjects")
    
    if n_kept == 0:
        log.warning("No alignments match filtering criteria")
        return None
    
    if args.skip_reassign:
        # Export filtered rowids
        filtered_ids_file = os.path.join(tmp_files["db"], "filtered_ids.parquet")
        
        # Create filtered rowid array
        filtered_rowids = initialize_mmap_array(
            total_positions=n_kept,
            dtype=numpy_arrays["rowid"].dtype,
            mmap_folder=tmp_files["mmap"],
            array_name="filtered_rowids",
        )
        
        # Copy filtered rowids
        apply_mask_parallel(
            numpy_arrays["rowid"],
            alignment_mask,
            filtered_rowids,
        )
        filtered_rowids.flush()
        
        # Export to Parquet
        with DatabaseManager(
            temp_dir=tmp_files['db'],
            threads=args.threads,
            memory_limit=args.max_memory,
            max_memory_pct=60,
            enable_progress=True
        ) as db_manager:
            import pyarrow as pa
            rowid_table = pa.table({"rowid": filtered_rowids})
            db_manager.con.register("filtered_rowids", rowid_table)
            
            db_manager.execute(
                f"COPY filtered_rowids TO '{filtered_ids_file}' (FORMAT 'parquet')"
            )
        
        return filtered_ids_file
    else:
        log.info("Running reassignment without scale threshold")
        
        # Create filtered arrays for reassignment
        filtered_arrays = create_filtered_mmap_arrays(
            source_arrays=numpy_arrays,
            filter_mask=alignment_mask,
            mmap_folder=tmp_files["mmap"],
            prefix="filtered_for_reassign",
        )
        
        # Run reassignment on filtered data with all parameters from CLI
        reassigned_rowids = reassign_module.reassign(
            args=type('Args', (), {
                'filtered_arrays': filtered_arrays,
                'mmap_dir': tmp_files["mmap"],
                'threads': args.threads,
                'n_iters': args.n_iters,
                'min_improvement': getattr(args, 'min_improvement', 1e-4),
                'adaptive_convergence': getattr(args, 'adaptive_convergence', False),
                'max_memory': args.max_memory,  # Pass as string
                'selection_mode': getattr(args, 'selection_mode', 'hard_cutoff'),
                'reference_bias': getattr(args, 'reference_bias', 0.0),
                'handle_ties': getattr(args, 'handle_ties', 'keep_all'),
                'min_assignment_confidence': getattr(args, 'min_assignment_confidence', 0.01),
                'min_confidence_margin': getattr(args, 'min_confidence_margin', 0.0),
                'acceleration_method': getattr(args, 'acceleration_method', 'hybrid'),
                'anderson_memory': getattr(args, 'anderson_memory', 10),
                'lbfgs_memory': getattr(args, 'lbfgs_memory', 10),
            })()
        )
        
        # Handle case where reassignment returns None or empty results
        if reassigned_rowids is None:
            log.error("Reassignment returned None - using filtered rowids as fallback")
            reassigned_rowids = filtered_arrays["rowid"].tolist()
        elif len(reassigned_rowids) == 0:
            log.warning("Reassignment returned empty list - no alignments selected")
            return None
        
        # Export reassigned rowids
        reassigned_file = os.path.join(tmp_files["db"], "reassigned_ids.parquet")
        with DatabaseManager(
            temp_dir=tmp_files['db'],
            threads=args.threads,
            memory_limit=args.max_memory,  # Pass as string
            max_memory_pct=60,
            enable_progress=True
        ) as db_manager:
            # Create PyArrow table from the numpy array of row IDs
            import pyarrow as pa
            rowid_table = pa.table({"rowid": pa.array(reassigned_rowids, type=pa.int64())})
            
            # Register and export
            db_manager.con.register("reassigned_rowids", rowid_table)
            db_manager.execute(f"""
                COPY reassigned_rowids TO '{reassigned_file}' (FORMAT 'parquet')
            """)
        
        return reassigned_file


def _extract_filtered_subjects(
    subject_ids: np.memmap,
    filter_mask: np.memmap,
    mmap_folder: str,
) -> np.memmap:
    """Extract subjects that passed the filter without using DataFrame operations."""
    n_filtered = np.sum(filter_mask)
    
    # Create output array for filtered subjects
    filtered_subjects = initialize_mmap_array(
        total_positions=n_filtered,
        dtype=subject_ids.dtype,
        mmap_folder=mmap_folder,
        array_name="filtered_subjects",
    )
    
    # Copy filtered subjects
    apply_mask_parallel(subject_ids, filter_mask, filtered_subjects)
    filtered_subjects.flush()
    
    return filtered_subjects


def analyze_alignments_mmap(
    result_file: str,
    tmp_files: Dict[str, str],
    num_threads: int = 1,
    max_memory: str = "8GB",
) -> Dict[str, np.memmap]:
    """Analyze alignments using only memory-mapped arrays."""
    log.info(f"Analyzing alignments from {result_file} using memory mapping")

    # Create memmap subfolder
    memmap_dir = tmp_files["mmap"]
    os.makedirs(memmap_dir, exist_ok=True)

    # Define columns to extract with their types
    columns_info = {
        "subject_numeric_id": ("BIGINT", "int64"),
        "subjectStart": ("INTEGER", "int32"), 
        "subjectEnd": ("INTEGER", "int32"),
        "alnLength": ("INTEGER", "int32"),
        "qlen": ("INTEGER", "int32"),
        "percIdentity": ("FLOAT4", "float32"),
        "slen": ("INTEGER", "int32"),
        "rowid": ("BIGINT", "int64"),
    }

    # Process directly to memory-mapped arrays
    mmap_arrays = process_parquet_to_memmap(
        db_file=result_file,
        total_rows=None,  # Will be determined automatically
        columns_info=columns_info,
        memmap_dir=memmap_dir,
        temp_dir=tmp_files["db"],
        num_threads=num_threads,
        max_memory=max_memory,
        skip_export=True,
    )

    return mmap_arrays


def create_filtered_df_from_arrays(numpy_arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Create filtered DataFrame directly from numpy arrays."""
    return pd.DataFrame(
        {
            "query_numeric_id": numpy_arrays["query_numeric_id"],
            "subject_numeric_id": numpy_arrays["subject_numeric_id"],
            "bitScore": numpy_arrays["bitScore"],
            "alnLength": numpy_arrays["alnLength"],
            "subjectStart": numpy_arrays["subjectStart"],
            "subjectEnd": numpy_arrays["subjectEnd"],
            "percIdentity": numpy_arrays["percIdentity"],
            "rowid": numpy_arrays["rowid"],  # Add rowid to DataFrame creation
        }
    )


def process_filtered_data(
    filtered_ids_path: str,
    db_file: str,
    tmp_files: Dict[str, str],
    args: Any,
) -> str:
    """Process filtered data using DuckDB and write results to optimized Parquet (single file or directory)."""
    log.info(f"Processing filtered IDs from {filtered_ids_path}")

    with DatabaseManager(
        database=db_file,
        temp_dir=tmp_files['db'],
        threads=args.threads,
        memory_limit=args.max_memory,
        max_memory_pct=60,
        enable_progress=True
    ) as db_manager:
        
        # Load filtered IDs
        db_manager.execute(
            f"CREATE TEMPORARY TABLE filtered_ids AS SELECT * FROM read_parquet('{filtered_ids_path}')"
        )
        
        # Join with original data using DuckDB's built-in rowid
        db_manager.execute(
            """
            CREATE TEMPORARY TABLE filtered_results AS
            SELECT fb.*, fb.rowid
            FROM filtered_blast fb
            INNER JOIN filtered_ids fi ON fb.rowid = fi.rowid
            """
        )
        
        # Get total rows for export
        total_rows = db_manager.execute(
            "SELECT COUNT(*) FROM filtered_results"
        ).fetchone()[0]
        log.info(f"Filtered results contain {total_rows:,} rows")

        # Define columns for export - include rowid
        columns_info = {
            "percIdentity": ColumnInfo("percIdentity", "FLOAT4", "float32"),
            "alnLength": ColumnInfo("alnLength", "INTEGER", "int32"),
            "subjectStart": ColumnInfo("subjectStart", "INTEGER", "int32"),
            "subjectEnd": ColumnInfo("subjectEnd", "INTEGER", "int32"),
            "qlen": ColumnInfo("qlen", "INTEGER", "int32"),
            "slen": ColumnInfo("slen", "INTEGER", "int32"),
            "subject_numeric_id": ColumnInfo("subject_numeric_id", "BIGINT", "int64"),
            "query_numeric_id": ColumnInfo("query_numeric_id", "BIGINT", "int64"),
            "bitScore": ColumnInfo("bitScore", "FLOAT4", "float32"),
            "rowid": ColumnInfo("rowid", "BIGINT", "int64"),
        }

        # Export to Parquet using optimized function
        output_path = export_to_parquet(
            db_manager=db_manager,
            columns_info=columns_info,
            output_dir=tmp_files["tmp"],
            total_rows=total_rows,
            keep_db=args.keep_db,
            output_files={},  # Not used here, but required by function signature
            table_name="filtered_results",
        )

        # Check if output_path is a directory or a single file
        if os.path.isdir(output_path):
            # If it's a directory, return the wildcard pattern
            final_output_path = f"{output_path}/*.parquet"
            log.info(
                f"Results written to directory {output_path} with files matching {final_output_path}"
            )
        else:
            # If it's a single file, return the file path directly
            final_output_path = output_path
            log.info(f"Results written to single file {final_output_path}")

    return final_output_path


def cleanup_mmap_files(mmap_folder: str) -> None:
    """Clean up memory mapped files in the given folder with robust handling"""
    if not os.path.exists(mmap_folder):
        return

    # First do a garbage collection to release any lingering references
    gc.collect()

    # Process all files in the directory and subdirectories
    for root, dirs, files in os.walk(mmap_folder, topdown=False):
        for filename in files:
            if filename.startswith(".nfs"):
                continue  # Skip NFS temporary files

            file_path = os.path.join(root, filename)
            try:
                # For memmap files, try to ensure they're properly closed
                if filename.endswith(".dat"):
                    try:
                        # Try to open and immediately close to check if it's locked
                        with open(file_path, "a"):
                            pass
                    except PermissionError:
                        # If file is locked, force garbage collection and wait
                        log.warning(
                            f"File {file_path} appears to be locked. Forcing cleanup..."
                        )
                        gc.collect()
                        time.sleep(2)

                # Try to delete the file
                if os.path.exists(file_path):
                    os.unlink(file_path)

            except Exception as e:
                log.warning(f"Error deleting file {file_path}: {e}")
                # In case of failure, try to make the file writable and retry
                try:
                    os.chmod(file_path, 0o666)
                    os.unlink(file_path)
                except Exception:
                    pass

        # Now try to delete the empty directories (bottom-up due to topdown=False)
        if root != mmap_folder:  # Don't delete the main mmap folder yet
            try:
                os.rmdir(root)
            except Exception as e:
                log.warning(f"Error deleting directory {root}: {e}")

def cleanup_db_files(db_file: str) -> None:
    """Clean up database files"""
    if os.path.exists(db_file):
        try:
            os.unlink(db_file)
        except Exception as e:
            log.warning(f"Error deleting file {db_file}: {e}")

def cleanup_temp_files(tmp_files: Dict[str, str]) -> None:
    """Clean up all temporary files and directories"""
    for dir_type, dir_path in tmp_files.items():
        cleanup_mmap_files(dir_path)
        try:
            if os.path.exists(dir_path):
                os.rmdir(dir_path)
        except Exception as e:
            log.warning(f"Error deleting directory {dir_path}: {e}")

def analyze_alignments(
    result_file: str, chunk_size: int = 1_000_000
) -> Dict[str, np.ndarray]:
    """Memory efficient version using chunked processing."""
    log.info(f"Analyzing alignments")

    with duckdb.connect() as con:
        con.execute("SET memory_limit='4GB'")  # Limit memory usage

        # Use streaming counts
        single_count = 0
        multi_count = 0
        total_unique = 0

        # Process in chunks
        query = f"""
        SELECT query_numeric_id, COUNT(*) as count
        FROM read_parquet('{result_file}')
        GROUP BY query_numeric_id
        """

        for chunk in con.execute(query).fetch_arrow_chunks():
            df_chunk = chunk.to_pandas()
            single_count += (df_chunk["count"] == 1).sum()
            multi_count += (df_chunk["count"] > 1).sum()
            total_unique += len(df_chunk)
            del df_chunk  # Explicit cleanup

        log.info(f"Number of unique reads: {total_unique:,}")
        log.info(f"Number of single alignment: {single_count:,}")
        log.info(f"Number of multi-alignment: {multi_count:,}")

        # Use memory-mapped output for large arrays
        arrays = {}
        mmap_dir = "tmp_mmap"
        os.makedirs(mmap_dir, exist_ok=True)

        # Get total count first
        total_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{result_file}')"
        ).fetchone()[0]

        # Create memory-mapped arrays
        for col in [
            "subject_numeric_id",
            "subjectStart",
            "subjectEnd",
            "alnLength",
            "qlen",
            "percIdentity",
            "slen",
        ]:
            mmap_file = os.path.join(mmap_dir, f"{col}.mmap")
            dtype = np.int32 if col not in ["percIdentity"] else np.float32
            arrays[col] = np.memmap(
                mmap_file, dtype=dtype, mode="w+", shape=(total_rows,)
            )

        # Process in chunks
        offset = 0
        chunk_query = f"""
        SELECT subject_numeric_id, subjectStart, subjectEnd, alnLength,
               qlen, percIdentity, slen
        FROM read_parquet('{result_file}')
        """

        for chunk in con.execute(chunk_query).fetch_arrow_chunks():
            df_chunk = chunk.to_pandas()
            chunk_size = len(df_chunk)

            for col in arrays:
                arrays[col][offset : offset + chunk_size] = df_chunk[col].values

            offset += chunk_size
            del df_chunk

        # Ensure arrays are flushed
        for arr in arrays.values():
            arr.flush()

    return arrays


def analyze_alignments_mmap(
    result_file: str,
    tmp_files: Dict[str, str],
    num_threads: int = 1,
    max_memory: Optional[int] = None,
) -> Dict[str, np.memmap]:
    """
    Analyze alignments from a Parquet file or directory using memory-mapped arrays for large datasets.

    Args:
        result_file: Path to Parquet file or directory of Parquet files
        tmp_files: Dictionary with paths to temporary directories
        num_threads: Number of threads to use for processing
        max_memory: Maximum memory to use in bytes (optional)

    Returns:
        Dictionary of memory-mapped arrays with alignment data
    """
    log.info(f"Analyzing alignments from {result_file} using memory mapping")

    # Create memmap subfolder
    memmap_dir = tmp_files["mmap"]
    os.makedirs(memmap_dir, exist_ok=True)

    # Define columns to extract with their types (DuckDB type, NumPy type) - include rowid
    columns_info = {
        "subject_numeric_id": ("BIGINT", "int64"),
        "subjectStart": ("INTEGER", "int32"),
        "subjectEnd": ("INTEGER", "int32"),
        "alnLength": ("INTEGER", "int32"),
        "qlen": ("INTEGER", "int32"),
        "percIdentity": ("FLOAT4", "float32"),
        "slen": ("INTEGER", "int32"),
        "rowid": ("BIGINT", "int64"),  # Include rowid in analysis
    }

    with duckdb.connect() as con:
        # Configure DuckDB
        con.execute(f"SET threads={num_threads}")
        con.execute(f"SET temp_directory='{tmp_files['db']}'")
        con.execute("SET enable_progress_bar=true")

        # Create temporary table from result file
        con.execute(
            f"CREATE TEMPORARY TABLE alignments AS SELECT * FROM read_parquet('{result_file}')"
        )

        # Get counts for query stats
        query_counts = con.execute(
            """
            SELECT query_numeric_id, COUNT(*) as count
            FROM alignments
            GROUP BY query_numeric_id
            """
        ).fetchall()

        # Calculate alignment stats
        single_count = sum(1 for qid, cnt in query_counts if cnt == 1)
        multi_count = sum(1 for qid, cnt in query_counts if cnt > 1)
        total_unique = len(query_counts)

        # Log statistics
        log.info(f"Number of unique reads: {total_unique:,}")
        log.info(f"Number of single alignment: {single_count:,}")
        log.info(f"Number of multi-alignment: {multi_count:,}")

        # Get total row count
        total_rows = con.execute("SELECT COUNT(*) FROM alignments").fetchone()[0]
        log.info(f"Total alignments to process: {total_rows:,}")

        # Export data to a temporary parquet file - include rowid
        temp_parquet_file = os.path.join(tmp_files["db"], "temp_alignments.parquet")

        # Export the needed columns to parquet
        export_query = f"""
        COPY (
            SELECT
                subject_numeric_id,
                subjectStart,
                subjectEnd,
                alnLength,
                qlen,
                percIdentity,
                slen,
                rowid
            FROM alignments
        ) TO '{temp_parquet_file}' (FORMAT 'parquet')
        """
        con.execute(export_query)

    # Process the exported parquet file into memory-mapped arrays
    mmap_arrays = process_parquet_to_memmap(
        db_file=temp_parquet_file,  # Use the exported parquet file
        total_rows=total_rows,
        columns_info=columns_info,
        memmap_dir=memmap_dir,
        temp_dir=tmp_files["db"],
        num_threads=num_threads,
        max_memory=max_memory,
        skip_export=True,  # Skip export since we already created the parquet file
    )

    # Clean up temporary parquet file
    try:
        os.remove(temp_parquet_file)
    except Exception as e:
        log.warning(f"Failed to remove temporary parquet file: {e}")

    return mmap_arrays


def save_results(
    final_stats: Dict[str, np.memmap],  # Changed from pd.DataFrame to Dict
    result_file: str,
    out_files: Dict[str, str],
    mapping_file: str,
    anvio: bool,
    annotation_source: str,
    tmp_files: Dict[str, str],
    threads: int = 1,
) -> None:
    with DatabaseManager(
        temp_dir=tmp_files['db'],
        threads=threads,
        memory_limit=None,  # Use auto-detection for this case
        max_memory_pct=60,
        enable_progress=True
    ) as db_manager:
        # Convert memmap arrays to PyArrow table for DuckDB registration
        import pyarrow as pa
        
        # Create PyArrow table from memmap arrays
        final_stats_table = pa.table({
            col_name: pa.array(arr) for col_name, arr in final_stats.items()
        })
        
        # Register final_stats table
        db_manager.con.register("final_stats", final_stats_table)

        # Create a temporary view for result files
        db_manager.execute(
            f"CREATE TEMPORARY VIEW alignments AS SELECT * FROM read_parquet('{result_file}')"
        )

        # Get column info from the view
        table_info = db_manager.execute("PRAGMA table_info('alignments')").fetchall()
        columns = [col[1].lower() for col in table_info]
        cigar_columns = ", cigar, qaln, taln" if "cigar" in columns else ""
        all_columns = f"queryId, subjectId, percIdentity, alnLength, mismatchCount, gapOpenCount, queryStart, queryEnd, subjectStart, subjectEnd, eVal, bitScore, qlen, slen{cigar_columns}"

        # Create unique subjects view
        unique_subjects_query = """
            SELECT DISTINCT subjectId, subject_numeric_id
            FROM alignments
        """
        db_manager.execute(
            "CREATE TEMPORARY TABLE unique_subjects AS " + unique_subjects_query
        )

        # Export coverage results
        coverage_query = """
            WITH merged AS (
                SELECT 
                    us.subjectId as reference,
                    fs.depth_mean,
                    fs.depth_std,
                    fs.depth_evenness,
                    fs.breadth,
                    fs.num_alignments as n_alns,
                    fs.avg_read_length,
                    fs.std_read_length as stdev_read_length,
                    fs.avg_alignment_length,
                    fs.avg_identity,
                    fs.std_identity as stdev_identity
                FROM final_stats fs
                LEFT JOIN unique_subjects us ON fs.subject_numeric_id = us.subject_numeric_id
            )
            SELECT 
                reference,
                depth_mean,
                depth_std,
                depth_evenness,
                breadth,
                n_alns,
                avg_read_length,
                stdev_read_length,
                avg_alignment_length,
                avg_identity,
                stdev_identity
            FROM merged
        """
        db_manager.execute(
            f"COPY ({coverage_query}) TO '{out_files['coverage']}' (HEADER, DELIMITER '\t')"
        )

        # Export multimap results
        db_manager.execute(
            f"COPY (SELECT {all_columns} FROM alignments) TO '{out_files['multimap']}' (HEADER, DELIMITER '\t')"
        )

        # Gene abundances (if applicable)
        if mapping_file:
            log.info("Aggregating gene abundances")
            # Convert final_stats back to DataFrame for gene abundances function
            final_stats_df = pd.DataFrame({
                col_name: np.array(arr) for col_name, arr in final_stats.items()
            })
            
            gene_abundances, gene_abundances_agg = aggregate_gene_abundances(
                mapping_file=mapping_file,
                gene_abundances=final_stats_df,
                num_threads=threads,
                temp_dir=tmp_files["db"],
            )

            if gene_abundances is None:
                log.info("Couldn't map anything to the references.")
                return

            db_manager.register("gene_abundances", gene_abundances)
            db_manager.register("gene_abundances_agg", gene_abundances_agg)

            db_manager.execute(
                f"COPY (SELECT * FROM gene_abundances) TO '{out_files['group_abundances']}' "
                "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
            )
            db_manager.execute(
                f"COPY (SELECT * FROM gene_abundances_agg) TO '{out_files['group_abundances_agg']}' "
                "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
            )

            if anvio:
                gene_abundances_anvio = convert_to_anvio(
                    df=gene_abundances, annotation_source=annotation_source
                )
                db_manager.register("gene_abundances_anvio", gene_abundances_anvio)
                db_manager.execute(
                    f"COPY (SELECT * FROM gene_abundances_anvio) TO '{out_files['group_abundances_anvio']}' "
                    "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
                )


def close_and_cleanup_mmap_arrays(
    arrays: Dict[str, np.ndarray], tmp_files: Dict[str, str]
) -> None:
    """Explicitly close memory-mapped arrays and perform thorough cleanup"""
    # First delete all references to the arrays
    if arrays:
        log.info("Explicitly closing memory-mapped arrays")
        for name, arr in arrays.items():
            if isinstance(arr, np.memmap):
                try:
                    # For numpy memmap arrays, explicitly close/delete
                    del arr
                except Exception as e:
                    log.warning(f"Error closing memmap array {name}: {e}")

        # Clear the dictionary itself
        arrays.clear()
    # Force multiple garbage collections
    gc.collect()
    gc.collect()
    # Allow some time for OS to release file locks
    time.sleep(1)
    # Now cleanup the directory with extra retries
    retry_cleanup_mmap_files(tmp_files["mmap"])


def retry_cleanup_mmap_files(mmap_path, max_attempts=3, delay=1):
    """
    Attempt to clean up memory-mapped files with retry logic.

    Args:
        mmap_path: Path to the mmap file or directory containing mmap files
        max_attempts: Maximum number of retry attempts per file
        delay: Delay in seconds between retry attempts

    Returns:
        tuple: (success_count, failed_files) - Number of successfully cleaned files and list of files that couldn't be cleaned
    """
    log = logging.getLogger(__name__)
    success_count = 0
    failed_files = []

    if not mmap_path:
        log.debug("No memory-mapped path provided to clean up")
        return success_count, failed_files

    # Handle both file and directory cases
    if os.path.isdir(mmap_path):
        files_to_clean = glob.glob(os.path.join(mmap_path, "*"))
        # Also try to remove the directory itself after cleaning files
        dir_to_remove = mmap_path
    else:
        files_to_clean = [mmap_path] if os.path.exists(mmap_path) else []
        dir_to_remove = None

    # Clean individual files
    for file_path in files_to_clean:
        for attempt in range(max_attempts):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    log.debug(f"Successfully removed mmap file: {file_path}")
                    success_count += 1
                break
            except (OSError, PermissionError) as e:
                if attempt < max_attempts - 1:
                    log.debug(
                        f"Attempt {attempt+1}/{max_attempts} to remove {file_path} failed: {e}. Retrying..."
                    )
                    time.sleep(delay)
                else:
                    log.warning(
                        f"Failed to remove mmap file {file_path} after {max_attempts} attempts: {e}"
                    )
                    failed_files.append(file_path)

    # Try to remove the directory if it was a directory
    if dir_to_remove and os.path.exists(dir_to_remove):
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(dir_to_remove)
                log.debug(f"Successfully removed mmap directory: {dir_to_remove}")
                break
            except (OSError, PermissionError) as e:
                if attempt < max_attempts - 1:
                    log.debug(
                        f"Attempt {attempt+1}/{max_attempts} to remove directory {dir_to_remove} failed: {e}. Retrying..."
                    )
                    time.sleep(delay)
                else:
                    log.warning(
                        f"Failed to remove mmap directory {dir_to_remove} after {max_attempts} attempts: {e}"
                    )
                    failed_files.append(dir_to_remove)

    return success_count, failed_files


def safe_cleanup(tmp_dir_obj):
    """
    A safer version of tempfile's cleanup that handles potential issues with memory-mapped files.
    This function is designed to replace the standard cleanup method of TemporaryDirectory
    to better handle cases where memory-mapped files might still be locked.

    Args:
        tmp_dir_obj: The TemporaryDirectory object whose cleanup method is being replaced
    """
    import tempfile

    # Get the directory path before any cleanup happens
    dir_path = tmp_dir_obj.name if hasattr(tmp_dir_obj, "name") else None
    if not dir_path or not os.path.exists(dir_path):
        return

    # Force garbage collection to release file handles
    gc.collect()
    gc.collect()
    # Allow some time for resources to be released
    time.sleep(0.5)

    try:
        # Try to use the original cleanup method first
        original_cleanup = getattr(tempfile.TemporaryDirectory, "_cleanup", None)
        if original_cleanup:
            original_cleanup(tmp_dir_obj)
        else:
            # If we can't access the original cleanup, use rmtree directly
            shutil.rmtree(dir_path, ignore_errors=True)
        log.debug(f"Successfully cleaned up temporary directory: {dir_path}")
    except Exception as e:
        log.warning(f"Standard cleanup of temporary directory {dir_path} failed: {e}")
        # Try our more aggressive cleanup approach
        try:
            # If dir still exists, try force_delete_mmap_folder
            if os.path.exists(dir_path):
                log.info(f"Attempting aggressive cleanup of {dir_path}")
                force_delete_mmap_folder(dir_path)
        except Exception as e2:
            log.warning(f"Failed to perform aggressive cleanup of {dir_path}: {e}")


def force_delete_mmap_folder(mmap_folder: str) -> None:
    """Aggressive cleanup of memory-mapped files - to be called before tmp_dir_obj.cleanup()"""

    if not os.path.exists(mmap_folder):
        return

    log.info(f"Forcibly cleaning mmap directory: {mmap_folder}")

    # First try to dereference any remaining arrays and force garbage collection
    gc.collect()
    gc.collect()
    time.sleep(1)  # Give OS time to release locks

    # Get list of all files
    all_files = []
    for root, dirs, files in os.walk(mmap_folder, topdown=False):
        for f in files:
            all_files.append(os.path.join(root, f))

    if all_files:
        log.info(f"Found {len(all_files)} files to remove")

        # Try to delete each file individually
        for file_path in all_files:
            try:
                os.unlink(file_path)
            except Exception as e:
                try:
                    # Try to make writable and delete
                    os.chmod(file_path, 0o666)
                    os.unlink(file_path)
                except Exception:
                    log.warning(f"Failed to remove: {file_path}")

    # Try to remove directory tree with force option
    try:
        # Using system command as a last resort
        if os.path.exists(mmap_folder):
            log.info(f"Using rm -rf to force delete: {mmap_folder}")
            os.system(f"rm -rf {mmap_folder}")
    except Exception as e:
        log.warning(f"Failed to force-delete mmap folder: {e}")


def main() -> None:
    args, filters = get_arguments()
    setup_logging(args.debug)
    out_files = create_output_files(prefix=args.prefix, input_file=args.input)
    tmp_dir_obj, tmp_files = setup_temporary_directory(base_dir=args.tmp_dir)
    np_arrays = None  # Track the arrays for cleanup

    try:
        final_stats, unique_subjects, inverse_indices, numpy_arrays, db_file = (
            process_data(
                args,
                filters,
                tmp_dir_obj,
                tmp_files,
                out_files,
                enable_initial_filtering=args.enable_initial_filtering,
            )
        )

        filtered_ids_path = efficient_filter_arrays(
            final_stats,
            numpy_arrays,
            tmp_files,
            args,
            max_chunk_size=100_000_000,  # Can be adjusted based on dataset size and memory
        )
        if filtered_ids_path is None:
            log.warning("No matching IDs found. Exiting.")
            return

        log.info(
            f"Number of alignments after filtering: Reading from {filtered_ids_path}"
        )
        result_file = process_filtered_data(filtered_ids_path, db_file, tmp_files, args)

        # Analyze directly from file
        np_arrays = analyze_alignments_mmap(
            result_file=result_file,
            tmp_files=tmp_files,
            num_threads=args.threads,
            max_memory=args.max_memory,
        )

        # Recalculate statistics - returns memmap arrays, not DataFrame
        log.info("Getting coverage statistics")
        final_stats_df, unique_subjects, inverse_indices, numpy_arrays = (
            calculate_statistics(
                np_arrays,
                tmp_files,
                num_threads=args.threads,
                max_memory=args.max_memory,
            )
        )
        del inverse_indices

        # Convert DataFrame to memmap arrays for consistent handling
        final_stats_mmap = {}
        for col in final_stats_df.columns:
            col_data = final_stats_df[col].values
            col_mmap = initialize_mmap_array(
                total_positions=len(col_data),
                dtype=col_data.dtype,
                mmap_folder=tmp_files["mmap"],
                array_name=f"final_stats_{col}",
            )
            col_mmap[:] = col_data
            col_mmap.flush()
            final_stats_mmap[col] = col_mmap

        # Apply final filters using memmap arrays
        if filters:
            log.info("Applying final filters")
            filter_mask = apply_filters_to_mmap(
                stats_arrays=final_stats_mmap,
                filters=filters,
                mmap_folder=tmp_files["mmap"],
            )
            
            # Count remaining subjects
            n_kept = np.sum(filter_mask)
            log.info(f"References kept: {n_kept:,}")
            
            # Create filtered final_stats
            filtered_final_stats = {}
            for col_name, col_array in final_stats_mmap.items():
                filtered_array = initialize_mmap_array(
                    total_positions=n_kept,
                    dtype=col_array.dtype,
                    mmap_folder=tmp_files["mmap"],
                    array_name=f"filtered_final_stats_{col_name}",
                )
                apply_mask_parallel(col_array, filter_mask, filtered_array)
                filtered_array.flush()
                filtered_final_stats[col_name] = filtered_array
            
            final_stats_mmap = filtered_final_stats
            
            # Filter result_file based on filtered subjects
            filtered_result_file = os.path.join(
                tmp_files["db"], "final_filtered_results.parquet"
            )
            
            # Get filtered subject IDs
            filtered_subject_ids = final_stats_mmap['subject_numeric_id']
            
            with DatabaseManager(
                temp_dir=tmp_files['db'],
                threads=args.threads,
                memory_limit=args.max_memory,
                max_memory_pct=60,
                enable_progress=True
            ) as db_manager:
                # Register filtered subject IDs
                import pyarrow as pa
                filtered_subjects_table = pa.table({"subject_numeric_id": filtered_subject_ids})
                db_manager.con.register("filtered_subjects", filtered_subjects_table)
                
                db_manager.execute(
                    f"""
                    COPY (
                        SELECT r.*
                        FROM read_parquet('{result_file}') r
                        INNER JOIN filtered_subjects fs
                        ON r.subject_numeric_id = fs.subject_numeric_id
                    ) TO '{filtered_result_file}' (FORMAT 'parquet')
                """
                )
            result_file = filtered_result_file

        # Save results
        save_results(
            final_stats=final_stats_mmap,  # Pass memmap arrays instead of DataFrame
            result_file=result_file,
            out_files=out_files,
            tmp_files=tmp_files,
            threads=args.threads,
            mapping_file=args.mapping_file,
            anvio=args.anvio,
            annotation_source=args.annotation_source,
        )
    except Exception as e:
        log.error(f"Error occurred: {e}")
        raise e
    finally:
        # Cleanup all references explicitly
        time.sleep(1)  # Allow some time for OS to release file locks
        close_and_cleanup_mmap_arrays(np_arrays, tmp_files)
        
        # Try cleanup with error handling for NFS issues
        try:
            tmp_dir_obj.cleanup()
        except OSError as e:
            if "Device or resource busy" in str(e) or "nfs" in str(e).lower():
                log.warning(f"NFS cleanup issue (this is normal on shared filesystems): {e}")
                log.info("Results were saved successfully despite cleanup warning")
            else:
                raise
        
        log.info("ALL DONE.")


@njit(parallel=True, fastmath=True)
def _create_subject_mask(
    alignment_subject_ids: np.ndarray,
    target_subjects: np.ndarray,
    output_mask: np.ndarray,
) -> None:
    """Create boolean mask for alignments that match target subjects."""
    # First, sort target subjects for binary search
    sorted_targets = np.sort(target_subjects)
    
    # Process alignments in parallel
    for i in prange(len(alignment_subject_ids)):
        subject_id = alignment_subject_ids[i]
        
        # Binary search for subject_id in sorted_targets
        left = 0
        right = len(sorted_targets) - 1
        found = False
        
        while left <= right:
            mid = (left + right) // 2
            if sorted_targets[mid] == subject_id:
                found = True
                break
            elif sorted_targets[mid] < subject_id:
                left = mid + 1
            else:
                right = mid - 1
        
        output_mask[i] = found
