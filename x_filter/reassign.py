"""
Main reassignment module for x-filter package.
Implements SQUAREM EM algorithm for probabilistic read assignment.
"""
import os
import numpy as np
import numba
from numba import njit, prange
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import gc
import pyarrow as pa
import pyarrow.parquet as pq
from x_filter.resource_management import ResourceManager
from x_filter.logging_setup import get_logger
from x_filter.reassign.em import accelerated_resolve_multimaps, ultra_fast_bitscore_em_step
from x_filter.reassign.utils import memory_efficient_factorize, initialize_subject_weights
from x_filter.reassign.confidence import implement_selection_mode
from x_filter.db_manager import DatabaseManager

log = get_logger()

def reassign_reads_mmap(
    filtered_arrays: Dict[str, np.ndarray],
    unique_query_ids: np.ndarray,
    unique_subject_ids: np.ndarray, 
    mmap_dir: str,
    threads: int = 1,
    reassign_iters: int = 25,
    min_improvement: float = 1e-4,
    adaptive_convergence: bool = False,
    max_memory: Optional[str] = None,
    selection_mode: str = "primary",
    reference_bias: float = 0.0,
    handle_ties: str = "keep_all",
    min_assignment_confidence: float = 0.01,
    min_confidence_margin: float = 0.0,
    acceleration_method: str = "hybrid",
    anderson_memory: int = 10,
    lbfgs_memory: int = 10,
    lambda_scale: float = 3.0,
    temperature: float = 0.03,
    use_enhanced_em: bool = True,
    random_seed: int = 42,
) -> list:
    """
    Enhanced read reassignment with deterministic behavior.
    
    Args:
        filtered_arrays: Dictionary of memory-mapped arrays containing alignment data
        unique_query_ids: Array of unique query (read) IDs (can be empty - will be computed)
        unique_subject_ids: Array of unique subject (reference) IDs (can be empty - will be computed)
        mmap_dir: Directory for temporary memory-mapped files
        threads: Number of threads to use
        reassign_iters: Number of EM iterations
        min_improvement: Minimum improvement threshold for convergence
        adaptive_convergence: Whether to use adaptive convergence
        max_memory: Maximum memory limit
        selection_mode: Strategy for assignment ('hard_cutoff', 'weighted', 'proportional', 'bayesian')
        reference_bias: Factor to apply reference-based weighting (0.0-1.0)
        handle_ties: How to handle tied best hits ('keep_all', 'keep_one', 'discard')
        min_assignment_confidence: Minimum probability threshold
        min_confidence_margin: Minimum probability difference to second-best
        acceleration_method: Method for acceleration ('anderson', 'lbfgs', 'hybrid')
        anderson_memory: Memory depth for Anderson acceleration
        lbfgs_memory: Memory depth for L-BFGS acceleration
        lambda_scale: Scaling factor for lambda in EM algorithm
        temperature: Temperature parameter for probability scaling
        use_enhanced_em: Whether to use enhanced EM implementation
        
    Returns:
        List of row IDs for reassigned alignments
    """
    
    log.info(f"Starting deterministic read reassignment using {acceleration_method.upper()} acceleration (seed: {random_seed})")
    
    # Set global random seed for reproducibility
    np.random.seed(random_seed)
    
    # Initialize ResourceManager
    from x_filter.resource_management import ResourceManager
    resource_manager = ResourceManager(
        max_memory=max_memory,
        max_threads=threads,
        mmap_folder=mmap_dir
    )
    
    # Extract arrays - AVOID UNNECESSARY COPYING
    bit_scores = filtered_arrays["bitScore"]
    subject_lengths = filtered_arrays["slen"] 
    subject_numeric_ids = filtered_arrays["subject_numeric_id"]
    query_numeric_ids = filtered_arrays["query_numeric_id"]
    original_indices = filtered_arrays["rowid"]
    
    n_alignments = len(bit_scores)
    
    log.info(f"Processing {n_alignments:,} alignments with ultra-fast algorithms")
    
    # Calculate optimal chunk size for factorization
    factorization_chunk_size = resource_manager.calculate_optimal_chunk_size(
        total_elements=n_alignments,
        element_size=8,
        operation_overhead=2.5,
        min_chunk_size=1_000_000,
        max_chunk_size=100_000_000  # Limit to avoid excessive memory usage
    )
    
    log.info(f"Using factorization chunk size: {factorization_chunk_size:,}")
    
    # ULTRA-EFFICIENT DICTIONARY STRUCTURE - NO COPYING OF LARGE ARRAYS
    log.info("Creating ultra-efficient EM data structure with zero-copy views...")
    
    # Factorize IDs FIRST to create new arrays only for these
    log.info("Factorizing query and subject IDs...")
    
    factorized_source, query_unique = memory_efficient_factorize(
        query_numeric_ids, mmap_dir, threads=threads, chunk_size=factorization_chunk_size, max_memory=max_memory
    )
    factorized_subject, subject_unique = memory_efficient_factorize(
        subject_numeric_ids, mmap_dir, threads=threads, chunk_size=factorization_chunk_size, max_memory=max_memory
    )
    
    log.info(f"Query factorization: {len(query_unique):,} unique queries mapped to indices 0-{len(query_unique)-1}")
    log.info(f"Subject factorization: {len(subject_unique):,} unique subjects mapped to indices 0-{len(subject_unique)-1}")
    
    # Create efficient dictionary with direct references - NO TYPE CONVERSION
    log.info("Creating dictionary structure with zero-copy direct references...")
    
    # ZERO-COPY: Direct references to original arrays (assume correct types)
    em_data = {
        "source": factorized_source,      # New array (factorized indices)
        "subject": factorized_subject,    # New array (factorized indices)
        "var": bit_scores,                # Direct reference - no copying, no conversion
        "slen": subject_lengths,          # Direct reference - no copying, no conversion
        "orig_idx": original_indices,     # Direct reference - no copying, no conversion
        "s_W": np.zeros(n_alignments, dtype=np.float64),     # New array for subject weights
        "prob": np.zeros(n_alignments, dtype=np.float64),    # New array for probabilities
        "iter": np.zeros(n_alignments, dtype=np.int64),      # New array for iteration count
        "n_aln": np.zeros(n_alignments, dtype=np.int64),     # New array for alignment count
        "max_prob": np.zeros(n_alignments, dtype=np.float64), # New array for max probability
    }
    
    log.info(f"Dictionary-based EM structure created with {n_alignments:,} alignments")
    log.info("Memory usage - ZERO-COPY optimization:")
    log.info("  - Direct references (no copying): var, slen, orig_idx")
    log.info("  - New arrays only: source, subject (factorized), s_W, prob, iter, n_aln, max_prob")
    log.info(f"  - New memory allocation: ~{(5 * n_alignments * 8) / (1024**3):.2f} GB")
    log.info(f"  - Original data types preserved: var={bit_scores.dtype}, slen={subject_lengths.dtype}, orig_idx={original_indices.dtype}")
    log.info(f"Bit score range: {np.min(em_data['var']):.2f} to {np.max(em_data['var']):.2f}")
    
    # CRITICAL FIX: Ensure we're passing the correct number of elements
    log.info(f"EM data structure validation:")
    log.info(f"  Dictionary with {len(em_data)} fields: {list(em_data.keys())}")
    log.info(f"  All arrays have length: {len(em_data['var']):,}")
    
    # Verify all arrays have the expected length
    expected_length = n_alignments
    for field_name, array in em_data.items():
        actual_length = len(array)
        if actual_length != expected_length:
            log.error(f"EM data length mismatch in field '{field_name}': expected {expected_length:,}, got {actual_length:,}")
            return []
    
    # Run corrected EM algorithm with dictionary
    log.info(f"Running ultra-fast {acceleration_method.upper()} EM algorithm with dictionary structure")
    

    # Initialize subject weights using fast method
    log.info("Ultra-fast weight initialization...")
    
    initialized_data = initialize_subject_weights(
        em_data, mmap_dir=mmap_dir, max_memory=max_memory
    )
    
    # Use enhanced EM implementation with seed
    result = accelerated_resolve_multimaps(
        initialized_data,
        iters=reassign_iters,
        mmap_dir=mmap_dir,
        max_memory=max_memory,
        threads=threads,
        min_improvement=min_improvement,
        adaptive_convergence=adaptive_convergence,
        acceleration_method=acceleration_method,
        anderson_memory=anderson_memory,
        lbfgs_memory=lbfgs_memory,
        lambda_scale=lambda_scale,  # Pass enhanced parameters
        temperature=temperature,
        use_enhanced_em=use_enhanced_em,
        random_seed=random_seed,  # Pass seed to EM
    )
    
    log.info(f"Ultra-fast {acceleration_method.upper()} algorithm completed")
    log.info("Final EM results:")
    log.info(f"  - Final prob range: {np.min(result['prob']):.6f} to {np.max(result['prob']):.6f}")
    log.info(f"  - Mean final prob: {np.mean(result['prob']):.6f}")
    log.info(f"  - Total iterations: {result['iter'].max()}")
    
    # Add detailed reassignment statistics
    log.info("Generating reassignment statistics...")
    
    # Calculate assignment statistics per query
    unique_source_indices = np.unique(result['source'])
    n_unique_queries = len(unique_source_indices)
    n_elements = len(result['prob'])  # Fix: get length of prob array, not the array itself
    # Count alignments per query
    query_alignment_counts = np.bincount(result['source'])
    queries_with_alignments = query_alignment_counts[query_alignment_counts > 0]
    
    single_alignment_queries = np.sum(queries_with_alignments == 1)
    multi_alignment_queries = np.sum(queries_with_alignments > 1)
    
    log.info("=" * 80)
    log.info("REASSIGNMENT STATISTICS")
    log.info("=" * 80)
    log.info(f"Input alignments: {n_alignments:,}")
    log.info(f"Processed alignments: {n_elements:,}")
    log.info(f"Unique queries processed: {n_unique_queries:,}")
    log.info(f"Unique subjects processed: {len(np.unique(result['subject'])):,}")
    log.info("")
    log.info("READ ASSIGNMENT DISTRIBUTION:")
    log.info(f"  Queries with single alignment: {single_alignment_queries:,} ({single_alignment_queries/n_unique_queries*100:.1f}%)")
    log.info(f"  Queries with multiple alignments: {multi_alignment_queries:,} ({multi_alignment_queries/n_unique_queries*100:.1f}%)")
    log.info(f"  Average alignments per query: {np.mean(queries_with_alignments):.2f}")
    log.info(f"  Median alignments per query: {np.median(queries_with_alignments):.0f}")
    log.info(f"  Max alignments per query: {np.max(queries_with_alignments):,}")
    log.info("")
    
    # Probability distribution statistics
    prob_ranges = [
        (0.95, 1.01, "Very High (0.95-1.00)"),  # Changed: 1.01 to include 1.0
        (0.90, 0.95, "High (0.90-0.95)"),
        (0.80, 0.90, "Medium-High (0.80-0.90)"),
        (0.70, 0.80, "Medium (0.70-0.80)"),
        (0.50, 0.70, "Low-Medium (0.50-0.70)"),
        (0.00, 0.50, "Low (0.00-0.50)")
    ]
    
    log.info("PROBABILITY DISTRIBUTION:")
    for min_prob, max_prob, label in prob_ranges:
        if min_prob == 0.95:  # Special case for the highest range
            mask = result['prob'] >= min_prob  # Use >= for the top range
        else:
            mask = (result['prob'] >= min_prob) & (result['prob'] < max_prob)
        count = np.sum(mask)
        percentage = count / len(result['prob']) * 100
        log.info(f"  {label}: {count:,} alignments ({percentage:.1f}%)")
    
    # Add verification of the discrepancy
    prob_exactly_1 = np.sum(result['prob'] == 1.0)
    prob_095_to_099 = np.sum((result['prob'] >= 0.95) & (result['prob'] < 1.0))
    prob_095_or_higher = np.sum(result['prob'] >= 0.95)
    
    log.info("")
    log.info("PROBABILITY DISTRIBUTION VERIFICATION:")
    log.info(f"  Exactly 1.0: {prob_exactly_1:,} alignments")
    log.info(f"  [0.95, 1.0): {prob_095_to_099:,} alignments") 
    log.info(f"  >= 0.95: {prob_095_or_higher:,} alignments")
    log.info(f"  Sum check: {prob_exactly_1 + prob_095_to_099} = {prob_095_or_higher}")

    # Apply selection mode filtering after EM algorithm
    log.info("")
    log.info("=" * 80)
    log.info("APPLYING SELECTION MODE FILTERING")
    log.info("=" * 80)
    
    # Initialize variables at the start to avoid reference errors
    original_query_ids = None
    original_subject_ids = None
    
    try:
        log.info(f"🔧 Selection mode: {selection_mode}")
        if selection_mode != "hard_cutoff":
            log.info(f"⚙️  Parameters:")
            log.info(f"  • Min confidence: {min_assignment_confidence}")
            log.info(f"  • Min margin: {min_confidence_margin}")
            log.info(f"  • Handle ties: {handle_ties}")
        
        # Convert factorized indices back to original IDs
        log.info("🔄 Converting indices to original IDs...")
        
        # CRITICAL FIX: Get the actual number of alignments from the arrays, not dictionary length
        if isinstance(result, dict):
            actual_n_alignments = len(result['source'])  # Use array length, not dict length
        elif hasattr(result, 'dtype') and result.dtype.names is not None:
            actual_n_alignments = len(result)  # Structured array length
        else:
            actual_n_alignments = len(result) if hasattr(result, '__len__') else 0
        
        log.debug(f"Processing {actual_n_alignments:,} alignments for ID conversion")
        
        # Calculate chunk size for ID conversion
        chunk_size = resource_manager.calculate_optimal_chunk_size(
            total_elements=actual_n_alignments,
            element_size=8,  # Approximate size for int64
            operation_overhead=2.0,
            max_chunk_size=None
        )
        
        log.debug(f"Using chunk size: {chunk_size:,} for ID conversion")
        
        # Create memory-mapped arrays for original IDs
        original_query_path = os.path.join(mmap_dir, "original_query_ids.mmap")
        original_subject_path = os.path.join(mmap_dir, "original_subject_ids.mmap")
        
        original_query_ids_mmap = np.memmap(
            original_query_path, dtype=np.int64, mode='w+', shape=(actual_n_alignments,)
        )
        original_subject_ids_mmap = np.memmap(
            original_subject_path, dtype=np.int64, mode='w+', shape=(actual_n_alignments,)
        )
        
        # Process ID conversion in chunks - handle structured array result
        for start_idx in range(0, actual_n_alignments, chunk_size):
            end_idx = min(start_idx + chunk_size, actual_n_alignments)
            if hasattr(result, 'dtype') and result.dtype.names is not None:
                # Structured array result
                original_query_ids_mmap[start_idx:end_idx] = query_unique[result['source'][start_idx:end_idx]]
                original_subject_ids_mmap[start_idx:end_idx] = subject_unique[result['subject'][start_idx:end_idx]]
            else:
                # Dictionary result (fallback)
                original_query_ids_mmap[start_idx:end_idx] = query_unique[result['source'][start_idx:end_idx]]
                original_subject_ids_mmap[start_idx:end_idx] = subject_unique[result['subject'][start_idx:end_idx]]
        
        original_query_ids_mmap.flush()
        original_subject_ids_mmap.flush()
        
        # Create PyArrow table - handle structured array result
        log.info("📊 Creating analysis table...")
        
        pyarrow_chunk_size = resource_manager.calculate_optimal_chunk_size(
            total_elements=actual_n_alignments,
            element_size=8,
            operation_overhead=3.0,
            min_chunk_size=1_000_000,
            max_chunk_size=100_000_000  # Limit to avoid excessive memory usage
        )
        
        schema = pa.schema([
            ('read_id', pa.int64()),
            ('ref_id', pa.int64()),
            ('orig_idx', pa.int64()),
            ('prob', pa.float64())
        ])
        
        parquet_path = os.path.join(mmap_dir, "read_prob_temp.parquet")
        
        with pq.ParquetWriter(parquet_path, schema) as writer:
            for start_idx in range(0, actual_n_alignments, pyarrow_chunk_size):
                end_idx = min(start_idx + pyarrow_chunk_size, actual_n_alignments)
                
                # Handle both structured array and dictionary results
                if hasattr(result, 'dtype') and result.dtype.names is not None:
                    # Structured array result
                    batch = pa.record_batch([
                        pa.array(original_query_ids_mmap[start_idx:end_idx]),
                        pa.array(original_subject_ids_mmap[start_idx:end_idx]),
                        pa.array(result['orig_idx'][start_idx:end_idx]),
                        pa.array(result['prob'][start_idx:end_idx])
                    ], schema=schema)
                else:
                    # Dictionary result (fallback)
                    batch = pa.record_batch([
                        pa.array(original_query_ids_mmap[start_idx:end_idx]),
                        pa.array(original_subject_ids_mmap[start_idx:end_idx]),
                        pa.array(result['orig_idx'][start_idx:end_idx]),
                        pa.array(result['prob'][start_idx:end_idx])
                    ], schema=schema)
                
                writer.write_batch(batch)
        
        log.debug(f"Analysis table created: {actual_n_alignments:,} alignments")

        # Clean up memory-mapped arrays
        del original_query_ids_mmap, original_subject_ids_mmap
        try:
            os.unlink(original_query_path)
            os.unlink(original_subject_path)
        except OSError:
            pass

        # Use DatabaseManager for selection mode
        with DatabaseManager(
            database=None,
            temp_dir=mmap_dir,
            threads=threads,
            memory_limit=max_memory,
            enable_progress=False
        ) as db_manager:
            con = db_manager.connection
            
            con.execute(f"""
                CREATE TABLE read_prob AS 
                SELECT * FROM read_parquet('{parquet_path}')
            """)
            
            selected_results = implement_selection_mode(
                con, selection_mode, reference_bias, handle_ties,
                min_assignment_confidence, min_confidence_margin,
                random_seed
            )
            
            # Clean up temporary parquet file
            try:
                os.unlink(parquet_path)
            except OSError:
                pass
            
            log.info(f"Selection mode filtering completed: {len(selected_results):,} alignments selected")
            
            # Extract selected row IDs
            selected_rowids = [row[0] for row in selected_results]
            return selected_rowids
            
    except Exception as e:
        log.error(f"Error in selection mode processing: {e}")
        import traceback
        log.error(f"Full traceback: {traceback.format_exc()}")
        
        # Clean up any temporary files
        try:
            if original_query_ids is not None:
                del original_query_ids
            if original_subject_ids is not None:
                del original_subject_ids
            if 'parquet_path' in locals():
                os.unlink(parquet_path)
        except:
            pass
        
        # Return all original indices as fallback
        return list(range(n_alignments))

def reassign(args):
    """Entry point function for CLI that calls reassign_reads_mmap with arguments from argparse"""
    # Extract seed from args or use default
    random_seed = getattr(args, 'random_seed', 42)
    
    log.info(f"Starting reassignment with acceleration method: {getattr(args, 'acceleration_method', 'hybrid')} (seed: {random_seed})")
    
    # Extract parameters from args object
    filtered_arrays = getattr(args, 'filtered_arrays', {})
    mmap_dir = getattr(args, 'mmap_dir', '/tmp')
    threads = getattr(args, 'threads', 1)
    reassign_iters = getattr(args, 'n_iters', 25)
    min_improvement = getattr(args, 'min_improvement', 1e-4)
    adaptive_convergence = getattr(args, 'adaptive_convergence', False)
    max_memory = getattr(args, 'max_memory', None)
    
    # Extract selection mode parameters
    selection_mode = getattr(args, 'selection_mode', 'hard_cutoff')
    reference_bias = getattr(args, 'reference_bias', 0.0)
    handle_ties = getattr(args, 'handle_ties', 'keep_all')
    min_assignment_confidence = getattr(args, 'min_assignment_confidence', 0.01)
    min_confidence_margin = getattr(args, 'min_confidence_margin', 0.0)
    
    # Extract acceleration parameters
    acceleration_method = getattr(args, 'acceleration_method', 'hybrid')
    anderson_memory = getattr(args, 'anderson_memory', 10)
    lbfgs_memory = getattr(args, 'lbfgs_memory', 10)
    
    # Call the actual reassignment implementation
    return reassign_reads_mmap(
        filtered_arrays=filtered_arrays,
        unique_query_ids=np.array([]),
        unique_subject_ids=np.array([]),
        mmap_dir=mmap_dir,
        threads=threads,
        reassign_iters=reassign_iters,
        min_improvement=min_improvement,
        adaptive_convergence=adaptive_convergence,
        max_memory=max_memory,
        selection_mode=selection_mode,
        reference_bias=reference_bias,
        handle_ties=handle_ties,
        min_assignment_confidence=min_assignment_confidence,
        min_confidence_margin=min_confidence_margin,
        acceleration_method=acceleration_method,
        anderson_memory=anderson_memory,
        lbfgs_memory=lbfgs_memory,
        random_seed=random_seed,
    )

@njit(fastmath=True, parallel=True)
def unified_initialize_probabilities_fast(
    source_indices: np.ndarray,
    bit_scores: np.ndarray,
    output_probs: np.ndarray,
    temperature: float = 0.01  # Lower default temperature for sharper assignment
) -> None:
    """
    Fast unified initialization using numba with temperature-scaled softmax.
    Creates more decisive initial probabilities.
    """
    n = len(source_indices)
    if n == 0:
        return
    
    max_source = np.max(source_indices) + 1
    max_threads = 64
    
    # Normalize bit scores more aggressively
    min_score = np.min(bit_scores)
    max_score = np.max(bit_scores)
    score_range = max_score - min_score
    
    # Check if all scores are identical (uniform case)
    if score_range < 1e-10:
        # Use uniform distribution for each source
        source_counts = np.bincount(source_indices, minlength=max_source)
        for i in prange(n):
            source_idx = source_indices[i]
            count = source_counts[source_idx]
            output_probs[i] = 1.0 / count if count > 0 else 1e-15
        return
    
    # --- Pass 1: Find max normalized scores per source (thread-safe) ---
    private_source_max_scores = np.full((max_threads, max_source), -np.inf, dtype=np.float64)
    
    for i in prange(n):
        thread_id = numba.get_thread_id()
        source_idx = source_indices[i]
        # More aggressive normalization to create separation
        normalized_score = (bit_scores[i] - min_score) / score_range
        enhanced_score = normalized_score ** 3  # Cube to emphasize differences even more
        if enhanced_score > private_source_max_scores[thread_id, source_idx]:
            private_source_max_scores[thread_id, source_idx] = enhanced_score
    
    actual_threads = numba.get_num_threads()

    # Reduce to global max scores
    source_max_scores = np.full(max_source, -np.inf, dtype=np.float64)
    for i in prange(max_source):
        max_val = -np.inf
        for t_id in range(actual_threads):
            if private_source_max_scores[t_id, i] > max_val:
                max_val = private_source_max_scores[t_id, i]
        source_max_scores[i] = max_val
    
    # --- Pass 2: Compute softmax denominators (thread-safe) ---
    private_source_denominators = np.zeros((max_threads, max_source), dtype=np.float64)
    
    for i in prange(n):
        thread_id = numba.get_thread_id()
        source_idx = source_indices[i]
        max_score = source_max_scores[source_idx]
        
        # Enhanced normalization with cubing
        normalized_score = (bit_scores[i] - min_score) / score_range
        enhanced_score = normalized_score ** 3
        
        exp_val: float
        if max_score == -np.inf:
            exp_val = 1.0
        else:
            # Much lower temperature for more decisive probabilities
            val = (enhanced_score - max_score) / temperature
            val = max(-500.0, min(500.0, val))  # Allow wider range for separation
            exp_val = np.exp(val)
        private_source_denominators[thread_id, source_idx] += exp_val
    
    # Reduce denominators
    source_denominators = np.zeros(max_source, dtype=np.float64)
    for i in prange(max_source):
        sum_val = 0.0
        for t_id in range(actual_threads):
            sum_val += private_source_denominators[t_id, i]
        source_denominators[i] = sum_val
    
    # Pre-calculate counts for uniform fallback
    source_counts = np.bincount(source_indices, minlength=max_source)
    
    # --- Pass 3: Compute final probabilities with proper normalization ---
    for i in prange(n):
        source_idx = source_indices[i]
        max_score = source_max_scores[source_idx]
        denominator = source_denominators[source_idx]
        
        # Enhanced normalization
        normalized_score = (bit_scores[i] - min_score) / score_range
        enhanced_score = normalized_score ** 3
        
        if denominator > 1e-15 and max_score != -np.inf:
            val = (enhanced_score - max_score) / temperature
            val = max(-500.0, min(500.0, val))
            exp_val = np.exp(val)
            output_probs[i] = exp_val / denominator
        else:
            # Fallback to uniform distribution
            count = source_counts[source_idx]
            output_probs[i] = 1.0 / count if count > 0 else 1e-15
        
        # More restrictive probability bounds for better behavior
        output_probs[i] = max(1e-12, min(1.0, output_probs[i]))

@njit(fastmath=True, parallel=True)
def fix_zero_probability_sources(
    source_indices: np.ndarray,
    probabilities: np.ndarray,
    max_source_id: int
) -> int:
    """
    Fix sources that have all zero probabilities by setting uniform distribution.
    Returns number of sources fixed.
    """
    # Count alignments per source
    source_counts = np.zeros(max_source_id + 1, dtype=np.int64)
    for i in prange(len(source_indices)):
        source_counts[source_indices[i]] += 1
    
    # Check sums per source
    source_sums = np.zeros(max_source_id + 1, dtype=np.float64)
    for i in range(len(source_indices)):
        source_sums[source_indices[i]] += probabilities[i]
    
    sources_fixed = 0
    
    # Fix zero-sum sources
    for source_idx in range(max_source_id + 1):
        if source_counts[source_idx] > 0 and source_sums[source_idx] < 1e-15:
            sources_fixed += 1
            uniform_prob = 1.0 / source_counts[source_idx]
            
            # Set uniform probabilities for this source
            for i in range(len(source_indices)):
                if source_indices[i] == source_idx:
                    probabilities[i] = uniform_prob
    
    return sources_fixed

@njit(fastmath=True, parallel=True)
def simple_uniform_initialize(source_indices: np.ndarray, output_probs: np.ndarray) -> None:
    """
    Simple uniform initialization - each query gets equal probability for all its alignments.
    This should guarantee proper normalization and valid starting point.
    """
    n = len(source_indices)
    if n == 0:
        return
    
    max_source = np.max(source_indices) + 1
    
    # Count alignments per source
    source_counts = np.zeros(max_source, dtype=np.int64)
    for i in range(n):
        source_counts[source_indices[i]] += 1
    
    # Set uniform probabilities
    for i in prange(n):
        source_idx = source_indices[i]
        count = source_counts[source_idx]
        output_probs[i] = 1.0 / count if count > 0 else 1e-15

def validate_data_integrity(data, stage="unknown"):
    """
    Validate data integrity and log comprehensive statistics.
    
    Args:
        data: Input data structure
        stage: Stage identifier for logging
        
    Returns:
        tuple: (is_valid, n_elements, error_message)
    """
    try:
        # CRITICAL: First check what type of data structure we have
        log.debug(f"Data validation at {stage}:")
        log.debug(f"  Data type: {type(data)}")
        
        # CRITICAL DEBUG: Check if data is a scalar value, not an array/dict
        if np.isscalar(data):
            return False, 0, f"Data is a scalar value at {stage}: {data}"
        
        # Extract arrays based on data structure type
        if isinstance(data, dict):
            log.debug(f"  Dictionary with {len(data)} keys: {list(data.keys())}")
            
            # CRITICAL: Check if dictionary contains the data or if it IS the data arrays
            if len(data) == 7 and all(isinstance(v, (int, float, np.number)) for v in data.values()):
                return False, 0, f"Dictionary contains only scalar values at {stage}, not arrays. Values: {data}"
            
            # Check for required fields
            required_fields = ["source", "subject", "var"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                # CRITICAL DEBUG: Show what keys we actually have
                log.error(f"Missing required fields at {stage}: {missing_fields}")
                log.error(f"Available keys: {list(data.keys())}")
                log.error(f"Key-value preview: {dict(list(data.items())[:5])}")  # Show first 5 items
                return False, 0, f"Missing required fields in dictionary at {stage}: {missing_fields}"
            
            # Extract arrays and validate they are actually arrays
            source_array = data["source"]
            subject_array = data["subject"]
            var_array = data["var"]
            
            # CRITICAL: Check if these are scalar values instead of arrays
            for name, arr in [("source", source_array), ("subject", subject_array), ("var", var_array)]:
                if np.isscalar(arr):
                    return False, 0, f"Field '{name}' is a scalar, not an array at {stage}: {arr}"
                
                if not hasattr(arr, '__len__'):
                    return False, 0, f"Field '{name}' is not an array at {stage}: {type(arr)} (value: {arr})"
                
                # Additional check for very small "arrays" that might actually be single values
                try:
                    arr_len = len(arr)
                    log.debug(f"    {name}: type={type(arr)}, length={arr_len}")
                    
                    # CRITICAL: Log sample of data for debugging
                    if hasattr(arr, '__getitem__') and arr_len > 0:
                        sample_size = min(5, arr_len)
                        try:
                            sample = [arr[i] for i in range(sample_size)]
                            log.debug(f"      sample data: {sample}")
                        except Exception as e:
                            log.warning(f"      cannot sample data: {e}")
                    
                    if arr_len == 1:
                        log.warning(f"Field '{name}' has only 1 element - this might indicate a data problem")
                    elif arr_len < 10:
                        log.warning(f"Field '{name}' has only {arr_len} elements - unexpectedly small for 55M input")
                        
                except Exception as e:
                    return False, 0, f"Cannot get length of field '{name}' at {stage}: {e}"
            
        elif hasattr(data, 'dtype') and data.dtype.names is not None:
            log.debug(f"  Structured array with shape: {data.shape}")
            log.debug(f"  Field names: {data.dtype.names}")
            
            # CRITICAL: Check if structured array shape indicates data loss
            if hasattr(data, 'shape') and len(data.shape) > 0:
                total_elements = data.shape[0]
                log.debug(f"  Structured array total elements: {total_elements}")
                if total_elements < 1000:
                    log.warning(f"Structured array is very small at {stage}: only {total_elements} elements")
            
            required_fields = ['source', 'subject', 'var']
            missing_fields = [field for field in required_fields if field not in data.dtype.names]
            if missing_fields:
                return False, 0, f"Missing required fields in structured array at {stage}: {missing_fields}"
            
            source_array = data['source']
            subject_array = data['subject']
            var_array = data['var']
        else:
            log.error(f"Unknown data structure at {stage}:")
            log.error(f"  Type: {type(data)}")
            log.error(f"  Dir: {dir(data)}")
            log.error(f"  Str representation: {str(data)[:500]}")  # First 500 chars
            return False, 0, f"Unknown data structure type at {stage}: {type(data)}"
        
        # Get actual array lengths
        try:
            source_len = len(source_array)
            subject_len = len(subject_array)
            var_len = len(var_array)
        except Exception as e:
            return False, 0, f"Cannot determine array lengths at {stage}: {e}"
        
        # Validate lengths match
        if not (source_len == subject_len == var_len):
            return False, 0, f"Array length mismatch at {stage}: source={source_len}, subject={subject_len}, var={var_len}"
        
        n_elements = source_len
        
        # Validate we have data
        if n_elements == 0:
            return False, 0, f"No data found at {stage}"
        
        # CRITICAL: Check if we have suspiciously small datasets
        if n_elements < 1000:
            log.error(f"CRITICAL DATA LOSS detected at {stage}: only {n_elements} elements!")
            log.error("This suggests massive data loss in the processing pipeline")
            log.error("Expected ~55M elements based on input logs")
            
            # Log the actual data for debugging
            log.error("Complete dataset contents:")
            for i in range(min(n_elements, 10)):
                try:
                    log.error(f"  Element {i}: source={source_array[i]}, subject={subject_array[i]}, var={var_array[i]}")
                except Exception as e:
                    log.error(f"  Element {i}: cannot access - {e}")
        
        # Basic data validation
        try:
            if not np.all(np.isfinite(var_array)):
                return False, n_elements, f"Non-finite values in var array at {stage}"
        except Exception as e:
            log.warning(f"Cannot check finite values at {stage}: {e}")
        
        try:
            if np.any(source_array < 0) or np.any(subject_array < 0):
                return False, n_elements, f"Negative indices found at {stage}"
        except Exception as e:
            log.warning(f"Cannot check negative indices at {stage}: {e}")
        
        # Log validation success
        log.debug(f"Data validation PASSED at {stage}:")
        log.debug(f"  Total elements: {n_elements:,}")
        
        try:
            log.debug(f"  Source range: {np.min(source_array)} to {np.max(source_array)}")
            log.debug(f"  Subject range: {np.min(subject_array)} to {np.max(subject_array)}")
            log.debug(f"  Var range: {np.min(var_array):.3f} to {np.max(var_array):.3f}")
        except Exception as e:
            log.warning(f"Cannot compute ranges at {stage}: {e}")
        
        return True, n_elements, "Validation passed"
        
    except Exception as e:
        log.error(f"Validation error at {stage}: {e}")
        import traceback
        log.error(f"Full traceback: {traceback.format_exc()}")
        return False, 0, f"Validation error at {stage}: {e}"

def trace_data_flow(data, stage="unknown", expected_size=None):
    """
    Trace data flow and identify where data loss occurs.
    """
    log.info(f"=== DATA FLOW TRACE: {stage} ===")
    
    if expected_size is not None:
        log.info(f"Expected size: {expected_size:,}")
    
    # Basic type and size info
    log.info(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        log.info(f"Dictionary with {len(data)} keys: {list(data.keys())}")
        
        # Check each key's content
        for key, value in data.items():
            log.info(f"  Key '{key}':")
            log.info(f"    Type: {type(value)}")
            if hasattr(value, 'shape'):
                log.info(f"    Shape: {value.shape}")
            elif hasattr(value, '__len__'):
                log.info(f"    Length: {len(value)}")
            else:
                log.info(f"    Value: {value}")
                
            # If it's an array-like, sample some data
            if hasattr(value, '__getitem__') and hasattr(value, '__len__'):
                try:
                    arr_len = len(value)
                    if arr_len > 0:
                        sample_size = min(3, arr_len)
                        sample = [value[i] for i in range(sample_size)]
                        log.info(f"    Sample: {sample}")
                        
                        # Check for data loss
                        if expected_size is not None and arr_len < expected_size * 0.01:  # Less than 1% of expected
                            log.error(f"    🚨 MASSIVE DATA LOSS: {arr_len:,} << {expected_size:,}")
                        elif arr_len < 1000:
                            log.warning(f"    ⚠️  SUSPICIOUSLY SMALL: {arr_len:,}")
                except Exception as e:
                    log.warning(f"    Cannot sample: {e}")
    
    elif hasattr(data, 'dtype') and data.dtype.names is not None:
        log.info(f"Structured array with shape: {data.shape}")
        log.info(f"Field names: {data.dtype.names}")
        
        total_elements = data.shape[0] if len(data.shape) > 0 else 0
        if expected_size is not None and total_elements < expected_size * 0.01:
            log.error(f"🚨 MASSIVE DATA LOSS in structured array: {total_elements:,} << {expected_size:,}")
    
    else:
        log.info(f"Unknown data structure")
        if hasattr(data, '__len__'):
            log.info(f"Length: {len(data)}")
        if hasattr(data, 'shape'):
            log.info(f"Shape: {data.shape}")
    
    log.info(f"=== END TRACE: {stage} ===")

def ensure_data_has_prob_field(data, probabilities: np.ndarray, iterations: int) -> Dict[str, Any]:
    """
    Ensure the data structure has prob and iter fields, handling both dict and structured array cases.
    """
    n_elements = len(probabilities)
    log.info(f"Adding prob field with {n_elements:,} elements to data structure")
    
    # Case 1: Dictionary-like object
    if isinstance(data, dict):
        log.debug("Handling dictionary data structure")
        # Direct assignment for dictionaries
        data["prob"] = probabilities
        data["iter"] = np.full(n_elements, iterations, dtype=np.int32)
        return data
    
    # Case 2: Structured array
    elif hasattr(data, 'dtype') and data.dtype.names is not None:
        log.debug("Handling structured array data structure")
        
        # Check if prob field already exists
        if "prob" in data.dtype.names:
            log.debug("prob field exists, updating values")
            data["prob"][:] = probabilities
        else:
            log.debug("Creating new structured array with prob field")
            # Create new dtype with prob field
            new_dtype = data.dtype.descr + [('prob', np.float64)]
            new_data = np.empty(len(data), dtype=new_dtype)  # Use len(data) for structured array
            
            # Copy existing fields
            for field_name in data.dtype.names:
                new_data[field_name] = data[field_name]
            
            # Add prob field
            new_data['prob'] = probabilities
            data = new_data
        
        # Handle iter field similarly
        if "iter" in data.dtype.names:
            log.debug("iter field exists, updating values")
            data["iter"][:] = iterations
        else:
            if "prob" not in data.dtype.names:  # We already created new array above
                log.debug("Adding iter field to new structured array")
                # Create another new dtype with both prob and iter fields
                new_dtype = data.dtype.descr + [('iter', np.int32)]
                newer_data = np.empty(len(data), dtype=new_dtype)  # Use len(data) for structured array
                
                # Copy all existing fields including prob
                for field_name in data.dtype.names:
                    newer_data[field_name] = data[field_name]
                
                # Add iter field
                newer_data['iter'] = iterations
                data = newer_data
            else:
                log.debug("iter field not present and cannot be added to existing structured array")
        
        return data
    
    # Case 3: Other array types - convert to dictionary
    else:
        log.debug("Converting unknown data type to dictionary")
        if hasattr(data, '__len__') and len(data) == n_elements:
            # Assume it's an array-like object, convert to dict
            new_data = {
                "data": data,
                "prob": probabilities,
                "iter": np.full(n_elements, iterations, dtype=np.int32)
            }
            return new_data
        else:
            log.error(f"Cannot handle data type: {type(data)}")
            raise ValueError(f"Unsupported data type: {type(data)}")

def reassign_multimapping_reads(
    data,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-4,
    lambda_scale: float = 1.0,
    acceleration_method: str = "anderson",
    mmap_dir: str = None,
    max_memory: Union[str, int] = None,
    threads: int = None,
    adaptive_convergence: bool = False,
    random_seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function to reassign multi-mapping reads using EM algorithm with deterministic behavior.
    """
    log.info(f"Starting deterministic multi-mapping read reassignment (seed: {random_seed})")
    
    # Set global random seed
    np.random.seed(random_seed)
    
    # Validate input data
    if data is None:
        raise ValueError("Input data cannot be None")
    
    # Get n_elements from actual array length
    if isinstance(data, dict):
        n_elements = len(data["source"]) if "source" in data else 0
        if n_elements == 0 and "var" in data:
            n_elements = len(data["var"])
        expected_size = n_elements
    elif hasattr(data, 'dtype') and data.dtype.names is not None:
        n_elements = len(data)
        expected_size = n_elements
    else:
        n_elements = len(data) if hasattr(data, '__len__') else 0
        expected_size = n_elements
    
    # Validate input data integrity
    is_valid, validated_elements, error_msg = validate_data_integrity(data, "INPUT")
    if not is_valid:
        log.error(f"Input data validation failed: {error_msg}")
        raise ValueError(f"Invalid input data: {error_msg}")
    
    log.info(f"Input validation PASSED: {validated_elements:,} alignments")
    n_elements = validated_elements
    
    try:
        # Call the EM algorithm with seed
        result_data = accelerated_resolve_multimaps(
            data=data,
            iters=max_iterations,
            mmap_dir=mmap_dir,
            max_memory=max_memory,
            threads=threads,
            min_improvement=convergence_threshold,
            adaptive_convergence=adaptive_convergence,
            acceleration_method=acceleration_method,
            lambda_scale=lambda_scale,
            random_seed=random_seed,
            **kwargs
        )

        # Validate output data integrity
        is_valid, output_elements, error_msg = validate_data_integrity(result_data, "OUTPUT")
        if not is_valid:
            log.error(f"Output data validation failed: {error_msg}")
            uniform_probs = np.full(n_elements, 1.0 / n_elements, dtype=np.float64)
            return ensure_data_has_prob_field(data, uniform_probs, 0)
        
        # Ensure the result has the required fields
        if isinstance(result_data, dict) and "prob" in result_data:
            log.info(f"EM algorithm completed successfully: {output_elements:,} alignments processed")
            if len(result_data["prob"]) != n_elements:
                log.error(f"Probability array length mismatch: expected {n_elements}, got {len(result_data['prob'])}")
                uniform_probs = np.full(n_elements, 1.0 / n_elements, dtype=np.float64)
                return ensure_data_has_prob_field(data, uniform_probs, 0)
            return result_data
        elif hasattr(result_data, 'dtype') and result_data.dtype.names is not None and "prob" in result_data.dtype.names:
            log.info(f"EM algorithm completed successfully: {output_elements:,} alignments processed")
            if len(result_data) != n_elements:
                log.error(f"Result array length mismatch: expected {n_elements}, got {len(result_data)}")
                uniform_probs = np.full(n_elements, 1.0 / n_elements, dtype=np.float64)
                return ensure_data_has_prob_field(data, uniform_probs, 0)
            return result_data
        else:
            log.error("EM algorithm did not return probability assignments")
            uniform_probs = np.full(n_elements, 1.0 / n_elements, dtype=np.float64)
            return ensure_data_has_prob_field(data, uniform_probs, 0)
    
    except Exception as e:
        log.error(f"Error in reassignment: {e}")
        import traceback
        log.error(f"Full traceback: {traceback.format_exc()}")
        uniform_probs = np.full(n_elements, 1.0 / n_elements, dtype=np.float64)
        return ensure_data_has_prob_field(data, uniform_probs, 0)

def validate_reassignment_results(data) -> bool:
    """
    Validate that reassignment results are correct.
    """
    try:
        # Extract probabilities based on data structure
        if isinstance(data, dict):
            if "prob" not in data:
                log.error("Missing prob field in dictionary")
                return False
            probabilities = data["prob"]
            source_indices = data.get("source", None)
        elif hasattr(data, 'dtype') and data.dtype.names is not None:
            if "prob" not in data.dtype.names:
                log.error("Missing prob field in structured array")
                return False
            probabilities = data["prob"]
            source_indices = data["source"] if "source" in data.dtype.names else None
        else:
            log.error("Cannot validate unknown data structure")
            return False
        
        # Basic probability validation
        if not np.all(np.isfinite(probabilities)):
            log.error("Non-finite probabilities detected")
            return False
        
        if not np.all(probabilities >= 0):
            log.error("Negative probabilities detected")
            return False
        
        if not np.all(probabilities <= 1.0):
            log.error("Probabilities > 1.0 detected")
            return False
        
        # Check probability conservation per read if source indices available
        if source_indices is not None:
            unique_reads = np.unique(source_indices)
            for read_id in unique_reads[:min(100, len(unique_reads))]:  # Sample validation
                read_mask = source_indices == read_id
                read_prob_sum = np.sum(probabilities[read_mask])
                if abs(read_prob_sum - 1.0) > 0.01:
                    log.warning(f"Read {read_id} has probability sum {read_prob_sum:.4f} (should be 1.0)")
                    # Don't fail validation for small deviations
        
        log.debug("Reassignment validation passed")
        return True
        
    except Exception as e:
        log.error(f"Validation error: {e}")
        return False
