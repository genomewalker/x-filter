"""
Statistics generation functions for reassignment results.
"""
import numpy as np
from typing import Dict
import duckdb
from x_filter.logging_setup import get_logger
from x_filter.core_processing import initialize_mmap_array

log = get_logger()


def generate_assignment_statistics(con: duckdb.DuckDBPyConnection):
    """Generate statistics about alignments, reads and references after reassignment."""
    log.info("Generating detailed statistics about reassignment results...")
    
    # First check if tables exist
    try:
        table_check = con.execute("""
        SELECT 
            COUNT(*) AS count, 
            'reassigned_alignments' AS table_name 
        FROM information_schema.tables 
        WHERE table_name='reassigned_alignments'
        """).fetchone()
        
        if table_check[0] == 0:
            log.warning("reassigned_alignments table does not exist. Skipping statistics generation.")
            return None
    except Exception as e:
        log.warning(f"Error checking for reassigned tables: {e}")
        return None
    
    # Simplified approach - get statistics separately to avoid complex UNION issues
    try:
        # Get read mapping statistics
        read_mapping_stats = con.execute("""
            WITH read_stats AS (
                SELECT 
                    read_id,
                    COUNT(*) as aln_count
                FROM reassigned_alignments
                GROUP BY read_id
            )
            SELECT
                COUNT(*) as total_reads,
                SUM(CASE WHEN aln_count = 1 THEN 1 ELSE 0 END) as unique_reads,
                SUM(CASE WHEN aln_count > 1 THEN 1 ELSE 0 END) as multimapping_reads,
                AVG(aln_count) as mean_alns_per_read,
                MEDIAN(aln_count) as median_alns_per_read,
                MIN(aln_count) as min_alns_per_read,
                MAX(aln_count) as max_alns_per_read
            FROM read_stats
        """).fetchone()
        
        # Get reference statistics
        ref_stats = con.execute("""
            SELECT
                COUNT(*) as total_refs,
                AVG(aln_count) as mean_alns_per_ref,
                MEDIAN(aln_count) as median_alns_per_ref,
                MIN(aln_count) as min_alns_per_ref,
                MAX(aln_count) as max_alns_per_ref
            FROM reassigned_refs
        """).fetchone()
        
        # Get total alignments
        total_alignments = con.execute("""
            SELECT COUNT(*) as total_alignments FROM reassigned_alignments
        """).fetchone()[0]
        
        # Extract values from the results
        total_reads = read_mapping_stats[0]
        unique_reads = read_mapping_stats[1]
        multimapping_reads = read_mapping_stats[2]
        mean_alns_per_read = read_mapping_stats[3]
        median_alns_per_read = read_mapping_stats[4]
        min_alns_per_read = read_mapping_stats[5]
        max_alns_per_read = read_mapping_stats[6]
        
        total_refs = ref_stats[0]
        mean_alns_per_ref = ref_stats[1]
        median_alns_per_ref = ref_stats[2]
        min_alns_per_ref = ref_stats[3]
        max_alns_per_ref = ref_stats[4]
        
        # Create a rich summary report
        log.info("-" * 80)
        log.info("REASSIGNMENT SUMMARY")
        log.info("-" * 80)
        
        # Print read statistics
        log.info(f"READS:")
        log.info(f"  Total reads with alignments: {total_reads:,}")
        log.info(f"  Unique reads (single alignment): {unique_reads:,} ({unique_reads/total_reads*100:.1f}%)")
        log.info(f"  Multimapping reads: {multimapping_reads:,} ({multimapping_reads/total_reads*100:.1f}%)")
        log.info(f"  Average alignments per read: {mean_alns_per_read:.2f}")
        log.info(f"  Median alignments per read: {median_alns_per_read}")
        log.info(f"  Maximum alignments per read: {max_alns_per_read:,}")
        
        # Print reference statistics
        log.info(f"REFERENCES:")
        log.info(f"  Total references: {total_refs:,}")
        log.info(f"  Average alignments per reference: {mean_alns_per_ref:.2f}")
        log.info(f"  Median alignments per reference: {median_alns_per_ref}")
        log.info(f"  Min alignments per reference: {min_alns_per_ref}")
        log.info(f"  Max alignments per reference: {max_alns_per_ref:,}")
        
        # Print alignment statistics
        log.info(f"ALIGNMENTS:")
        log.info(f"  Total alignments: {total_alignments:,}")
        log.info(f"  Alignments per unique read: 1")
        
        # Calculate average alignments per multimapping read correctly
        if multimapping_reads > 0:
            multimapping_alns = total_alignments - unique_reads
            avg_alns_per_multimapping = multimapping_alns / multimapping_reads
            log.info(f"  Average alignments per multimapping read: {avg_alns_per_multimapping:.2f}")
        else:
            log.info(f"  Average alignments per multimapping read: N/A")
        
    except Exception as e:
        log.warning(f"Error generating detailed statistics: {e}")
        # Fall back to basic statistics
        basic_stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM reassigned_alignments) as total_alignments,
            (SELECT COUNT(*) FROM reassigned_refs) as total_refs,
            (SELECT COUNT(*) FROM reassigned_reads) as total_reads,
            (SELECT COUNT(DISTINCT read_id) FROM reassigned_alignments WHERE 
             read_id IN (SELECT read_id FROM reassigned_alignments GROUP BY read_id HAVING COUNT(*) = 1)) as unique_reads
        """
        basic_stats = con.execute(basic_stats_query).fetchone()
        
        # Use basic stats instead
        total_alignments = basic_stats[0]
        total_refs = basic_stats[1] 
        total_reads = basic_stats[2]
        unique_reads = basic_stats[3]
        multimapping_reads = total_reads - unique_reads
        
        # Create a rich summary report with basic stats
        log.info("-" * 80)
        log.info("REASSIGNMENT SUMMARY (basic statistics)")
        log.info("-" * 80)
        
        # Print read statistics
        log.info(f"READS:")
        log.info(f"  Total reads with alignments: {total_reads:,}")
        log.info(f"  Unique reads (single alignment): {unique_reads:,} ({unique_reads/total_reads*100:.1f}%)")
        log.info(f"  Multimapping reads: {multimapping_reads:,} ({multimapping_reads/total_reads*100:.1f}%)")
        
        # Print reference statistics
        log.info(f"REFERENCES:")
        log.info(f"  Total references: {total_refs:,}")
        
        # Print alignment statistics
        log.info(f"ALIGNMENTS:")
        log.info(f"  Total alignments: {total_alignments:,}")
        
        # Calculate average alignments per multimapping read correctly
        if multimapping_reads > 0:
            multimapping_alns = total_alignments - unique_reads
            avg_alns_per_multimapping = multimapping_alns / multimapping_reads
            log.info(f"  Average alignments per multimapping read: {avg_alns_per_multimapping:.2f}")
        else:
            log.info(f"  Average alignments per multimapping read: N/A")
    
    log.info("-" * 80)

def generate_confidence_statistics(con: duckdb.DuckDBPyConnection):
    """Generate statistics about assignment confidence."""
    log.info("Generating statistics about assignment confidence...")
    
    # Calculate basic statistics for confidence metrics
    basic_confidence_stats = con.execute("""
        SELECT
            COUNT(*) AS total_assignments,
            AVG(primary_prob) AS avg_primary_prob,
            MEDIAN(primary_prob) AS median_primary_prob,
            MIN(primary_prob) AS min_primary_prob,
            MAX(primary_prob) AS max_primary_prob,
            
            AVG(secondary_prob) AS avg_secondary_prob,
            MEDIAN(secondary_prob) AS median_secondary_prob,
            MIN(secondary_prob) AS min_secondary_prob,
            MAX(secondary_prob) AS max_secondary_prob,
            
            AVG(prob_diff) AS avg_prob_diff,
            MEDIAN(prob_diff) AS median_prob_diff,
            MIN(prob_diff) AS min_prob_diff,
            MAX(prob_diff) AS max_prob_diff,
            
            AVG(prob_ratio) AS avg_prob_ratio,
            MEDIAN(prob_ratio) AS median_prob_ratio,
            MIN(prob_ratio) AS min_prob_ratio,
            MAX(prob_ratio) AS max_prob_ratio
        FROM alignment_confidence
    """).fetchone()
    
    log.info("-" * 80)
    log.info("ASSIGNMENT CONFIDENCE STATISTICS")
    log.info("-" * 80)
    log.info(f"Total assignments: {basic_confidence_stats[0]:,}")
    log.info(f"Average primary probability: {basic_confidence_stats[1]:.3f}")
    log.info(f"Median primary probability: {basic_confidence_stats[2]:.3f}")
    log.info(f"Min primary probability: {basic_confidence_stats[3]:.3f}")
    log.info(f"Max primary probability: {basic_confidence_stats[4]:.3f}")
    
    log.info(f"Average secondary probability: {basic_confidence_stats[5]:.3f}")
    log.info(f"Median secondary probability: {basic_confidence_stats[6]:.3f}")
    log.info(f"Min secondary probability: {basic_confidence_stats[7]:.3f}")
    log.info(f"Max secondary probability: {basic_confidence_stats[8]:.3f}")
    
    log.info(f"Average probability difference: {basic_confidence_stats[9]:.3f}")
    log.info(f"Median probability difference: {basic_confidence_stats[10]:.3f}")
    log.info(f"Min probability difference: {basic_confidence_stats[11]:.3f}")
    log.info(f"Max probability difference: {basic_confidence_stats[12]:.3f}")
    
    log.info(f"Average probability ratio: {basic_confidence_stats[13]:.3f}")
    log.info(f"Median probability ratio: {basic_confidence_stats[14]:.3f}")
    log.info(f"Min probability ratio: {basic_confidence_stats[15]:.3f}")
    log.info(f"Max probability ratio: {basic_confidence_stats[16]:.3f}")
    log.info("-" * 80)

def print_probability_histogram(con: duckdb.DuckDBPyConnection, total_reads: int):
    """Generate and print histogram of assignment probabilities."""
    try:
        prob_histogram = con.execute("""
            WITH probability_ranges AS (
                SELECT 
                    CASE 
                        WHEN primary_prob >= 0.95 THEN '0.95-1.00'
                        WHEN primary_prob >= 0.90 THEN '0.90-0.95'
                        WHEN primary_prob >= 0.85 THEN '0.85-0.90'
                        WHEN primary_prob >= 0.80 THEN '0.80-0.85'
                        WHEN primary_prob >= 0.75 THEN '0.75-0.80'
                        WHEN primary_prob >= 0.70 THEN '0.70-0.75'
                        WHEN primary_prob >= 0.60 THEN '0.60-0.70'
                        WHEN primary_prob >= 0.50 THEN '0.50-0.60'
                        ELSE '<0.50'
                    END AS prob_range,
                    COUNT(*) AS read_count
                FROM alignment_confidence
                GROUP BY prob_range
                ORDER BY prob_range DESC
            )
            SELECT * FROM probability_ranges
        """).fetchall()

        log.info("Assignment probability histogram:")
        for range_name, count in prob_histogram:
            bar_length = min(40, max(1, int(40 * count / total_reads)))
            bar = '#' * bar_length
            log.info(f"{range_name:>10}: {bar} {count:,} ({count/total_reads*100:.1f}%)")

    except Exception as e:
        log.error(f"Error generating probability histogram: {e}")

def generate_assignment_statistics_mmap(
    probabilities: np.ndarray,
    query_inverse_indices: np.ndarray,
    unique_queries: np.ndarray,
    mmap_folder: str,
) -> Dict[str, np.ndarray]:
    """Generate assignment statistics for each query using memory-mapped arrays."""
    
    n_queries = len(unique_queries)
    
    # Initialize statistics arrays
    max_prob = initialize_mmap_array(
        total_positions=n_queries,
        dtype=np.float64,
        mmap_folder=mmap_folder,
        array_name="max_prob_per_query",
    )
    
    sum_prob = initialize_mmap_array(
        total_positions=n_queries,
        dtype=np.float64,
        mmap_folder=mmap_folder,
        array_name="sum_prob_per_query",
    )
    
    count_alignments = initialize_mmap_array(
        total_positions=n_queries,
        dtype=np.int32,
        mmap_folder=mmap_folder,
        array_name="count_alignments_per_query",
    )
    
    # Calculate statistics for each query
    for i, query_idx in enumerate(unique_queries):
        query_mask = query_inverse_indices == i
        query_probs = probabilities[query_mask]
        
        if len(query_probs) > 0:
            max_prob[i] = np.max(query_probs)
            sum_prob[i] = np.sum(query_probs)
            count_alignments[i] = len(query_probs)
        else:
            max_prob[i] = 0.0
            sum_prob[i] = 0.0
            count_alignments[i] = 0
    
    return {
        "max_probability": max_prob,
        "sum_probability": sum_prob,
        "alignment_count": count_alignments,
    }


def generate_confidence_statistics_mmap(
    probabilities: np.ndarray,
    query_inverse_indices: np.ndarray,
    unique_queries: np.ndarray,
    mmap_folder: str,
) -> np.ndarray:
    """Generate confidence statistics for each query using memory-mapped arrays."""
    
    n_queries = len(unique_queries)
    
    confidence = initialize_mmap_array(
        total_positions=n_queries,
        dtype=np.float64,
        mmap_folder=mmap_folder,
        array_name="confidence_per_query",
    )
    
    # Calculate confidence for each query
    for i, query_idx in enumerate(unique_queries):
        query_mask = query_inverse_indices == i
        query_probs = probabilities[query_mask]
        
        if len(query_probs) > 1:
            # Sort probabilities in descending order
            sorted_probs = np.sort(query_probs)[::-1]
            # Confidence is the difference between top two probabilities
            confidence[i] = sorted_probs[0] - sorted_probs[1]
        elif len(query_probs) == 1:
            # Single alignment has maximum confidence
            confidence[i] = 1.0
        else:
            # No alignments
            confidence[i] = 0.0
    
    return confidence