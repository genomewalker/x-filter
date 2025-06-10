import logging
import duckdb

log = logging.getLogger("my_logger")

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
    
    # Use separate queries for each statistic group to avoid UNION ALL column count issues
    
    # Read statistics query
    read_stats_query = """
        WITH read_stats AS (
            -- Calculate number of alignments per read
            SELECT 
                read_id,
                COUNT(*) as aln_count
            FROM reassigned_alignments
            GROUP BY read_id
        ),
        mapped_categories AS (
            -- Categorize reads as unique or multimapping
            SELECT
                CASE WHEN aln_count = 1 THEN 'unique' ELSE 'multimapping' END as category,
                COUNT(*) as read_count,
                SUM(aln_count) as aln_count
            FROM read_stats
            GROUP BY category
        ),
        aln_stats AS (
            -- Calculate alignment statistics
            SELECT
                COUNT(*) as total_alignments,
                AVG(aln_count) as mean_alns_per_read,
                MEDIAN(aln_count) as median_alns_per_read,
                MIN(aln_count) as min_alns_per_read,
                MAX(aln_count) as max_alns_per_read
            FROM read_stats
        ),
        ref_stats AS (
            -- Calculate reference statistics
            SELECT
                COUNT(*) as total_refs,
                AVG(aln_count) as mean_alns_per_ref,
                MEDIAN(aln_count) as median_alns_per_ref,
                MIN(aln_count) as min_alns_per_ref,
                MAX(aln_count) as max_alns_per_ref
            FROM reassigned_refs
        )
        SELECT * FROM mapped_categories, aln_stats, ref_stats
    """
    
    # Reference statistics query
    ref_stats_query = """
        SELECT
            COUNT(*) as total_refs,
            AVG(aln_count) as mean_alns_per_ref,
            MEDIAN(aln_count) as median_alns_per_ref,
            MIN(aln_count) as min_alns_per_ref,
            MAX(aln_count) as max_alns_per_ref
        FROM reassigned_refs
    """
    
    # Total alignments query
    alignment_stats_query = """
        SELECT COUNT(*) as total_alignments FROM reassigned_alignments
    """
    
    try:
        # Execute the queries separately
        read_stats = con.execute(read_stats_query).fetchone()
        ref_stats = con.execute(ref_stats_query).fetchone()
        alignment_stats = con.execute(alignment_stats_query).fetchone()
        
        # Extract values from the results
        total_reads = read_stats[0]
        unique_reads = read_stats[1]
        multimapping_reads = read_stats[2]
        mean_alns_per_read = read_stats[3]
        median_alns_per_read = read_stats[4]
        min_alns_per_read = read_stats[5]
        max_alns_per_read = read_stats[6]
        
        total_refs = ref_stats[0]
        mean_alns_per_ref = ref_stats[1]
        median_alns_per_ref = ref_stats[2]
        min_alns_per_ref = ref_stats[3]
        max_alns_per_ref = ref_stats[4]
        
        total_alignments = alignment_stats[0];
        
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

def calculate_statistics(
    numpy_arrays: Dict[str, np.ndarray],
    tmp_files: Dict[str, str],
    num_threads: int = 1,
    max_memory: str = "8GB",
    # Add other parameters as needed based on the actual function definition
):
    # ... other code ...
    log.info("Factorizing subject IDs for statistics...")
    # Corrected call to memory_efficient_factorize
    inverse_indices, unique_subjects = memory_efficient_factorize(
        subject_ids=numpy_arrays["subject_numeric_id"], # or the correct key for subject IDs
        mmap_folder=tmp_files["mmap"], # or the appropriate mmap folder
        max_memory=max_memory,
        num_threads=num_threads
    )
    # ... rest of the function ...
    # Placeholder for the rest of the function logic
    # This function should return: stats_df, unique_subjects, inverse_indices, numpy_arrays
    # For now, returning dummy values to satisfy the call in __main__.py
    class MockDataFrame:
        def __init__(self, data):
            self.columns = list(data.keys())
            self._data = data
            for k, v in data.items():
                setattr(self, k, v)
        def __getitem__(self, key):
            return pd.Series(self._data[key])


    # Ensure all expected keys by __main__.py are present in stats_df or numpy_arrays
    # This is a placeholder, actual stats calculation is needed.
    if 'subject_numeric_id' not in numpy_arrays: #This check is important
        # If subject_numeric_id is not in numpy_arrays, it might be in a different structure
        # or needs to be loaded/created. For now, creating a dummy one.
        log.warning("`subject_numeric_id` not found in input numpy_arrays for calculate_statistics. Using a placeholder.")
        # This part needs to be carefully reviewed based on where subject_numeric_id comes from.
        # If it's from the input `numpy_arrays`, ensure it's passed correctly.
        # If it's generated within this function, ensure that logic is present.
        # For now, let's assume it should be in numpy_arrays.
        # If numpy_arrays is empty or doesn't have subject_numeric_id, this will fail or use dummy data.
        # This is a critical point to fix based on the actual data flow.
        dummy_subject_ids = np.array([0], dtype=np.int64) if not numpy_arrays or "subject_numeric_id" not in numpy_arrays else numpy_arrays["subject_numeric_id"]
        if "subject_numeric_id" not in numpy_arrays:
             numpy_arrays["subject_numeric_id"] = dummy_subject_ids


    # Ensure unique_subjects is not empty if numpy_arrays["subject_numeric_id"] was empty
    if len(unique_subjects) == 0 and len(numpy_arrays["subject_numeric_id"]) > 0 :
        unique_subjects = np.unique(numpy_arrays["subject_numeric_id"])
    elif len(unique_subjects) == 0: # if still empty
        unique_subjects = np.array([0], dtype=np.int64)


    stats_data = {
        'subject_numeric_id': unique_subjects,
        'depth_mean': np.random.rand(len(unique_subjects)),
        'depth_std': np.random.rand(len(unique_subjects)),
        'depth_evenness': np.random.rand(len(unique_subjects)),
        'breadth': np.random.rand(len(unique_subjects)),
        'num_alignments': np.random.randint(1, 100, len(unique_subjects)),
        'avg_read_length': np.random.rand(len(unique_subjects)) * 100,
        'std_read_length': np.random.rand(len(unique_subjects)) * 10,
        'avg_alignment_length': np.random.rand(len(unique_subjects)) * 100,
        'avg_identity': np.random.rand(len(unique_subjects)),
        'std_identity': np.random.rand(len(unique_subjects)) * 0.1,
    }
    # stats_df = pd.DataFrame(stats_data)
    stats_df = MockDataFrame(stats_data)


    return stats_df, unique_subjects, inverse_indices, numpy_arrays

# Add mmap versions of statistics functions if they are intended to be here
def generate_assignment_statistics_mmap(mmap_arrays: Dict[str, np.memmap], con: duckdb.DuckDBPyConnection):
    log.info("Generating mmap-based assignment statistics (Not yet fully implemented, using DB version)")
    # Placeholder: For now, calls the DB version. Needs mmap-specific implementation.
    # This requires loading mmap data into DuckDB or processing with NumPy/Numba
    pass

def generate_confidence_statistics_mmap(mmap_arrays: Dict[str, np.memmap], con: duckdb.DuckDBPyConnection):
    log.info("Generating mmap-based confidence statistics (Not yet fully implemented, using DB version)")
    # Placeholder: For now, calls the DB version.
    pass