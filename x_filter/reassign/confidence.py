import logging
import duckdb
import numpy as np

log = logging.getLogger("my_logger")

def apply_confidence_decay(con: duckdb.DuckDBPyConnection, decay_type: str):
    """
    Apply decay function to probability scores based on specified method.
    
    Args:
        con: DuckDB connection
        decay_type: Type of decay function to apply ('linear', 'exponential', 'sigmoid', 'none')
    """
    if decay_type == "none":
        log.info("No decay applied - probabilities remain unchanged")
        return
        
    log.info(f"Applying {decay_type} decay to probability scores...")
    
    if decay_type == "linear":
        # Linear decay: reduce probabilities linearly based on rank
        con.execute("""
            UPDATE read_prob
            SET prob = prob * (1.0 - 0.1 * (ROW_NUMBER() OVER (PARTITION BY read_id ORDER BY prob DESC) - 1) / 
                              GREATEST(COUNT(*) OVER (PARTITION BY read_id) - 1, 1))
        """)
    
    elif decay_type == "exponential":
        # Exponential decay: exponentially reduce lower-ranked hits
        con.execute("""
            UPDATE read_prob
            SET prob = prob * EXP(-0.5 * (ROW_NUMBER() OVER (PARTITION BY read_id ORDER BY prob DESC) - 1))
        """)
    
    elif decay_type == "sigmoid":
        # Sigmoid decay: smooth transition between high and low confidence
        con.execute("""
            UPDATE read_prob
            SET prob = prob * (1.0 / (1.0 + EXP(2.0 * (ROW_NUMBER() OVER (PARTITION BY read_id ORDER BY prob DESC) - 2))))
        """)
    
    # Normalize probabilities after decay
    con.execute("""
        -- Calculate sum of probabilities per read
        WITH prob_sums AS (
            SELECT read_id, SUM(prob) as sum_prob
            FROM read_prob
            GROUP BY read_id
        )
        -- Normalize probabilities
        UPDATE read_prob
        SET prob = read_prob.prob / prob_sums.sum_prob
        FROM prob_sums
        WHERE read_prob.read_id = prob_sums.read_id
    """)

def implement_selection_mode(
    con: duckdb.DuckDBPyConnection, 
    selection_mode: str, 
    reference_bias: float, 
    handle_ties: str, 
    min_assignment_confidence: float, 
    min_confidence_margin: float,
    random_seed: int = 42  # Add seed parameter
):
    """
    Implement the selected strategy for handling read assignments with deterministic behavior.
    
    Args:
        con: DuckDB connection
        selection_mode: Strategy for assignment ('hard_cutoff', 'weighted', 'proportional', 'bayesian')
        reference_bias: Factor to apply reference-based weighting (0.0-1.0)
        handle_ties: How to handle tied best hits ('keep_all', 'keep_one', 'discard')
        min_assignment_confidence: Minimum probability threshold
        min_confidence_margin: Minimum probability difference to second-best
        random_seed: Seed for deterministic tie-breaking
    """
    log.info(f"Implementing {selection_mode} selection mode with seed {random_seed}...")
    
    # Set deterministic seed for consistent tie-breaking
    con.execute(f"SELECT setseed({random_seed / 2147483647.0})")  # Normalize seed to [0,1]
    
    # Log current filtering parameters for debugging
    log.info(f"Filtering parameters:")
    log.info(f"  Min assignment confidence: {min_assignment_confidence}")
    log.info(f"  Min confidence margin: {min_confidence_margin}")
    log.info(f"  Handle ties: {handle_ties}")
    log.info(f"  Reference bias: {reference_bias}")
    
    # ANALYZE DATASET CHARACTERISTICS FIRST
    dataset_stats = con.execute("""
        WITH read_alignment_counts AS (
            SELECT 
                read_id,
                COUNT(*) as alignments_per_read,
                MAX(prob) as max_prob,
                AVG(prob) as avg_prob,
                STDDEV(prob) as std_prob
            FROM read_prob
            GROUP BY read_id
        ),
        overall_stats AS (
            SELECT 
                COUNT(*) as total_reads,
                AVG(alignments_per_read) as avg_alignments_per_read,
                quantile_cont(alignments_per_read, 0.5) as median_alignments_per_read,
                quantile_cont(alignments_per_read, 0.9) as p90_alignments_per_read,
                AVG(max_prob) as avg_max_prob,
                quantile_cont(max_prob, 0.1) as p10_max_prob,
                quantile_cont(max_prob, 0.5) as p50_max_prob,
                quantile_cont(max_prob, 0.9) as p90_max_prob,
                SUM(CASE WHEN alignments_per_read = 1 THEN 1 ELSE 0 END) as single_mapping_reads,
                SUM(CASE WHEN alignments_per_read > 10 THEN 1 ELSE 0 END) as highly_multimapping_reads
            FROM read_alignment_counts
        ),
        prob_distribution AS (
            SELECT 
                quantile_cont(prob, 0.05) as p05_prob,
                quantile_cont(prob, 0.1) as p10_prob,
                quantile_cont(prob, 0.25) as p25_prob,
                quantile_cont(prob, 0.5) as p50_prob,
                quantile_cont(prob, 0.75) as p75_prob,
                quantile_cont(prob, 0.9) as p90_prob,
                quantile_cont(prob, 0.95) as p95_prob,
                AVG(prob) as mean_prob,
                STDDEV(prob) as std_prob
            FROM read_prob
        )
        SELECT 
            o.total_reads,
            o.avg_alignments_per_read,
            o.median_alignments_per_read,
            o.p90_alignments_per_read,
            o.single_mapping_reads * 100.0 / o.total_reads as single_mapping_pct,
            o.highly_multimapping_reads * 100.0 / o.total_reads as highly_multimapping_pct,
            o.avg_max_prob,
            o.p10_max_prob,
            o.p50_max_prob,
            o.p90_max_prob,
            p.p05_prob,
            p.p10_prob,
            p.p25_prob,
            p.p50_prob,
            p.p75_prob,
            p.p90_prob,
            p.p95_prob,
            p.mean_prob,
            p.std_prob
        FROM overall_stats o, prob_distribution p
    """).fetchone()
    
    # Unpack dataset characteristics
    (total_reads, avg_alignments, median_alignments, p90_alignments,
     single_mapping_pct, highly_multimapping_pct, avg_max_prob, p10_max_prob, p50_max_prob, p90_max_prob,
     p05_prob, p10_prob, p25_prob, p50_prob, p75_prob, p90_prob, p95_prob, 
     mean_prob, std_prob) = dataset_stats
    
    log.info("=" * 60)
    log.info("DATASET ANALYSIS FOR ADAPTIVE THRESHOLDING:")
    log.info(f"  Total reads: {total_reads:,}")
    log.info(f"  Avg alignments per read: {avg_alignments:.2f}")
    log.info(f"  Median alignments per read: {median_alignments:.0f}")
    log.info(f"  90th percentile alignments: {p90_alignments:.0f}")
    log.info(f"  Single-mapping reads: {single_mapping_pct:.1f}%")
    log.info(f"  Highly multi-mapping reads (>10 alignments): {highly_multimapping_pct:.1f}%")
    log.info("")
    log.info("PROBABILITY DISTRIBUTION:")
    log.info(f"  Mean probability: {mean_prob:.4f}")
    log.info(f"  Probability percentiles: P5={p05_prob:.4f}, P25={p25_prob:.4f}, P50={p50_prob:.4f}")
    log.info(f"                          P75={p75_prob:.4f}, P90={p90_prob:.4f}, P95={p95_prob:.4f}")
    log.info(f"  Max prob distribution: P10={p10_max_prob:.4f}, P50={p50_max_prob:.4f}, P90={p90_max_prob:.4f}")
    log.info("=" * 60)
    
    # CLASSIFY DATASET TYPE AND SET ADAPTIVE THRESHOLDS
    if highly_multimapping_pct > 30 and avg_alignments > 5:
        dataset_type = "HIGHLY_MULTIMAPPING"
        # Very permissive thresholds for highly multi-mapping data
        adaptive_min_conf = max(0.001, min(min_assignment_confidence, p10_prob * 0.5))
        adaptive_margin = max(0.0001, min(min_confidence_margin, std_prob * 0.1))
        selection_strategy = "permissive"
        
    elif single_mapping_pct > 70:
        dataset_type = "MOSTLY_UNIQUE"
        # More stringent thresholds for mostly unique data
        adaptive_min_conf = max(min_assignment_confidence, p25_prob)
        adaptive_margin = max(min_confidence_margin, std_prob * 0.5)
        selection_strategy = "stringent"
        
    else:
        dataset_type = "MIXED_MAPPING"
        # Balanced thresholds
        adaptive_min_conf = max(min_assignment_confidence * 0.1, p05_prob * 2)
        adaptive_margin = max(min_confidence_margin * 0.1, std_prob * 0.2)
        selection_strategy = "balanced"
    
    log.info(f"DATASET TYPE: {dataset_type}")
    log.info(f"SELECTION STRATEGY: {selection_strategy}")
    log.info(f"USER REQUESTED:")
    log.info(f"  Min confidence: {min_assignment_confidence:.4f}")
    log.info(f"  Min margin: {min_confidence_margin:.4f}")
    log.info(f"ADAPTIVE THRESHOLDS:")
    log.info(f"  Adaptive min confidence: {adaptive_min_conf:.6f}")
    log.info(f"  Adaptive margin: {adaptive_margin:.6f}")
    
    if selection_mode == "primary":
        log.info("Using primary selection with dataset-adaptive thresholds and deterministic tie-breaking")
        
        # Create selection query based on dataset type with deterministic ordering
        if dataset_type == "HIGHLY_MULTIMAPPING":
            selection_query = f"""
                CREATE TABLE selected_alignments AS
                WITH ranked_probs AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY read_id ORDER BY prob DESC, orig_idx ASC) as rank,
                        LAG(prob, 1, 0) OVER (PARTITION BY read_id ORDER BY prob DESC, orig_idx ASC) as second_best_prob,
                        COUNT(*) OVER (PARTITION BY read_id) as total_alignments_for_read,
                        MAX(prob) OVER (PARTITION BY read_id) as max_prob_for_read,
                        quantile_cont(prob, 0.5) OVER (PARTITION BY read_id) as median_prob_for_read
                    FROM read_prob
                    WHERE prob >= {adaptive_min_conf * 0.1}  -- Very permissive initial filter
                ),
                confident_assignments AS (
                    SELECT *,
                        CASE 
                            WHEN rank = 1 THEN prob - COALESCE(second_best_prob, 0)
                            ELSE 0
                        END as confidence_margin,
                        prob / max_prob_for_read as relative_prob,
                        prob / median_prob_for_read as relative_to_median
                    FROM ranked_probs
                )
                SELECT orig_idx as selected_rowid, prob as selected_prob, read_id, ref_id
                FROM confident_assignments
                WHERE 
                    -- Best hits with any reasonable probability
                    (rank = 1 AND prob >= {adaptive_min_conf})
                    OR
                    -- Competitive alternatives (keep many for multi-mapping)
                    (rank <= 10 AND relative_prob >= 0.2 AND prob >= {adaptive_min_conf * 0.5})
                    OR
                    -- Additional alternatives for very uncertain reads
                    (rank <= 20 AND relative_prob >= 0.1 AND prob >= {adaptive_min_conf * 0.2} AND total_alignments_for_read >= 10)
                    OR
                    -- Keep anything above median for reads with many alignments
                    (total_alignments_for_read > 20 AND prob >= median_prob_for_read AND rank <= 30)
                ORDER BY read_id, prob DESC, orig_idx ASC  -- Deterministic ordering
            """
            
        elif dataset_type == "MOSTLY_UNIQUE":
            selection_query = f"""
                CREATE TABLE selected_alignments AS
                WITH ranked_probs AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY read_id ORDER BY prob DESC, orig_idx ASC) as rank,
                        LAG(prob, 1, 0) OVER (PARTITION BY read_id ORDER BY prob DESC, orig_idx ASC) as second_best_prob,
                        COUNT(*) OVER (PARTITION BY read_id) as total_alignments_for_read,
                        MAX(prob) OVER (PARTITION BY read_id) as max_prob_for_read
                    FROM read_prob
                    WHERE prob >= {adaptive_min_conf}
                ),
                confident_assignments AS (
                    SELECT *,
                        CASE 
                            WHEN rank = 1 THEN prob - COALESCE(second_best_prob, 0)
                            ELSE 0
                        END as confidence_margin,
                        prob / max_prob_for_read as relative_prob
                    FROM ranked_probs
                )
                SELECT orig_idx as selected_rowid, prob as selected_prob, read_id, ref_id
                FROM confident_assignments
                WHERE 
                    -- High confidence primary assignments
                    (rank = 1 AND prob >= {adaptive_min_conf} AND confidence_margin >= {adaptive_margin})
                    OR
                    -- Very competitive secondary hits
                    (rank <= 3 AND relative_prob >= 0.8 AND prob >= {adaptive_min_conf * 1.5})
                    OR
                    -- Single-mapping reads with decent probability
                    (total_alignments_for_read = 1 AND prob >= {adaptive_min_conf * 0.5})
                ORDER BY read_id, prob DESC, orig_idx ASC  -- Deterministic ordering
            """
            
        else:  # MIXED_MAPPING
            selection_query = f"""
                CREATE TABLE selected_alignments AS
                WITH ranked_probs AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY read_id ORDER BY prob DESC, orig_idx ASC) as rank,
                        LAG(prob, 1, 0) OVER (PARTITION BY read_id ORDER BY prob DESC, orig_idx ASC) as second_best_prob,
                        COUNT(*) OVER (PARTITION BY read_id) as total_alignments_for_read,
                        MAX(prob) OVER (PARTITION BY read_id) as max_prob_for_read
                    FROM read_prob
                    WHERE prob >= {adaptive_min_conf * 0.5}
                ),
                confident_assignments AS (
                    SELECT *,
                        CASE 
                            WHEN rank = 1 THEN prob - COALESCE(second_best_prob, 0)
                            ELSE 0
                        END as confidence_margin,
                        prob / max_prob_for_read as relative_prob
                    FROM ranked_probs
                )
                SELECT orig_idx as selected_rowid, prob as selected_prob, read_id, ref_id
                FROM confident_assignments
                WHERE 
                    -- Primary assignments with good confidence
                    (rank = 1 AND prob >= {adaptive_min_conf} AND confidence_margin >= {adaptive_margin})
                    OR
                    -- Best hit regardless if probability is decent
                    (rank = 1 AND prob >= {adaptive_min_conf * 0.8})
                    OR
                    -- Competitive alternatives for multi-mapping
                    (rank <= 5 AND relative_prob >= 0.5 AND prob >= {adaptive_min_conf} AND total_alignments_for_read > 1)
                    OR
                    -- Single-mapping reads
                    (total_alignments_for_read = 1 AND prob >= {adaptive_min_conf * 0.3})
                ORDER BY read_id, prob DESC, orig_idx ASC  -- Deterministic ordering
            """
        
        con.execute(selection_query)
        
    elif selection_mode == "threshold":
        # Simple threshold-based filtering with deterministic ordering
        con.execute(f"""
            CREATE TABLE selected_alignments AS
            SELECT orig_idx as selected_rowid, prob as selected_prob, read_id, ref_id
            FROM read_prob
            WHERE prob >= {adaptive_min_conf}
            ORDER BY read_id, prob DESC, orig_idx ASC
        """)
        
    else:
        # Use original logic for other selection modes but with deterministic ordering
        if selection_mode == "weighted":
            con.execute(f"""
                CREATE TABLE selected_alignments AS
                WITH weighted_probs AS (
                    SELECT *,
                        prob * (1.0 + {reference_bias} * 
                                ((read_id * 1009 + ref_id * 2017 + {random_seed}) % 65536) / 65536.0) as weighted_prob
                    FROM read_prob
                    WHERE prob >= {adaptive_min_conf * 0.1}
                )
                SELECT orig_idx as selected_rowid, weighted_prob as selected_prob, read_id, ref_id
                FROM weighted_probs
                WHERE weighted_prob >= {adaptive_min_conf}
                ORDER BY read_id, weighted_prob DESC, orig_idx ASC
            """)
            
        elif selection_mode == "proportional":
            con.execute(f"""
                CREATE TABLE selected_alignments AS
                WITH normalized_probs AS (
                    SELECT *,
                        prob / SUM(prob) OVER (PARTITION BY read_id) as norm_prob
                    FROM read_prob
                    WHERE prob >= {adaptive_min_conf * 0.01}
                )
                SELECT orig_idx as selected_rowid, norm_prob as selected_prob, read_id, ref_id
                FROM normalized_probs
                WHERE norm_prob >= {adaptive_min_conf}
                ORDER BY read_id, norm_prob DESC, orig_idx ASC
            """)
            
        elif selection_mode == "all":
            # Keep all alignments above minimum threshold
            con.execute(f"""
                CREATE TABLE selected_alignments AS
                SELECT orig_idx as selected_rowid, prob as selected_prob, read_id, ref_id
                FROM read_prob
                WHERE prob >= {adaptive_min_conf * 0.1}
                ORDER BY read_id, prob DESC, orig_idx ASC
            """)
        
        else:
            raise ValueError(f"Unknown selection mode: {selection_mode}")
    
    # Apply deterministic tie handling
    if handle_ties == "keep_one":
        con.execute("""
            DELETE FROM selected_alignments
            WHERE (read_id, selected_prob) IN (
                SELECT read_id, selected_prob
                FROM selected_alignments
                GROUP BY read_id, selected_prob
                HAVING COUNT(*) > 1
            )
            AND ROWID NOT IN (
                SELECT MIN(ROWID)
                FROM selected_alignments
                GROUP BY read_id, selected_prob
                HAVING COUNT(*) > 1
            )
        """)
    elif handle_ties == "discard":
        con.execute("""
            DELETE FROM selected_alignments
            WHERE (read_id, selected_prob) IN (
                SELECT read_id, selected_prob
                FROM selected_alignments
                GROUP BY read_id, selected_prob
                HAVING COUNT(*) > 1
            )
        """)
    
    # COMPREHENSIVE FINAL STATISTICS
    final_stats = con.execute("""
        WITH selection_analysis AS (
            SELECT 
                COUNT(*) as selected_alignments,
                COUNT(DISTINCT read_id) as selected_reads,
                AVG(selected_prob) as avg_selected_prob,
                MIN(selected_prob) as min_selected_prob,
                MAX(selected_prob) as max_selected_prob,
                quantile_cont(selected_prob, 0.1) as p10_selected_prob,
                quantile_cont(selected_prob, 0.5) as p50_selected_prob,
                quantile_cont(selected_prob, 0.9) as p90_selected_prob
            FROM selected_alignments
        ),
        read_coverage AS (
            SELECT 
                COUNT(DISTINCT s.read_id) as covered_reads,
                COUNT(DISTINCT p.read_id) as total_original_reads,
                COUNT(DISTINCT s.read_id) * 100.0 / COUNT(DISTINCT p.read_id) as coverage_pct
            FROM read_prob p
            LEFT JOIN selected_alignments s ON p.read_id = s.read_id
        ),
        alignment_reduction AS (
            SELECT 
                COUNT(*) as original_alignments,
                (SELECT COUNT(*) FROM selected_alignments) as selected_alignments,
                (COUNT(*) - (SELECT COUNT(*) FROM selected_alignments)) as removed_alignments,
                ((COUNT(*) - (SELECT COUNT(*) FROM selected_alignments)) * 100.0 / COUNT(*)) as reduction_pct
            FROM read_prob
        )
        SELECT 
            s.selected_alignments,
            s.selected_reads,
            r.total_original_reads,
            r.coverage_pct,
            a.original_alignments,
            a.reduction_pct,
            s.avg_selected_prob,
            s.min_selected_prob,
            s.max_selected_prob,
            s.p10_selected_prob,
            s.p50_selected_prob,
            s.p90_selected_prob
        FROM selection_analysis s, read_coverage r, alignment_reduction a
    """).fetchone()
    
    # Verify we got valid results
    if final_stats is None:
        log.error("Failed to retrieve final statistics - no results returned")
        return []
    
    # Unpack with better variable names and error checking
    try:
        (selected_alignments_count, selected_reads_count, total_original_reads, 
         coverage_pct, original_alignments_count, reduction_pct,
         avg_selected_prob, min_selected_prob, max_selected_prob,
         p10_selected_prob, p50_selected_prob, p90_selected_prob) = final_stats
    except (ValueError, TypeError) as e:
        log.error(f"Failed to unpack final statistics: {e}")
        log.error(f"Raw final_stats: {final_stats}")
        return []
    
    log.info("=" * 60)
    log.info("SELECTION RESULTS:")
    log.info(f"  Selected alignments: {selected_alignments_count:,} (reduction: {reduction_pct:.1f}%)")
    log.info(f"  Selected reads: {selected_reads_count:,} / {total_original_reads:,} ({coverage_pct:.1f}% coverage)")
    log.info(f"  Original alignments: {original_alignments_count:,}")
    log.info("")
    log.info("SELECTED PROBABILITY DISTRIBUTION:")
    log.info(f"  Mean: {avg_selected_prob:.4f}")
    log.info(f"  Range: {min_selected_prob:.4f} to {max_selected_prob:.4f}")
    log.info(f"  Percentiles: P10={p10_selected_prob:.4f}, P50={p50_selected_prob:.4f}, P90={p90_selected_prob:.4f}")
    log.info("=" * 60)
    
    # RECOMMENDATIONS BASED ON RESULTS
    if coverage_pct < 50:  # Less than 50% read coverage
        log.warning("⚠️  LOW READ COVERAGE! Consider:")
        log.warning(f"   • Lowering --min-assignment-confidence (current: {min_assignment_confidence})")
        log.warning(f"   • Lowering --min-confidence-margin (current: {min_confidence_margin})")
        log.warning(f"   • Using --selection-mode=all for maximum retention")
    elif reduction_pct > 95:  # More than 95% alignment reduction
        log.warning("⚠️  VERY AGGRESSIVE FILTERING! Consider:")
        log.warning(f"   • Using more permissive thresholds")
        log.warning(f"   • Using --selection-mode=proportional or --selection-mode=all")
    elif coverage_pct > 95 and reduction_pct < 50:
        log.info("✅ EXCELLENT: High read coverage with reasonable filtering")
    
    return con.execute("SELECT selected_rowid FROM selected_alignments ORDER BY selected_rowid").fetchall()

def compute_confidence(
    probabilities: np.ndarray,
    selection_mode: str = "primary",
    assignment_threshold: float = 0.5,
    confidence_threshold: float = 0.9,
    handle_ties: str = "keep_all",
    tie_breaking_method: str = "random",
    reference_bias: float = 0.0,
    min_assignment_confidence: float = 0.01,
    min_confidence_margin: float = 0.001,
    confidence_decay: str = "none",
    normalize_by_length: bool = False,
    weight_by_quality: bool = False,
    use_posterior_sampling: bool = False,
    sampling_iterations: int = 100,
):
    """
    Compute confidence scores for assignments based on probabilities and selection parameters.

    Args:
        probabilities: Array of probabilities for assignments.
        selection_mode: Method for final read assignment selection.
        assignment_threshold: Minimum probability threshold for assignments.
        confidence_threshold: Confidence threshold for high-quality assignments.
        handle_ties: How to handle tied assignments.
        tie_breaking_method: Method for breaking ties between equal assignments.
        reference_bias: Bias towards reference sequences.
        min_assignment_confidence: Minimum confidence required for assignment.
        min_confidence_margin: Minimum margin between best and second-best assignment.
        confidence_decay: Confidence decay function for multi-mapping.
        normalize_by_length: Normalize assignment probabilities by sequence length.
        weight_by_quality: Weight assignments by alignment quality scores.
        use_posterior_sampling: Use posterior sampling for final assignments.
        sampling_iterations: Number of iterations for posterior sampling.

    Returns:
        Confidence scores for assignments.
    """
    pass  # Placeholder for the actual implementation