"""
Functions for handling alignment selection and confidence calculations.
"""
import logging

log = logging.getLogger("my_logger")

def implement_selection_mode(con, selection_mode, reference_bias, handle_ties, min_assignment_confidence, min_confidence_margin):
    """
    Implement the selected strategy for handling read assignments.
    """
    try:
        log.info(f"Implementing {selection_mode} selection mode...")
        
        # Begin transaction to ensure table consistency
        con.execute("BEGIN TRANSACTION")
        
        # Create primary alignments table
        log.debug("Creating primary alignments table...")
        con.execute("""
            CREATE TABLE primary_alignments AS
            WITH ranked_alignments AS (
                SELECT
                    read_id,
                    ref_id,
                    orig_idx,
                    prob,
                    ROW_NUMBER() OVER (PARTITION BY read_id ORDER BY prob DESC) as rank,
                    MAX(prob) OVER (PARTITION BY read_id) -
                    LEAD(prob) OVER (PARTITION BY read_id ORDER BY prob DESC) as prob_margin
                FROM read_prob
            )
            SELECT 
                read_id,
                ref_id as primary_ref_id,
                orig_idx as primary_rowid,
                prob as primary_prob
            FROM ranked_alignments
            WHERE prob >= ?
            AND (prob_margin >= ? OR prob_margin IS NULL)  -- NULL means it's the only alignment
        """, [min_assignment_confidence, min_confidence_margin])

        # Create keep_rowids table first since it's needed by later steps
        log.info("Creating keep_rowids table for final alignment selection")
        con.execute("""
            CREATE TEMP TABLE keep_rowids AS
            SELECT primary_rowid as rowid
            FROM primary_alignments
        """)

        # Calculate confidence metrics
        log.info("Calculating confidence metrics...")
        con.execute(f"""
            CREATE TABLE alignment_confidence AS
            WITH confidence_metrics AS (
                SELECT 
                    a1.primary_rowid,
                    a1.read_id,
                    a1.primary_ref_id,
                    a1.primary_prob,
                    MIN(a2.primary_prob) as secondary_prob
                FROM primary_alignments a1
                LEFT JOIN primary_alignments a2 
                    ON a1.read_id = a2.read_id 
                    AND a1.primary_ref_id != a2.primary_ref_id
                GROUP BY a1.primary_rowid, a1.read_id, a1.primary_ref_id, a1.primary_prob
            )
            SELECT
                primary_rowid,
                read_id,
                primary_ref_id,
                primary_prob,
                secondary_prob,
                (primary_prob - COALESCE(secondary_prob, 0)) as prob_diff,
                CASE 
                    WHEN COALESCE(secondary_prob, 0) = 0 THEN NULL
                    ELSE primary_prob / secondary_prob 
                END as prob_ratio
            FROM confidence_metrics
        """)

        # Verify tables were created
        for table in ['primary_alignments', 'keep_rowids', 'alignment_confidence']:
            table_exists = con.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{table}'").fetchone()[0]
            if table_exists == 0:
                raise Exception(f"Failed to create table: {table}")

        # Log statistics about selection with better error handling
        stats = con.execute("""
            SELECT COUNT(DISTINCT read_id) as unique_reads,
                   COUNT(*) as total_alignments,
                   AVG(primary_prob) as avg_probability,
                   MIN(primary_prob) as min_probability
            FROM primary_alignments
        """).fetchone()
        
        # Handle None values safely
        unique_reads = stats[0] if stats[0] is not None else 0
        total_alignments = stats[1] if stats[1] is not None else 0
        avg_probability = stats[2] if stats[2] is not None else 0.0
        min_probability = stats[3] if stats[3] is not None else 0.0
        
        log.info(f"Selected {total_alignments:,} alignments for {unique_reads:,} unique reads")
        log.info(f"Average probability: {avg_probability:.3f}, Minimum probability: {min_probability:.3f}")

        # Commit transaction
        con.execute("COMMIT")
        return True

    except Exception as e:
        # Rollback on error
        con.execute("ROLLBACK")
        log.error(f"Failed to implement selection mode: {str(e)}")
        raise
