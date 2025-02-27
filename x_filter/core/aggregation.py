# x_filter/core/aggregation.py

import os
import pandas as pd
import duckdb
from typing import Dict, Tuple, Optional, Any, Union
from pathlib import Path

from x_filter.utils.logging import get_logger, LogContext

log = get_logger(__name__)


def aggregate_gene_abundances(
    mapping_file: str,
    gene_abundances: pd.DataFrame,
    tmp_files: Dict[str, str],
    num_threads: int = 1,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Aggregate gene abundances based on mapping file.

    Args:
        mapping_file: Path to mapping file
        gene_abundances: Gene abundance data
        tmp_files: Dictionary of temporary file paths
        num_threads: Number of threads to use

    Returns:
        Tuple of (gene_abundances, gene_abundances_aggregated)
    """
    with LogContext(log, "Aggregating gene abundances"):
        # Use DuckDB for efficient aggregation
        with duckdb.connect(":memory:") as con:
            # Register gene_abundances DataFrame
            con.register("gene_abundances", gene_abundances)

            # Configure DuckDB
            con.execute(f"SET threads={num_threads};")
            con.execute(f"SET temp_directory='{tmp_files['db']}';")
            con.execute("SET enable_progress_bar=true;")
            con.execute("SET preserve_insertion_order=false;")

            # Read mapping file and perform merge
            columns = {"reference": "VARCHAR", "group": "VARCHAR"}
            query = f"""
            SELECT m.*, g.*
            FROM read_csv_auto('{mapping_file}', header=False, columns={columns}) AS m
            INNER JOIN gene_abundances g ON m.reference = g.reference
            """

            mappings = con.execute(query).fetchdf()

            if mappings.empty:
                log.warning(
                    "No mappings found - check mapping file format and contents"
                )
                return None, None

            # Perform aggregation
            agg_query = """
            SELECT
                "group",
                AVG(depth_mean) AS coverage_mean,
                STDDEV(depth_mean) AS coverage_stdev,
                MEDIAN(depth_mean) AS coverage_median,
                SUM(depth_mean) AS coverage_sum,
                COUNT(*) AS n_genes,
                AVG(avg_read_length) AS avg_read_length,
                AVG(std_read_length) AS stdev_read_length,
                AVG(avg_identity) AS avg_identity,
                AVG(std_identity) AS stdev_identity
            FROM mappings
            GROUP BY "group"
            """

            mappings_agg = con.execute(agg_query).fetchdf()

            log.info(
                f"Aggregated {len(mappings)} mappings into {len(mappings_agg)} groups"
            )

            return mappings, mappings_agg


def convert_to_anvio(df: pd.DataFrame, annotation_source: str) -> pd.DataFrame:
    """
    Convert a DataFrame to Anvi'o-compatible format.

    Args:
        df: Input DataFrame
        annotation_source: Source of annotation

    Returns:
        Anvi'o compatible DataFrame
    """
    with LogContext(log, "Converting to Anvi'o format"):
        # Create Anvi'o format DataFrame
        anvio_df = df.assign(
            source=annotation_source,
            group=lambda x: x["group"].str.replace("ko:", "", regex=False),
        ).rename(
            columns={
                "reference": "gene_id",
                "group": "enzyme_accession",
                "depth_mean": "coverage",
                "breadth": "detection",
            }
        )[
            ["gene_id", "enzyme_accession", "source", "coverage", "detection"]
        ]

        log.info(f"Created Anvi'o compatible DataFrame with {len(anvio_df)} rows")

        return anvio_df


def save_results(
    final_stats: pd.DataFrame,
    alignments_df: pd.DataFrame,
    out_files: Dict[str, str],
    mapping_file: Optional[str] = None,
    anvio: bool = False,
    annotation_source: str = "unknown",
    tmp_files: Optional[Dict[str, str]] = None,
    num_threads: int = 1,
) -> None:
    """
    Save results to output files.

    Args:
        final_stats: Final statistics DataFrame
        alignments_df: Alignments DataFrame
        out_files: Dictionary of output file paths
        mapping_file: Optional path to mapping file
        anvio: Whether to create Anvi'o output
        annotation_source: Source of annotation for Anvi'o
        tmp_files: Dictionary of temporary file paths
        num_threads: Number of threads to use
    """
    with LogContext(log, "Saving results"):
        # Create connection
        with duckdb.connect() as con:
            # Register DataFrames
            unique_subjects = alignments_df[
                ["subjectId", "subject_numeric_id"]
            ].drop_duplicates()
            con.register("final_stats", final_stats)
            con.register("unique_subjects", unique_subjects)

            # Perform merge and column renaming
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

            # Export coverage results
            log.info(f"Saving coverage statistics to {out_files['coverage']}")
            con.execute(
                f"COPY ({coverage_query}) TO '{out_files['coverage']}' "
                "(HEADER, DELIMITER '\t')"
            )

            # Register alignment DataFrame
            con.register("alignments", alignments_df)

            # Define columns for multimap output
            base_columns = [
                "queryId",
                "subjectId",
                "percIdentity",
                "alnLength",
                "mismatchCount",
                "gapOpenCount",
                "queryStart",
                "queryEnd",
                "subjectStart",
                "subjectEnd",
                "eVal",
                "bitScore",
                "qlen",
                "slen",
            ]

            # Check if CIGAR columns exist
            cigar_columns = (
                ["cigar", "qaln", "taln"] if "cigar" in alignments_df.columns else []
            )
            all_columns = base_columns + cigar_columns
            columns_str = ", ".join(all_columns)

            # Export alignments
            log.info(f"Saving filtered alignments to {out_files['multimap']}")
            con.execute(
                f"COPY (SELECT {columns_str} FROM alignments) TO '{out_files['multimap']}' "
                "(HEADER, DELIMITER '\t')"
            )

            # Process mapping file if provided
            if mapping_file and tmp_files:
                log.info("Processing mapping file")
                gene_abundances, gene_abundances_agg = aggregate_gene_abundances(
                    mapping_file=mapping_file,
                    gene_abundances=final_stats,
                    tmp_files=tmp_files,
                    num_threads=num_threads,
                )

                if gene_abundances is None:
                    log.warning("Couldn't map anything to the references.")
                    return

                # Register gene abundance DataFrames
                con.register("gene_abundances", gene_abundances)
                con.register("gene_abundances_agg", gene_abundances_agg)

                # Export gene abundances
                log.info(f"Saving group abundances to {out_files['group_abundances']}")
                con.execute(
                    f"COPY (SELECT * FROM gene_abundances) TO '{out_files['group_abundances']}' "
                    "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
                )

                log.info(
                    f"Saving aggregated group abundances to {out_files['group_abundances_agg']}"
                )
                con.execute(
                    f"COPY (SELECT * FROM gene_abundances_agg) TO '{out_files['group_abundances_agg']}"
                    "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
                )

                # Create Anvi'o output if requested
                if anvio:
                    log.info("Creating Anvi'o output")
                    gene_abundances_anvio = convert_to_anvio(
                        df=gene_abundances, annotation_source=annotation_source
                    )

                    con.register("gene_abundances_anvio", gene_abundances_anvio)

                    log.info(
                        f"Saving Anvi'o compatible output to {out_files['group_abundances_anvio']}"
                    )
                    con.execute(
                        f"COPY (SELECT * FROM gene_abundances_anvio) TO '{out_files['group_abundances_anvio']}' "
                        "(HEADER, DELIMITER '\t', COMPRESSION 'gzip')"
                    )
