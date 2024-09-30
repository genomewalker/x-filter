from typing import Tuple, Optional
import pandas as pd
import duckdb


def aggregate_gene_abundances(
    mapping_file: str, gene_abundances: pd.DataFrame, num_threads: int, temp_dir: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Aggregates gene abundances based on mapping file and gene abundance data using DuckDB.

    Args:
        mapping_file (str): Path to mapping file.
        gene_abundances (pd.DataFrame): Gene abundance data.
        num_threads (int): Number of threads to use for DuckDB operations.
        temp_dir (str): Temporary directory for DuckDB operations.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: A tuple containing two pandas DataFrames.
            The first DataFrame contains the merged mapping data and gene abundance data.
            The second DataFrame contains the aggregated gene abundances.
            Returns (None, None) if the resulting mappings DataFrame is empty.
    """
    # Use a context manager for the DuckDB connection
    with duckdb.connect(":memory:") as con:
        # Register the gene_abundances DataFrame as a table in DuckDB
        con.register("gene_abundances", gene_abundances)
        con.execute(f"SET threads={num_threads};")
        con.execute(f"SET temp_directory='{temp_dir}';")
        con.execute("SET enable_progress_bar=true;")
        con.execute("SET preserve_insertion_order=false;")
        con.execute("SET force_compression='auto';")

        # Read the mapping file and perform the merge operation in DuckDB
        columns = {"reference": "VARCHAR", "group": "VARCHAR"}
        query = f"""
        SELECT m.*, g.*
        FROM read_csv_auto('{mapping_file}', header=False, columns={columns}) AS m
        INNER JOIN gene_abundances g ON m.reference = g.reference
        """
        mappings = con.execute(query).fetchdf()

        if mappings.empty:
            return None, None

        # Perform the aggregation in DuckDB
        agg_query = """
        SELECT
            "group",
            AVG(depth_mean) AS coverage_mean,
            STDDEV(depth_mean) AS coverage_stdev,
            MEDIAN(depth_mean) AS coverage_median,
            SUM(depth_mean) AS coverage_sum,
            COUNT(*) AS n_genes,
            AVG(avg_read_length) AS avg_read_length,
            AVG(stdev_read_length) AS stdev_read_length,
            AVG(avg_identity) AS avg_identity,
            AVG(stdev_identity) AS stdev_identity
        FROM mappings
        GROUP BY "group"
        """
        mappings_agg = con.execute(agg_query).fetchdf()

    return mappings, mappings_agg


def convert_to_anvio(df: pd.DataFrame, annotation_source: str) -> pd.DataFrame:
    """
    Converts a pandas dataframe to an Anvio-compatible format.

    Args:
        df (pd.DataFrame): The input dataframe to convert.
        annotation_source (str): The source of the annotation.

    Returns:
        pd.DataFrame: The converted dataframe with columns:
            gene_id, enzyme_accession, source, coverage, detection
    """
    return df.assign(
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
