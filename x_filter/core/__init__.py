# x_filter/core/__init__.py

"""
Core functionality modules for xFilter.

This package contains the core functional modules that implement
the main filtering, coverage calculation, and read reassignment 
algorithms of xFilter.
"""

from x_filter.core.filtering import filter_arrays, apply_filters
from x_filter.core.coverage import calculate_coverage_statistics
from x_filter.core.reassignment import reassign_reads
from x_filter.core.preprocessing import process_input_data
from x_filter.core.aggregation import aggregate_gene_abundances, convert_to_anvio
from x_filter.core.io import setup_temporary_directory, cleanup_temp_files
