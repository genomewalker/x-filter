# x_filter/db/__init__.py

"""
Database operations modules for xFilter.

This package contains modules for database operations, including
DuckDB interactions and memory-mapped array management.
"""

from x_filter.db.mmap import (
    filter_arrays_by_subjects,
    initialize_mmap_array,
    memory_efficient_factorize,
    cleanup_mmap_files,
    slice_arrays,
    get_representative_indices,
)
