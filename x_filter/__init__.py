# x_filter/__init__.py

"""
xFilter: A BLASTx filtering tool for ancient DNA

A tool to filter BLASTx results with special emphasis on ancient DNA studies,
implementing the FAMLI algorithm and adding features for ancient DNA analysis.
"""

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from x_filter.core import filtering, coverage, reassignment
from x_filter.utils import logging

# Set up logging when imported
logging.setup_logging()
