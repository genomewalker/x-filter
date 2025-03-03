# x_filter/__main__.py

import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from x_filter.cli import parse_arguments
from x_filter.utils.logging import setup_logging, get_logger, LogContext
from x_filter.core.io import setup_temporary_directory, cleanup_temp_files
from x_filter.core.preprocessing import process_input_data
from x_filter.core.filtering import filter_arrays, apply_filters
from x_filter.core.coverage import calculate_coverage_statistics
from x_filter.core.aggregation import save_results

log = get_logger(__name__)


def main():
    """Main entry point for xFilter."""
    # Parse command-line arguments
    args, filters = parse_arguments()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level)

    log.info(f"Starting xFilter v{get_version()}")
    log.info(f"Input file: {args.input}")

    start_time = time.time()

    # Create output files and temporary directories
    out_files = create_output_files(prefix=args.prefix, input_file=args.input)
    tmp_dir, tmp_files = setup_temporary_directory(base_dir=args.tmp_dir)

    try:
        with LogContext(log, "Processing data"):
            # Process input file to get arrays
            stats_df, unique_subjects, inverse_indices, arrays, db_file = (
                process_input_data(
                    args.input,
                    tmp_dir=tmp_dir,
                    tmp_files=tmp_files,
                    out_files=out_files,
                    num_threads=args.threads,
                    evalue=args.evalue,
                    bitscore=args.bitscore,
                    max_memory=args.max_memory,
                    disable_initial_filtering=args.disable_initial_filtering,
                    keep_db=args.keep_db,
                )
            )

            # Apply user-specified filters
            if filters and not args.disable_initial_filtering:
                stats_df = apply_filters(stats_df, filters)

            # Filter arrays
            filtered_df = filter_arrays(
                stats_df=stats_df,
                unique_subjects=unique_subjects,
                inverse_indices=inverse_indices,
                numpy_arrays=arrays,
                tmp_files=tmp_files,
                skip_reassign=args.skip_reassign,
                iters=args.n_iters,
                max_memory=args.max_memory,
                num_threads=args.threads,
            )

            log.info(f"Number of alignments after filtering: {len(filtered_df):,}")

            # Process filtered data
            df = process_filtered_data(filtered_df, db_file, tmp_files, args)

            # Calculate final coverage statistics
            log.info("Calculating final coverage statistics")
            analysis_arrays = analyze_alignments(df)

            final_stats, _, _, _ = calculate_coverage_statistics(
                analysis_arrays,
                tmp_files,
                num_threads=args.threads,
                rm_dups=False,
                max_memory=args.max_memory,
            )

            # Apply final filters if any
            if filters:
                log.info("Applying final filters")
                final_stats = apply_filters(final_stats, filters)

                # Filter results to keep only subjects that passed the filter
                keep_subjects = set(final_stats["subject_numeric_id"])
                df = df[df["subject_numeric_id"].isin(keep_subjects)]

            log.info(f"Final number of references: {len(final_stats):,}")

            # Save results
            save_results(
                final_stats=final_stats,
                alignments_df=df,
                out_files=out_files,
                mapping_file=args.mapping_file,
                anvio=args.anvio,
                annotation_source=args.annotation_source,
                tmp_files=tmp_files,
                num_threads=args.threads,
            )

        elapsed_time = time.time() - start_time
        log.info(f"xFilter completed successfully in {elapsed_time:.2f} seconds")

    except Exception as e:
        log.error(f"Error: {str(e)}")
        if args.debug:
            import traceback

            log.debug(traceback.format_exc())
        return 1

    finally:
        # Clean up temporary files
        cleanup_temp_files(tmp_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
