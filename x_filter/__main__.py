"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
"""


import logging
from re import A
import pandas as pd
import numpy as np
from x_filter.utils import get_arguments, create_output_files, apply_parallel, concat_df
from x_filter.filter import resolve_multimaps, read_and_filter_alns, get_stats
import datatable as dt

log = logging.getLogger("my_logger")


def main():

    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)s ::: %(asctime)s ::: %(message)s"
    )

    args = get_arguments()
    logging.getLogger("my_logger").setLevel(
        logging.DEBUG if args.debug else logging.INFO
    )

    # Create output files
    out_files = create_output_files(prefix=args.prefix, input=args.input)

    filter_conditions = {
        "bitscore": args.bitscore,
        "evalue": args.evalue,
        "expected_breadth_ratio": args.expected_breadth_ratio,
        "depth": args.depth,
        "depth_evenness": args.depth_evenness,
    }

    col_names = (
        [
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
        ],
    )

    # Read in aln file and filter
    logging.info(
        f"Reading and filtering alignments [evalue: {str(args.evalue)}; bitscore: {str(args.bitscore)}]..."
    )
    alns = read_and_filter_alns(
        aln=args.input,
        bitscore=args.bitscore,
        col_names=col_names,
        evalue=args.evalue,
        threads=args.threads,
    )

    logging.info(
        f"Resolving multimapping alignments [iters: {str(args.iters)}; scale: {str(args.scale)}]..."
    )
    # Process multimapping reads
    alns_filtered = resolve_multimaps(
        df=alns,
        threads=args.threads,
        scale=args.scale,
        iters=args.iters,
    )

    logging.info("Combining filtered alignments...")
    queries = dt.unique(alns_filtered["queryId"]).to_pandas()
    outer_join = alns.to_pandas().merge(queries, how="outer", indicator=True)
    anti_join = outer_join[~(outer_join._merge == "both")].drop("_merge", axis=1)
    alns = concat_df([anti_join, alns_filtered.to_pandas()])

    df = (
        dt.Frame(alns)[
            :,
            ["subjectId", "subjectStart", "subjectEnd", "slen"],
        ][:, :, dt.sort("subjectId")]
        .to_pandas()
        .rename(
            columns={
                "subjectId": "Chromosome",
                "subjectStart": "Start",
                "subjectEnd": "End",
                "slen": "len",
            }
        )
    )

    logging.info("Getting coverage statistics...")
    results = apply_parallel(df.groupby("Chromosome"), get_stats, threads=args.threads)

    logging.info("Filtering references...")
    logging.info(
        f"depth >= {filter_conditions['depth']} & depth_evenness <= {filter_conditions['depth_evenness']} & expected_breadth_ratio >= {filter_conditions['expected_breadth_ratio']}"
    )

    results_filtered = results.loc[
        (results["depth_mean"] >= filter_conditions["depth"])
        & (results["depth_evenness"] <= filter_conditions["depth_evenness"])
        & (results["breadth_exp_ratio"] >= filter_conditions["expected_breadth_ratio"])
    ]
    # write stats to file
    logging.info(f"Writing filtered alignments to {out_files['multimap']}")
    alns = alns[col_names[0]]
    alns.to_csv(out_files["multimap"], sep="\t", index=False, compression="gzip")

    logging.info(f"Writing coverage statistics to {out_files['coverage']}")
    results_filtered.to_csv(
        out_files["coverage"], sep="\t", index=False, compression="gzip"
    )

    logging.info(f"ALL DONE.")


if __name__ == "__main__":
    main()
