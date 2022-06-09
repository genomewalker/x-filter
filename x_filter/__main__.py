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
from x_filter.utils import (
    get_arguments,
    create_output_files,
    create_filter_conditions,
)
from x_filter.filter import (
    resolve_multimaps,
    read_and_filter_alns,
    initialize_subject_weights,
    get_coverage_stats,
    aggregate_gene_abundances,
    convert_to_anvio,
)
import datatable as dt
import numpy as np
import gc

log = logging.getLogger("my_logger")


def main():

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s ::: %(asctime)s ::: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
        "breadth": args.breadth,
        "breadth_expected_ratio": args.breadth_expected_ratio,
        "depth": args.depth,
        "depth_evenness": args.depth_evenness,
    }

    filter_conditions = {k: v for k, v in filter_conditions.items() if v is not None}

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
        f"Reading and filtering alignments [evalue: {str(args.evalue)}; bitscore: {str(args.bitscore)}]"
    )

    alns = read_and_filter_alns(
        aln=args.input,
        bitscore=args.bitscore,
        col_names=col_names,
        evalue=args.evalue,
        threads=args.threads,
        evalue_perc=args.evalue_perc,
        evalue_perc_step=args.evalue_perc_step,
    )

    logging.info("Getting coverage statistics")
    # df = alns[
    #     :,
    #     ["subjectId", "queryId", "subjectStart", "subjectEnd", "slen", "qlen"],
    # ][
    #     :,
    #     {
    #         "Chromosome": dt.f.subjectId,
    #         "Query": dt.f.queryId,
    #         "Start": dt.f.subjectStart,
    #         "End": dt.f.subjectEnd,
    #         "slen": dt.f.slen,
    #         "qlen": dt.f.qlen,
    #     },
    # ].to_pandas()

    results = get_coverage_stats(alns, trim=args.trim)
    # Filter results
    logging.info(f"Filtering references")
    logging.info(f"::: [Filter:{args.filter}; Value:{filter_conditions[args.filter]}]")
    dt_filter_conditions = create_filter_conditions(
        filter_type=args.filter, filter_conditions=filter_conditions
    )

    # Select these references that passed the filtering from all alignments
    refs = results[dt_filter_conditions, :].copy(deep=True)

    refs = dt.unique(results[:, {"subjectId": dt.f.reference}][:, "subjectId"])
    refs[:, dt.update(keep="keep")]

    del results
    gc.collect()

    refs.key = "subjectId"
    alns = alns[
        :,
        :,
        dt.join(
            refs,
        ),
    ]
    alns = alns[dt.f.keep == "keep", :]
    del alns["keep"]
    gc.collect()
    # Count how many queries we kept
    alns.key = ["queryId", "subjectId"]

    nqueries = np.unique(alns[:, ["queryId"]]).size

    logging.info(
        f"{refs.shape[0]:,} references and {nqueries:,} queries passing filter"
    )

    logging.info(
        f"Resolving multimapping alignments [iters: {str(args.iters)}; scale: {str(args.scale)}]"
    )

    # Process multimapping reads
    # initialize the dataframe
    logging.info(f"::: Initializing data")
    alns_mp = alns[:, ["queryId", "subjectId", "bitScore", "slen"]]
    alns_mp = initialize_subject_weights(alns_mp)
    # print(
    #     alns_mp[
    #         (dt.f.subjectId == "nca:Noca_4326")
    #         & (dt.f.queryId == "11348bc767_000000480528__1"),
    #         :,
    #     ]
    # )
    if alns_mp is not None:
        alns_filtered = resolve_multimaps(
            df=alns_mp,
            threads=args.threads,
            scale=args.scale,
            iters=args.iters,
        )
        alns_filtered = alns_filtered[:, ["queryId", "subjectId"]]
        logging.info(f"Garbage collection")
        gc.collect()
        logging.info(f"Removing multimapping queries")
        alns = alns_filtered[:, :, dt.join(alns)]
    else:
        logging.info("No multimapping reads found.")

    logging.info("Getting coverage statistics")

    df = alns[
        :,
        [
            "subjectId",
            "queryId",
            "subjectStart",
            "subjectEnd",
            "slen",
            "qlen",
            "alnLength",
        ],
    ]

    results = get_coverage_stats(df, trim=args.trim)

    # Filter results
    logging.info(f"Filtering references")
    logging.info(f"::: [Filter:{args.filter}; Value:{filter_conditions[args.filter]}]")
    dt_filter_conditions = create_filter_conditions(
        filter_type=args.filter, filter_conditions=filter_conditions
    )

    # Select these references that passed the filtering from all alignments
    results_filtered = results[dt_filter_conditions, :]

    refs = dt.unique(results_filtered[:, {"subjectId": dt.f.reference}][:, "subjectId"])
    refs[:, dt.update(keep="keep")]

    refs.key = "subjectId"
    alns = alns[
        :,
        :,
        dt.join(
            refs,
        ),
    ]
    alns = alns[dt.f.keep == "keep", :]
    del alns["keep"]
    # Count how many queries we kept

    nqueries = np.unique(alns[:, ["queryId"]]).size

    logging.info(
        f"{refs.shape[0]:,} references and {nqueries:,} queries passing filter"
    )

    # write stats to file
    logging.info(f"Writing filtered alignments to {out_files['multimap']}")
    alns[dt.bool8] = dt.int32
    alns = alns.to_pandas()[col_names[0]]
    alns.to_csv(out_files["multimap"], sep="\t", index=False, compression="gzip")

    logging.info(f"Writing coverage statistics to {out_files['coverage']}")
    results_filtered.to_pandas().to_csv(
        out_files["coverage"], sep="\t", index=False, compression="gzip"
    )
    # use map file to aggregate gene abundances
    if args.mapping_file:
        logging.info("Aggregating gene abundances")
        gene_abundances, gene_abundances_agg = aggregate_gene_abundances(
            mapping_file=args.mapping_file,
            gene_abundances=results_filtered,
            threads=args.threads,
        )

        if gene_abundances is None:
            logging.info("Couldn't map anything to the references.")
            logging.info(f"ALL DONE.")
            exit(0)
        logging.info(f"Writing group abundances to {out_files['group_abundances']}")
        gene_abundances.to_csv(
            out_files["group_abundances"], sep="\t", index=False, compression="gzip"
        )

        if args.anvio:
            gene_abundances_agg_anvio = convert_to_anvio(
                df=gene_abundances, annotation_source=args.annotation_source
            )
            logging.info(
                f"Writing abundances with anvi'o format to {out_files['group_abundances_anvio']}"
            )
            gene_abundances_agg_anvio.to_csv(
                out_files["group_abundances_anvio"],
                sep="\t",
                index=False,
                compression="gzip",
            )

        logging.info(
            f"Writing aggregated group abundances to {out_files['group_abundances_agg']}"
        )
        gene_abundances_agg.to_csv(
            out_files["group_abundances_agg"], sep="\t", index=False, compression="gzip"
        )


if __name__ == "__main__":
    main()
