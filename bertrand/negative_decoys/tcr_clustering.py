import argparse
import logging
import os

import numpy as np
from fastcluster import linkage


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script performs agglomerative clustering with complete linkage \n"
        "for a precomputed distance matrix using `fastcluster`.\n"
        "This computation is memory and CPU intensive."
    )
    parser.add_argument(
        "--distances",
        type=str,
        required=True,
        help="Path to pairwise distance matrix in condensed format",
    )
    parser.add_argument(
        "--out-linkage", required=True, type=str, help="Path to output linkage file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if os.path.isfile(args.out_linkage):
        logging.info(f"Linkage already computed: {args.out_linkage}")
        logging.info(f"(Re)move the file if you want to compute it again.")
        exit(0)

    logging.info(f"Loading distance matrix from {args.distances}")
    dist = np.load(args.distances)
    dist = dist["distance"]

    logging.info("Agglomerative clustering")
    link = linkage(dist, "complete", "precomputed")

    logging.info(f"Saving linkage to {args.out_linkage}")
    np.savez_compressed(args.out_linkage, linkage=link)
