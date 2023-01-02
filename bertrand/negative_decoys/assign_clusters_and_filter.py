import argparse
import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster


def parse_args():
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script assigns clusters for TCR dataset and filters reference TCRs in clusters with positive TCRs. "
    )
    parser.add_argument(
        "--tcrs",
        type=str,
        required=True,
        help="Path to positive&reference TCRs dataset",
    )
    parser.add_argument(
        "--linkage", type=str, required=True, help="Path to clustering output",
    )
    parser.add_argument(
        "--out-tcrs", required=True, type=str, help="Path to output filtered",
    )

    parser.add_argument(
        "--linkage-thr", type=int, default=3, help="Cluster linkage threshold",
    )
    return parser.parse_args()


def assign_clusters(dataset: pd.DataFrame, linkage: np.ndarray, thr: int) -> None:
    """
    Assign clusters from agglomerative clustering
    :param dataset: positive&reference TCRs - pd.DataFrame
    :param linkage: result from agglomerative clustering (`tcr_clustering.py`)
    :param thr: threshold for maximum intra-cluster distance (complete linkage)
    Adds `tcr_cluster` column to the data frame.
    """
    clusters = fcluster(linkage, thr, criterion="distance")
    dataset.loc[:, "tcr_cluster"] = clusters


def filter_reference_in_positive_clusters(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Removed reference TCRs that cluster together with positive TCRs
    :param dataset: positive&reference TCRs
    :return: filtered dataset
    """
    positive_clusters = dataset.query("y==1").tcr_cluster.unique()
    mask = (dataset.y == 1) | ~dataset.tcr_cluster.isin(positive_clusters)
    dataset = dataset[mask]
    return dataset.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    logging.info("Reading input data")
    dataset = pd.read_csv(args.tcrs)
    linkage = np.load(args.linkage)["linkage"]

    logging.info("Assigning clusters")
    assign_clusters(dataset, linkage, args.linkage_thr)

    logging.info("Filtering reference TCRs")
    dataset = filter_reference_in_positive_clusters(dataset)

    logging.info("Saving TCR dataset")
    dataset.to_csv(args.out_tcrs, index=False)
