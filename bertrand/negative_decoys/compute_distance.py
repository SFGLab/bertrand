import argparse
import logging
import os
from typing import Generator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from leven import levenshtein
from scipy.spatial.distance import squareform


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script computes pairwise Levenshtein distances for a set of TCRs.\n"
        "This computation is memory and CPU intensive."
    )
    parser.add_argument(
        "--tcrs",
        type=str,
        required=True,
        help="Path to positive&reference TCR dataset. (Result of `outliers_filtering.py`",
    )
    parser.add_argument(
        "--out-distances",
        required=True,
        type=str,
        help="Path to output pairwise distance matrix",
    )
    parser.add_argument(
        "--out-distances-cache",
        type=str,
        default=None,
        help="Path to output cache file. Due possible to OOM errors, \n"
        "it is sometimes desirable to save the batched results directly.",
    )
    parser.add_argument(
        "--cpu", type=int, default=-1, help="Total number of CPUs to use."
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")

    return parser.parse_args()


def batch_generator(X: np.ndarray, bs=512) -> Generator[np.ndarray, None, None]:
    """
    Generates chunks of the array
    :param X: array of observations
    :param bs: size of the chunk
    :return: batches generator
    """
    for i in range(0, len(X), bs):
        yield X[i : (i + bs)]


def cdist_parallel(
    X: np.ndarray, Y: np.ndarray, metric, bsx=512, n_jobs=-1, cache_fn=None
) -> np.ndarray:
    """
    Computes distances between each pairs of observations in X and Y
    See `scipy.spatial.distance.cdist`.
    This computations is parallelized into `bsx`-sized chunks.
    Memory intensive for big (over 100k) distance matrices.
    We can use dtype=int8 because the metric is Levenshtein distance.
    :param X: first set of observations
    :param Y: second set of observations
    :param metric: callable distance function
    :param bsx: batch size
    :param n_jobs: number of CPUs available
    :param cache_fn: cache file to store chunks of computation results
        (sometimes the concatenated distance matrix doesn't fit into memory)
    :return: distance matrix
    """

    def process_batch_cdist(X: np.ndarray) -> np.ndarray:
        """
        This function processes a single batch
        Note that `Y` and `metric` are captured from the parent context.
        :param X: batch of data
        :return: distances for the batch
        """
        d = np.zeros(
            (len(X), len(Y)), dtype="uint8"
        )  # uint8 optimized for Levenshtein distance
        for i in range(len(X)):
            for j in range(len(Y)):
                d[i, j] = metric(X[i], Y[j])
        return d

    X_batch_iter = batch_generator(X, bs=bsx)
    parallel = Parallel(
        n_jobs=n_jobs, batch_size=1, verbose=2, backend="loky", max_nbytes="20M"
    )
    res = parallel(delayed(process_batch_cdist)(Xb) for Xb in X_batch_iter)
    logging.info("Distance computation complete")
    if cache_fn:
        logging.info(f"Saving to cache {cache_fn}")
        np.savez_compressed(cache_fn, *res)
        logging.info("Saved!")
    logging.info("Concatenating results")
    dist_matrix = np.vstack(res)
    return dist_matrix


def pairwise_cdr3b_distances(
    dataset: pd.DataFrame, args: argparse.Namespace
) -> np.ndarray:
    """
    computes pairwise Levenshtein distance and saves the results
    :param dataset: dataframe with CDR3b column
    :param args: call args, see `parse_args`
    :return: distance matrix in a condensed format
    """
    cdrs = dataset.CDR3b.values
    logging.info(f"{len(cdrs) // args.batch_size} tasks to compute")
    dist_all_sq = cdist_parallel(
        cdrs,
        cdrs,
        levenshtein,
        bsx=args.batch_size,
        n_jobs=args.cpu,
        cache_fn=args.out_distances_cache,
    )
    # dist_all_sq is a square matrix, we need to convert it to the the condensed form
    logging.info(f"Converting the distance matrix to condensed format")
    dist_all = squareform(dist_all_sq)
    # Save the distance matrix
    logging.info(f"Saving the distance matrix to {args.out_distances}")
    np.savez_compressed(args.out_distances, distance=dist_all)
    return dist_all


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    if os.path.isfile(args.out_distances):
        logging.info(f"Pairwise distances already computed: {args.out_distances}")
        logging.info(f"(Re)move the file if you want to compute them again.")
        exit(0)

    # Read the dataset
    dataset = pd.read_csv(args.tcrs, low_memory=False)
    mem_gb = (len(dataset) ** 2) / 1024 / 1024 / 1024
    logging.info(f"Computing pairwise distances for {len(dataset)} observations")
    logging.info(
        f"Memory required: at least 2.5 x {mem_gb:.2}Gb (parallel results, "
        f"square dist matrix, condensed dist matrix)"
    )

    # Compute distances
    pairwise_cdr3b_distances(dataset, args)
