import argparse
import os

import numpy as np
import pandas as pd
from leven import levenshtein

from bertrand.negative_decoys.compute_distance import cdist_parallel
import logging


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script performs basic reference TCRs filtering for the purpose of peptide:TCR "
        "negative decoys generation"
    )
    parser.add_argument(
        "--binders",
        type=str,
        required=True,
        help="Path to peptide:TCR binders dataset (.csv.gz)",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        required=True,
        help="Path to TCR repertoire files (directory)",
    )
    parser.add_argument(
        "--out-distances",
        type=str,
        required=True,
        help="Path to output reference-positive distance matrix  (.npz)",
    )
    parser.add_argument(
        "--out-tcrs",
        type=str,
        required=True,
        help="Path to output positive&reference TCR dataset (.csv.gz)",
    )
    parser.add_argument("--cpu", type=int, default=-1, help="Number of CPUs available")
    return parser.parse_args()


def read_positives_and_group_by_tcr(filename: str) -> pd.DataFrame:
    """
    Reads the dataframe of binding peptide:TCR pairs,
    groups them by CDR3b,
    and creates a dataframe of positive TCRs
    :param filename: path to binders dataset, created by EDA.ipynb
    :return: positive TCRs - pd.DataFrame with `CDR3b` and `y` columns, `y` equal to 1
    """
    logging.info(f"Reading binders from {filename}")
    df = pd.read_csv(filename, low_memory=False)
    df.loc[:, "y"] = 1
    positive_cdr3b = df.CDR3b.drop_duplicates().reset_index(drop=True)
    peptide_info = df.groupby("CDR3b").agg(
        {
            "peptide_seq": lambda p: "|".join(sorted(set(list(p)))),
            "peptide_cluster": lambda p: "|".join(sorted(set(list(p)))),
        }
    )
    peptide_info = peptide_info.loc[positive_cdr3b]
    positive_tcrs_df = peptide_info.reset_index()
    positive_tcrs_df.loc[:, "y"] = 1
    logging.info(
        f"{len(df)} observations, {len(positive_tcrs_df)} unique CDR3b sequences"
    )
    return positive_tcrs_df


def read_reference_tcrs(input_dir: str) -> pd.Series:
    """
    Reads healthy donors' TCRs from Oakes et al. 2017
    :param input_dir: folder with files
    :return: CDR3 beta sequences of CD8+ TCRs
    """
    logging.info(f"Reading reference TCRs from {input_dir}")
    neg_tcrs = []
    for filename in sorted(os.listdir(input_dir)):
        filename = os.fsdecode(filename)
        if filename.endswith(".cdr3.gz") and "beta" in filename and "CD8" in filename:
            df = pd.read_csv(input_dir + "/" + filename, header=None)
            neg_tcrs.append(df.iloc[:, 0])
    return pd.concat(neg_tcrs).reset_index(drop=True)


def basic_filtering(
    reference_tcrs: pd.Series, positive_tcrs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Basic filtering of reference TCRs:
    1. TCRs identical to any binding TCR are removed,
    2. TCRs shorted than 10 or longer than 20 are removed,
    3. TCRs that do not start with Cysteine are removed,
    4. Duplicated TCRs are removed.
    :param reference_tcrs: reference CDR3b sequences
    :param positive_tcrs_df: positive TCRs from `read_positives_and_group_by_tcr`
    :return: filtered reference TCRs with `CDR3b` and `y` columns, `y` equal to 0
    """
    before = len(reference_tcrs)
    reference_tcrs = reference_tcrs[~reference_tcrs.isin(positive_tcrs_df.CDR3b)]
    reference_tcrs = reference_tcrs[reference_tcrs.str.len().isin(np.arange(10, 21))]
    reference_tcrs = reference_tcrs[reference_tcrs.str[0] == "C"]
    reference_tcrs = reference_tcrs.drop_duplicates().reset_index(drop=True)
    reference_tcrs_df = pd.DataFrame(data={"CDR3b": reference_tcrs, "y": 0.0})
    logging.info(
        f"Basic filtering: started with {before} reference TCRs, {len(reference_tcrs_df)} after filtering"
    )
    return reference_tcrs_df


def compute_reference_pos_distance_or_read_cache(
    reference_tcrs_df: pd.DataFrame,
    positive_tcrs_df: pd.DataFrame,
    distance_matrix_fn: str,
    n_cpus: int = -1,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Computes Levenshtein distances between all reference and positive TCRs.
    This distance computation is parallelized into `batch_size x len(positive_tcrs_df)` chunks
    This computation is quite memory-intensive.
    :param reference_tcrs_df: reference TCRs
    :param positive_tcrs_df: positive TCRs
    :param distance_matrix_fn: path to save the distance matrix
    :param n_cpus: number of CPUs available
    :param batch_size: size of the chunk
    :return: reference-positive distance matrix of shape=(len(reference_tcrs_df), len(positive_tcrs_df))
    """
    if not os.path.isfile(distance_matrix_fn):
        logging.info(
            f"Computing {len(reference_tcrs_df)} by {len(positive_tcrs_df)} distance matrix"
        )
        dist_neg_pos = cdist_parallel(
            reference_tcrs_df.CDR3b.values,
            positive_tcrs_df.CDR3b.values,
            levenshtein,
            bsx=batch_size,
            n_jobs=n_cpus,
            cache_fn=None,
        )
        logging.info(f"Saving the distance matrix to {distance_matrix_fn}")
        np.savez_compressed(distance_matrix_fn, distance=dist_neg_pos)

    else:
        logging.info(f"Loading the distance matrix from {distance_matrix_fn}")
        loaded = np.load(distance_matrix_fn)
        dist_neg_pos = loaded.get("distance")
    return dist_neg_pos


def min_pos_distance_filtering(
    reference_tcrs_df: pd.DataFrame,
    positive_tcrs_df: pd.DataFrame,
    dist_neg_pos: np.ndarray,
) -> pd.DataFrame:
    """
    Computes minimum positive distance, average positive distance and
    closest positive TCR for every reference TCR.
    Removes TCRs with 1 amino acid difference to any positive TCR.
    :param reference_tcrs_df: reference TCRs dataframe
    :param positive_tcrs_df: positive TCRs dataframe
    :param dist_neg_pos: reference-positive distance matrix
    :return: copy of reference TCRs dataframe with additional `min_pos_dist`,
        `avg_pos_dist` and `closest_pos_CDR3b` columns.
        TCRs with `min_pos_dist` == 1 are removed.
    """
    logging.info(f"Computing minimum positive distance")
    reference_tcrs_df = reference_tcrs_df.copy()
    min_dist = dist_neg_pos.min(axis=1)
    avg_dist = dist_neg_pos.mean(axis=1)
    closest_pos = dist_neg_pos.argmin(axis=1)

    reference_tcrs_df.loc[:, "min_pos_dist"] = min_dist
    reference_tcrs_df.loc[:, "avg_pos_dist"] = avg_dist
    reference_tcrs_df.loc[:, "closest_pos_CDR3b"] = positive_tcrs_df.iloc[
        closest_pos
    ].CDR3b.values
    reference_tcrs_df_filtered = reference_tcrs_df[
        reference_tcrs_df["min_pos_dist"] > 1
    ]
    n_before = len(reference_tcrs_df)
    n_after = len(reference_tcrs_df_filtered)
    logging.info(
        f"Min positive distance filtering: started with {n_before} reference TCRs, {n_after} after filtering"
    )
    return reference_tcrs_df_filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # Read data
    positive_tcrs_df = read_positives_and_group_by_tcr(args.binders)
    reference_tcrs = read_reference_tcrs(args.reference_dir)

    # Do basic filtering and concat
    reference_tcrs_df = basic_filtering(reference_tcrs, positive_tcrs_df)
    tcr_df = pd.concat([positive_tcrs_df, reference_tcrs_df], axis=0).reset_index(
        drop=True
    )

    # Remove reference TCRs different by 1 amino acid from any binding TCR
    dist_neg_pos = compute_reference_pos_distance_or_read_cache(
        reference_tcrs_df, positive_tcrs_df, args.out_distances
    )
    reference_tcrs_df_filtered = min_pos_distance_filtering(
        reference_tcrs_df, positive_tcrs_df, dist_neg_pos
    )
    tcr_df_curated = pd.concat([positive_tcrs_df, reference_tcrs_df_filtered], axis=0)

    # Save
    logging.info(f"Saving result to {args.out_tcrs}")
    tcr_df_curated.to_csv(args.out_tcrs, index=False)
