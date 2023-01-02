import argparse
import logging
import os

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script performs negative decoys generation for filtered and clustered reference TCRs. "
    )
    parser.add_argument(
        "--tcrs",
        type=str,
        required=True,
        help="Path to filtered and clustered positive&reference TCRs dataset",
    )

    parser.add_argument(
        "--binders", type=str, required=True, help="Path to positive TCRs dataset",
    )

    parser.add_argument(
        "--out-dir", required=True, type=str, help="Path to output folder",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Sampling is repeated for a number of different seeds.",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=3.0,
        help="Ratio of negative to positive examples per peptide.",
    )

    return parser.parse_args()


def generate_negative_decoys(
    dataset: pd.DataFrame, peptide_count: pd.Series, seed: int
) -> pd.DataFrame:
    """
    Generates negative observations by randomly pairing
    peptides from the binders peptide:TCR dataset and
    whole TCR clusters of reference TCRs
    :param dataset: positive&reference TCR dataset
    :param peptide_count: required number of negative TCRs for each peptide
    :param seed: np.random.seed for reproducibility
    :return: a copy of `dataset` with `peptide_seq` column filled with randomly assigned peptides for reference TCRs
    """
    np.random.seed(seed)
    dataset = dataset.copy()
    negative_clusters = dataset[(dataset.y == 0)].tcr_cluster.unique()

    assigned_clusters_mask = (dataset.y == 0) & (~dataset.peptide_seq.isna())
    used_clusters = dataset[assigned_clusters_mask].tcr_cluster.unique()
    negative_clusters = np.setdiff1d(negative_clusters, used_clusters)
    positive_clusters = dataset[(dataset.y == 1)].tcr_cluster.unique()
    print(len(used_clusters), len(positive_clusters))
    negative_clusters = np.setdiff1d(negative_clusters, positive_clusters)
    cluster_gb = dataset.groupby("tcr_cluster")

    negative_assigned_peptide_counts = dataset[
        assigned_clusters_mask
    ].peptide_seq.value_counts()

    peptide_actual_count = negative_assigned_peptide_counts.reindex(
        peptide_count.index
    ).fillna(0)
    count_diff = (peptide_count - peptide_actual_count).astype(int)

    negative_clusters_current = list(negative_clusters)

    for peptide, n_to_sample in zip(count_diff.index, count_diff):
        if n_to_sample <= 0:
            continue

        negative_clusters_current = assign_random_negative_clusters(
            dataset, peptide, n_to_sample, negative_clusters_current, cluster_gb
        )

        # print(len(negative_clusters_current), "clusters left")
    return dataset


def assign_random_negative_clusters(
    dataset: pd.DataFrame,
    peptide: str,
    n_to_sample: int,
    negative_clusters: np.ndarray,
    cluster_gb: pd.core.groupby.DataFrameGroupBy,
) -> np.ndarray:
    """
    Assigns random reference TCR clusters to `peptide` until `n_to_sample` number of
    negative peptide:TCR observations is reached.
    Cluster assignment prevents correlated observations,
    i.e. TCRs in the same cluster with different peptides.
    This method modifies the `peptide_seq` column for the assigned TCRs.
    :param dataset: TCR dataset
    :param peptide: this peptide
    :param n_to_sample: number of TCRs
    :param negative_clusters: array of available negative clusters
    :param cluster_gb: TCR dataset grouped by clusters
    :return: an updated array of available negative
    """
    count_sampled = 0
    # print(peptide, n_to_sample, "to sample")
    while count_sampled < n_to_sample:
        if len(negative_clusters) == 0:
            raise Exception("all sampled!")
        ii = np.random.randint(0, len(negative_clusters))
        negative_clusters_sample = negative_clusters.pop(ii)
        cluster_df = cluster_gb.get_group(negative_clusters_sample)
        assert (cluster_df.y == 0).all()
        dataset.loc[cluster_df.index, "peptide_seq"] = peptide
        count_sampled += len(cluster_df)
    return negative_clusters


def get_negative_observations(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Return negative peptide:TCR observations
    :param dataset: TCR dataset with random peptides assigned for some TCRs
    :return: a subset of reference TCRs with `peptide_seq` assigned
    """
    return dataset[~dataset.peptide_seq.isna() & (dataset.y == 0)].copy()


def assign_tcr_cluster_for_binders(
    binders: pd.DataFrame, dataset: pd.DataFrame
) -> None:
    """
    Assigns TCR cluster from positive&reference TCR dataset for peptide:TCR binders dataset
    :param binders: peptide:TCR binders dataset
    :param dataset: positive&reference TCR dataset
    Adds `tcr_cluster` column to `binders`.
    """
    tcr_cluster_for_positives = dataset.query("y==1").set_index("CDR3b").tcr_cluster
    binders.loc[:, "tcr_cluster"] = tcr_cluster_for_positives.loc[binders.CDR3b].values


def get_n_negatives_per_peptide(binders: pd.DataFrame, ratio: float) -> pd.Series:
    """
    Calculates how many negative examples are needed for every peptide
    for a given negative to positive ratio
    :param binders: dataset of peptide:TCR binders
    :param ratio: negative to positive observations ratio
    :return: Number of required negative observations for every peptide in `binders`
    """
    peptide_count = (binders.peptide_seq.value_counts() * ratio).round().astype(int)
    return peptide_count


def check_n_negatives_per_peptide(peptide_count, dataset):
    """
    Checks if the number of requested negatives examples 
    is smaller than the number of reference TCRs
    :param peptide_count: number of required negative observation per peptide
    :param dataset: dataset of reference TCRs
    """
    total = peptide_count.sum()
    n_reference = (dataset.y == 0).sum()
    logging.info(f"{total} negative observations to generate")
    if total > n_reference:
        logging.error(
            f"Number of total reference TCRs {n_reference} is less than the number of obs. required to generate {total}"
        )
        raise Exception(
            "Too much negative observations requested. Maybe try lower ratio"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logging.info("Reading input data")
    binders = pd.read_csv(args.binders)
    binders["y"] = 1
    dataset = pd.read_csv(args.tcrs)

    assign_tcr_cluster_for_binders(binders, dataset)
    peptide_count = get_n_negatives_per_peptide(binders, args.ratio)
    check_n_negatives_per_peptide(peptide_count, dataset)

    # Peptide info will be added to negative observations
    peptide_info = binders.groupby("peptide_seq").agg(
        {"peptide_source": "first", "peptide_cluster": "first", "is_cancer": "first"}
    )

    for seed in range(42, 42 + args.n_seeds):
        logging.info(f"Generating negative decoys seed={seed}")

        dataset_sampled = generate_negative_decoys(dataset, peptide_count, seed)
        sampled_negatives = get_negative_observations(dataset_sampled)
        sampled_negatives_pepinfo = pd.merge(
            sampled_negatives.drop(columns=["peptide_cluster"]),
            peptide_info,
            left_on="peptide_seq",
            right_index=True,
        )

        ds = pd.concat([binders, sampled_negatives_pepinfo], axis=0)
        ds.index = np.arange(len(ds))

        fn = os.path.join(args.out_dir, f"dataset_{seed}.csv.gz")
        logging.info(f"Saving {fn}")
        ds.to_csv(fn, index=False)
