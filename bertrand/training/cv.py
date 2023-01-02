from typing import Tuple

import numpy as np
import pandas as pd

from bertrand.training.config import DEFAULT_TEST_SPLIT, DEFAULT_VAL_SPLIT


def filter_cancer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Observations from cancer peptides are removed,
    because they constitute an independent test set
    :param df: peptide:TCR dataset
    :return: dataset without cancer observations
    """
    df_cancer = df[df.is_cancer]
    df = df[~df.is_cancer & ~df.tcr_cluster.isin(df_cancer.tcr_cluster)]
    return df


def split_peptide_clusters(
    df: pd.DataFrame, random_state: int, top: int = 2, mid: int = 5, low: int = 7
) -> Tuple[pd.Series, pd.Series]:
    """
    Splits peptide clusters in the dataset into 2 sets - train and validation.
    The split is balanced so that the validation set has some observations
    for more popular peptide clusters and for less popular ones as well.
    Peptide clusters with less than 10 observations are not eligible for the validation set.
    :param df: peptide:TCR dataset
    :param random_state: random state for the split
    :param top: number of peptide clusters with over 1000 observations in the validation set
    :param mid: number of peptide clusters with 100 to 1000 observations in the validation set
    :param low: number of peptide clusters with 10 to 100 observations in the validation set
    :return: train peptide clusters, validation peptide clusters
    """
    df = filter_cancer_dataset(df)

    train_val_peptides = pd.Series(df["peptide_cluster"].unique())

    n_binders_per_peptide = (
        df.y.groupby(df.peptide_cluster)
        .agg("sum")
        .sort_values(ascending=False, kind="stable")
    )
    possible_val_peptides = n_binders_per_peptide[
        (n_binders_per_peptide >= 10)
        & (n_binders_per_peptide.index.isin(train_val_peptides))
    ]

    top_val_peps = possible_val_peptides[possible_val_peptides >= 1000].sample(
        top, random_state=random_state
    )
    mid_val_peps = possible_val_peptides[
        (possible_val_peptides < 1000) & (possible_val_peptides >= 100)
    ].sample(mid, random_state=random_state)
    low_val_peps = possible_val_peptides[
        (possible_val_peptides > 10) & (possible_val_peptides < 100)
    ].sample(low, random_state=random_state)
    val_peptides = pd.concat([top_val_peps, mid_val_peps, low_val_peps])
    val_peptides = pd.Series(val_peptides.index)

    train_peptides = pd.Series(np.setdiff1d(train_val_peptides, val_peptides))
    return train_peptides, val_peptides


def split_by_peptide_tcr_clusters(
    df: pd.DataFrame, seed: int, top: int = 2, mid: int = 5, low: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset by peptide and TCR clusters,
    i.e. no peptides or TCRs in the training set
    are in clusters with peptides or TCRs in the validation set.
    :param df: peptide:TCR dataset
    :param seed: random state for the split
    :param top: number of peptide clusters with over 1000 obs. (see split_peptide_clusters)`
    :param mid: number of peptide clusters with 100 to 1000 obs.
    :param low: number of peptide clusters with 10 to 100 obs.
    :return: train dataset, validation dataset
    """
    train_peptides, val_peptides = split_peptide_clusters(
        df, random_state=seed, top=top, mid=mid, low=low
    )

    train_mask = df.peptide_cluster.isin(train_peptides)
    val_mask = df.peptide_cluster.isin(val_peptides)

    # Remove train observations in any validation TCR clusters
    val_tcr_clusters = df.loc[val_mask, "tcr_cluster"].unique()
    train_mask = train_mask & (
        ~df.loc[train_mask, "tcr_cluster"].isin(val_tcr_clusters)
    )

    return df[train_mask], df[val_mask]


def split_train_val_test(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into 3 subsets:
    `train` (used for model training),
    `val` (used for early stopping), and
    `test` (used for model evaluation).
    The validation set has less (1+4+5) peptide clusters compared to the test set (2+5+7)
    :param df: peptide:TCR dataset
    :param seed: random state for the split
    :return: train, val and test sets
    """
    train_val, test = split_by_peptide_tcr_clusters(df, seed, **DEFAULT_TEST_SPLIT)
    train, val = split_by_peptide_tcr_clusters(train_val, seed, **DEFAULT_VAL_SPLIT)
    return train, val, test
