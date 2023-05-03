from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import EvalPrediction
from transformers.trainer_utils import PredictionOutput


def mean_auroc_per_peptide_cluster(
    eval_prediction: Union[EvalPrediction, PredictionOutput],
    peptide_clusters: pd.Series,
    subset_mask: np.ndarray,
    agg: bool = True,
) -> Union[Dict, pd.DataFrame]:
    """
    Calculates mean and AUROC per peptide cluster for a subset of observations
    :param eval_prediction: model predictions
    :param peptide_clusters: peptide clusters
    :param subset_mask: subset mask
    :param agg: if False, does not aggregate per peptide
    :return: dictionary with `roc` and `roc_std` - std of AUROC per peptide
    """
    preds = eval_prediction.predictions
    probs = torch.nn.functional.softmax(torch.tensor(preds), 1).numpy()[:, 1]
    labels = eval_prediction.label_ids
    return mean_auroc_per_peptide_cluster_probs(
        labels, probs, peptide_clusters, subset_mask, agg
    )


def mean_auroc_per_peptide_cluster_probs(
    labels: np.ndarray,
    probs: np.ndarray,
    peptide_clusters: pd.Series,
    subset_mask: np.ndarray,
    agg: bool = True,
) -> Union[Dict, pd.DataFrame]:
    """
    Calculates mean and AUROC per peptide cluster for a subset of observations
    This version uses raw labels and probs instead of `EvalPrediction`
    :param labels: model targets
    :param probs: model predictions
    :param peptide_clusters: peptide clusters
    :param subset_mask: subset mask
    :param agg: if False, does not aggregate per peptide
    :return: dictionary with `roc` and `roc_std` - std of AUROC per peptide
    """
    labels = labels[subset_mask]
    probs = probs[subset_mask]
    peptide_clusters_subset = peptide_clusters[subset_mask]
    rocs = []
    aps = []
    counts = []
    peptide_clusters = []

    for peptide_cluster in np.unique(peptide_clusters_subset):
        peptide_cluster_mask = peptide_clusters_subset == peptide_cluster
        labels_pep = labels[peptide_cluster_mask]
        if len(set(labels_pep)) == 2:
            probs_pep = probs[peptide_cluster_mask]
            roc = roc_auc_score(labels_pep, probs_pep)
            ap = average_precision_score(labels_pep, probs_pep)

            rocs.append(roc)
            aps.append(ap)
            count = (labels_pep == 1).sum()
            counts.append(count)
            peptide_clusters.append(peptide_cluster)

    if agg:
        return {"roc": np.mean(rocs), "roc_std": np.std(rocs), "ap": np.mean(aps), "ap_std": np.std(aps)}
    return pd.DataFrame(data={"rocs": rocs, "aps": aps, "n": counts, "peptide": peptide_clusters,})
