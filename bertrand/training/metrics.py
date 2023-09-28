from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
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
    f1s = []
    accuracies = []
    pr_aucs = []
    counts = []
    peptide_clusters = []

    for peptide_cluster in np.unique(peptide_clusters_subset):
        peptide_cluster_mask = peptide_clusters_subset == peptide_cluster
        labels_pep = labels[peptide_cluster_mask]
        if len(set(labels_pep)) == 2:
            roc = roc_auc_score(labels_pep, probs[peptide_cluster_mask])
            f1 = f1_score(labels_pep, probs[peptide_cluster_mask] > 0.5)
            accuracy = accuracy_score(labels_pep, probs[peptide_cluster_mask] > 0.5)
            pr_auc = average_precision_score(labels_pep, probs[peptide_cluster_mask])
            rocs.append(roc)
            f1s.append(f1)
            accuracies.append(accuracy)
            pr_aucs.append(pr_auc)
            count = (labels_pep == 1).sum()
            counts.append(count)
            peptide_clusters.append(peptide_cluster)

    if agg:
        return {
            "roc": np.mean(rocs),
            "roc_std": np.std(rocs),
            "f1": np.mean(f1s),
            "f1_std": np.std(f1s),
            "accuracy": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "pr_auc": np.mean(pr_aucs),
            "pr_auc_std": np.std(pr_aucs),
        }
    return pd.DataFrame(data={
        "rocs": rocs,
        "f1s": f1s,
        "accuracies": accuracies,
        "pr_aucs": pr_aucs,
        "n": counts,
        "peptide": peptide_clusters,
    })
