import os
import shutil
from copy import deepcopy
from glob import glob
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput

from bertrand.training.config import SUPERVISED_TRAINING_ARGS
from bertrand.training.dataset import PeptideTCRDataset
from bertrand.training.metrics import mean_auroc_per_peptide_cluster
from bertrand.model.tokenization import tokenizer
from bertrand.training.prot_bert import ProteinClassifier
from bertrand.training.utils import get_last_ckpt, load_metrics_df

import argparse
import logging


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script evaluates the results of training\n"
        "For every cross-validation split, \n"
        "best model according to val AUROC is selected, \n"
        "it's performance on test and cancer sets is reported"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True, help="Path to `train.py` results",
    )
    parser.add_argument(
        "--datasets-dir", type=str, required=True, help="Path to datasets",
    )

    parser.add_argument(
        "--out", type=str, required=True, help="Path to output file",
    )
    return parser.parse_args()


def get_trainer(
    model: ProteinClassifier, test_dataset: PeptideTCRDataset, batch_size: int = 512
) -> Trainer:
    """
    Creates a Trainer for the model to do inference on a dataset
    :param model: BERTrand mode;
    :param test_dataset: dataset for inference
    :param batch_size: batch size for inference
    :return: pytorch-lightning Trainer object
    """
    args = deepcopy(SUPERVISED_TRAINING_ARGS)
    args["per_device_eval_batch_size"] = batch_size
    training_args = TrainingArguments(output_dir="./", **args)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    return trainer


def get_predictions(
    model: ProteinClassifier, test_dataset: PeptideTCRDataset
) -> PredictionOutput:
    """
    Performs inference on a dataset
    :param model: model for inference
    :param test_dataset: dataset for inference
    :return: model predictions
    """
    trainer = get_trainer(model, test_dataset)
    predictions = trainer.predict(test_dataset)
    return predictions


def evaluate_cancer(cancer_dataset: PeptideTCRDataset, ckpt: str) -> pd.DataFrame:
    """
    Evaluate the model on an independent cancer set
    :param cancer_dataset: peptide:TCR dataset for cancer peptides
    :param ckpt: model checkpoint folder
    :return: dataframe with AUROC for every peptide
    """
    model = ProteinClassifier.from_pretrained(ckpt)
    model.eval()

    predictions = get_predictions(model, cancer_dataset)

    cancer_res = mean_auroc_per_peptide_cluster(
        predictions,
        cancer_dataset.examples.peptide_seq,
        cancer_dataset.examples.split == "cancer",
        agg=False,
    )

    return cancer_res


def metrics_per_epoch(
    pred_list: List[PredictionOutput],
    val_dataset: PeptideTCRDataset,
    subset: str,
    metric: str = "rocs",
    peptide_col: str = "peptide_seq",
) -> pd.DataFrame:
    """
    Computes AUROC for every peptide for every epoch
    :param pred_list: list of predictions for every epoch
    :param val_dataset: dataset to evaluate on
    :param subset: subset of the dataset (can be `val` or `test`)
    :param peptide_col: name of the column for peptide group (can be `peptide_seq` or `peptide_cluster`)
    :return: dataframe of AUROC values of shape (n_epochs, n_peptides)
    """
    rocs_list = []
    n = None
    for i, predictions in enumerate(pred_list):
        epoch_results_val = mean_auroc_per_peptide_cluster(
            predictions,
            val_dataset.examples[peptide_col],
            val_dataset.examples.split == subset,
            agg=False,
        )
        n = epoch_results_val.set_index("peptide").n
        epoch_results_val = epoch_results_val.set_index("peptide")[metric]
        epoch_results_val.name = i
        rocs_list.append(epoch_results_val)

    results_val = pd.concat(rocs_list, axis=1)
    results_val["n"] = n
    results_val = (
        results_val.reset_index()
        .sort_values(by="n", ascending=False)
        .set_index(["peptide", "n"])
    )
    return results_val


def aggregate_metrics_per_epoch(
    metrics_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes AUROC average, weighted average and std for every epoch
    :param metrics_df: dataframe of AUROC values of shape (n_epochs, n_peptides)
    :return: average, weighted average and std, each of `n_epochs` length
    """
    n = metrics_df.index.get_level_values("n")
    weighted_average = metrics_df.apply(
        lambda col: np.average(col, weights=np.log10(n)), axis=0
    )
    average = metrics_df.mean(axis=0)
    std = metrics_df.std(axis=0)
    return weighted_average, average, std


def plot_metrics_per_epoch(
    auroc_per_epoch_val: pd.DataFrame,
    auroc_average_val: pd.Series,
    auroc_weighted_average_val: pd.Series,
    auroc_average_test: pd.Series,
    auroc_weighted_average_test: pd.Series,
    best_epoch: int,
    best_epoch_result: pd.DataFrame,
) -> plt.Figure:
    """
    Plots average AUROC and per-peptide AUROC for validation and test sets, as well as the best epoch and final result
    :param auroc_per_epoch_val: AUROC per epoch for the validation set
    :param auroc_average_val: average AUROC for the validation set
    :param auroc_weighted_average_val: weighted average AUROC for the validation set
    :param auroc_average_test: average AUROC for the test set
    :param auroc_weighted_average_test: weighted average AUROC for the test set
    :param best_epoch: best epoch index
    :param best_epoch_result: aggregated test set result for best epoch
    :return: figure with the plot
    """
    fig, axs = plt.subplots(nrows=2)
    epochs = np.arange(auroc_per_epoch_val.shape[1])
    axs[0].plot(epochs, auroc_average_val, label="average")
    axs[0].plot(epochs, auroc_weighted_average_val, label="weighted average")
    axs[0].legend()

    auroc_per_epoch_val.T.plot(ax=axs[1])
    axs[1].plot(epochs, auroc_weighted_average_test, "r--", linewidth=3)
    axs[1].plot(epochs, auroc_average_test, "g--", linewidth=3)
    axs[1].set_title("Test ROC AUC per peptide")
    axs[1].set_xlabel("epochs")
    axs[1].set_ylabel("ROC AUC")

    axs[1].plot([best_epoch, best_epoch], [0, 1], "b--")
    axs[0].plot([best_epoch, best_epoch], [0, 1], "b--")

    axs[1].set_ylim(0.3, 1.05)
    axs[0].set_ylim(0.5, 0.75)

    roc = best_epoch_result.rocs.mean()
    roc_std = best_epoch_result.rocs.std()
    axs[1].text(best_epoch, 0.4, "Mean ROC AUC=%.3f+-%.3f" % (roc, roc_std))

    fig.set_size_inches(20, 12)
    return fig


def get_epochs_to_step(model_dir: str) -> Union[Dict[int, int], None]:
    """
    Returns a mapping of epochs to steps for model training.
    We evaluate the model once an epoch, however pytorch-lightning names the checkpoints
    according to the number of training steps.
    :param model_dir: folder with model checkpoints
    :return: a dictionary epoch->step
    """
    last_checkpoint = get_last_ckpt(model_dir)
    if not last_checkpoint:
        return None
    metrics = load_metrics_df(last_checkpoint)
    metrics_eval = metrics[~metrics.eval_loss.isna()].copy()
    metrics_eval.epoch = metrics_eval.epoch.astype(int) - 1
    metrics_eval.set_index("epoch", inplace=True)
    return metrics_eval.step.to_dict()


def evaluate_model(
    model_dir: str, dataset: pd.DataFrame, dataset_name: str, plot: bool = True
) -> Union[pd.DataFrame, None]:
    """
    This function evaluates the previously trained model on the test set and an independent cancer set
    Best epoch is chosen based on weighted average AUROC on the validation set
    :param model_dir: model folder with checkpoints
    :param dataset: peptide:TCR dataset
    :param dataset_name: human-readable dataset name
    :param plot: if True, will create a plot and save it to a file `metrics.png` in `model_dir`
    :return: a dataframe of AUROC values separated by peptide for the test and cancer sets
    """
    cv_seed = int(os.path.split(model_dir)[-1].replace("cv_seed=", ""))
    logging.info(f"Now evaluating {model_dir}")
    epochs_to_step = get_epochs_to_step(model_dir)
    if not epochs_to_step:
        logging.info("Skipping, no checkpoints found")
        return None

    val_test_dataset = PeptideTCRDataset(dataset, cv_seed, "val+test")
    cancer_dataset = PeptideTCRDataset(dataset, cv_seed, "cancer")

    pred_fn = os.path.join(model_dir, "predictions.pkl")
    if not os.path.isfile(pred_fn):
        logging.info("Skipping, no predictions found")
        return None
    pred_list = pd.read_pickle(pred_fn)

    rocs_val = metrics_per_epoch(pred_list, val_test_dataset, subset="val")
    weighted_average_val, average_val, std_val = aggregate_metrics_per_epoch(rocs_val)

    rocs_test = metrics_per_epoch(pred_list, val_test_dataset, subset="test")
    weighted_average_test, average_test, std_test = aggregate_metrics_per_epoch(
        rocs_test
    )

    best_epoch = weighted_average_val.idxmax()
    best_predictions = pred_list[best_epoch]
    best_epoch_steps = epochs_to_step[best_epoch]
    best_epoch_checkpoint = os.path.join(model_dir, f"checkpoint-{best_epoch_steps}")

    test_res = mean_auroc_per_peptide_cluster(
        best_predictions,
        val_test_dataset.examples.peptide_seq,
        val_test_dataset.examples.split == "test",
        agg=False,
    )
    test_res = test_res.assign(cv_seed=cv_seed, dataset=dataset_name, subset="test")

    cancer_res = evaluate_cancer(cancer_dataset, best_epoch_checkpoint)
    cancer_res = cancer_res.assign(
        cv_seed=cv_seed, dataset=dataset_name, subset="cancer"
    )

    if plot:
        fig = plot_metrics_per_epoch(
            rocs_test,
            average_val,
            weighted_average_val,
            average_test,
            weighted_average_test,
            best_epoch,
            test_res,
        )
        fig.savefig(os.path.join(model_dir, "metrics.png"), dpi=300)
        plt.close(fig)

    if plot:
        for metric in ["rocs", "f1s", "accuracies", "pr_aucs"]:
            metric_val = metrics_per_epoch(pred_list, val_test_dataset, subset="val", metric=metric)
            weighted_average_val, average_val, std_val = aggregate_metrics_per_epoch(metric_val)

            metric_test = metrics_per_epoch(pred_list, val_test_dataset, subset="test", metric=metric)
            weighted_average_test, average_test, std_test = aggregate_metrics_per_epoch(
                metric_test
            )

            epochs = np.arange(metric_test.shape[1])

            fig = plt.figure()
            ax = fig.gca()
            ax.plot(epochs, average_val, label="average")
            ax.plot(epochs, weighted_average_val, label="weighted average")
            ax.legend()
            fig.savefig(os.path.join(model_dir, metric + ".png"), dpi=300)


    best_ckpt_result_dir = os.path.join(model_dir, "best_checkpoint")
    logging.info(f"Copying {best_epoch_checkpoint} to {best_ckpt_result_dir}")
    shutil.copytree(best_epoch_checkpoint, best_ckpt_result_dir, dirs_exist_ok=True)

    return pd.concat([test_res, cancer_res])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    results = []
    for dataset_fn in sorted(glob(os.path.join(args.datasets_dir, "*.csv.gz"))):
        dataset_name = os.path.basename(dataset_fn).replace(".csv.gz", "")
        logging.info(f"Dataset {dataset_name}")
        dataset_results_dir = os.path.join(args.results_dir, dataset_name)
        dataset = pd.read_csv(dataset_fn, low_memory=False)

        for model_dir in sorted(glob(os.path.join(dataset_results_dir, "cv_seed=*"))):
            model_results = evaluate_model(model_dir, dataset, dataset_name, plot=True)
            if model_results is not None:
                results.append(model_results)

    results = pd.concat(results)
    print(results.groupby(["dataset", "subset"]).rocs.mean())
    results.to_csv(args.out, index=False)
