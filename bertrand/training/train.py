import argparse
import glob
import logging
import os

import pandas as pd
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, BertForMaskedLM

from bertrand.training.dataset import PeptideTCRDataset
from bertrand.training.metrics import mean_auroc_per_peptide_cluster
from bertrand.training.config import BERT_CONFIG, SUPERVISED_TRAINING_ARGS
from bertrand.model.model import BERTrand
from bertrand.model.tokenization import tokenizer


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script performs cross-validation for BERTrand"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to peptide:TCR dataset",
    )

    parser.add_argument(
        "--model-ckpt",
        type=str,
        default=None,
        help="Path to model pre-trained checkpoint (omit for random init)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save weights and predictions.\n"
        "A subdirectory will be created for every csv.gz file in `input-dir`, "
        "every CV split will be in a separate subdirectory.",
    )

    parser.add_argument(
        "--n-splits", type=int, default=21, help="Number of CV splits",
    )

    return parser.parse_args()


def get_training_args(output_dir: str) -> TrainingArguments:
    """
    Returns pytorch-lightning training args.
    :param output_dir: folder to save checkpoints
    :return: training args
    """
    training_args = TrainingArguments(
        output_dir=output_dir, **SUPERVISED_TRAINING_ARGS,
    )
    return training_args


def train_and_evaluate(
    train_dataset: PeptideTCRDataset,
    val_dataset: PeptideTCRDataset,
    model_class,
    model_ckpt: str,
    output_dir: str,
) -> None:
    """
    Trains and evaluates the model.
    Returns predictions for the whole `val_dataset`, but computes metrics only for `split=='val'`,
    :param train_dataset: training set for the model
    :param val_dataset: validation and tests sets
    :param model_class: model class
    :param model_ckpt: model checkpoint (see train_mlm.py). if None, then weights are initialized randomly
    :param output_dir: folder to save model checkpoints and predictions for `val_dataset` for every epoch
    """
    predictions = []
    logging.info(f"Model class: {model_class}")

    def compute_metrics_and_save_predictions(p):
        predictions.append(p)
        return mean_auroc_per_peptide_cluster(
            p,
            val_dataset.examples.peptide_seq,
            val_dataset.examples.split == "val",
            True,
        )

    if model_ckpt:
        logging.info(f"Loading model from {model_ckpt}")
        model = model_class.from_pretrained(model_ckpt)
    else:
        logging.info("Initializing model from scratch")
        model = model_class(BERT_CONFIG)

    training_args = get_training_args(output_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_and_save_predictions,
    )

    trainer.train()
    pd.to_pickle(predictions, os.path.join(output_dir, "predictions.pkl"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    for dataset_fn in sorted(glob.glob(os.path.join(args.input_dir, "*.csv.gz"))):
        dataset = pd.read_csv(dataset_fn)
        dataset_name = os.path.basename(dataset_fn).replace(".csv.gz", "")
        logging.info(f"Dataset {dataset_name}, {len(dataset)} samples")
        for cv_seed in range(42, 42 + args.n_splits):
            logging.info(f"CV seed={cv_seed}")
            dataset_out_dir = os.path.join(
                args.output_dir, dataset_name, f"cv_seed={cv_seed}"
            )
            logging.info(f"Saving weights and predictions to {dataset_out_dir}")
            if os.path.isdir(dataset_out_dir):
                logging.info(f"Directory {dataset_out_dir} already exists, skipping")
                continue
            logging.info("Splitting the dataset")
            train_dataset = PeptideTCRDataset(dataset, cv_seed=cv_seed, subset="train")
            val_dataset = PeptideTCRDataset(dataset, cv_seed=cv_seed, subset="val+test")
            logging.info("Training started")
            model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
            train_and_evaluate(
                train_dataset, val_dataset, model, args.model_ckpt, dataset_out_dir,
            )
