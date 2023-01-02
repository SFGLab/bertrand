import argparse
import logging
import os
from glob import glob

import pandas as pd
from transformers import (
    Trainer,
    TrainingArguments,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
)

from bertrand.training.config import BERT_CONFIG, MLM_TRAINING_ARGS
from bertrand.pretraining.dataset_mlm import PeptideTCRMLMDataset
from bertrand.model.tokenization import tokenizer


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script pretrains BERT neural network using Masked Language Modeling"
    )
    parser.add_argument(
        "--train", type=str, required=True, help="Path to training set",
    )
    parser.add_argument(
        "--val", type=str, required=True, help="Path to validation set",
    )

    parser.add_argument(
        "--out-dir", type=str, required=True, help="Path to save model checkpoints",
    )

    parser.add_argument(
        "--mlm-frac", type=float, default=0.1, help="Fraction of masked tokens",
    )

    return parser.parse_args()


def get_training_args(output_dir: str) -> TrainingArguments:
    """
    Returns pytorch-lightning training args.
    :param output_dir: folder to save checkpoints
    :return: training args
    """
    training_args = TrainingArguments(
        output_dir=output_dir, **MLM_TRAINING_ARGS,  # output directory
    )
    return training_args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    output_dir = os.path.join(args.out_dir, "checkpoints")
    if os.path.isdir(output_dir) and len(glob(os.path.join(output_dir, "**/"))) > 0:
        logging.info(
            f"Some checkpoints are present in {output_dir}, (re)move them or set another out-dir"
        )
        exit(0)
    train = pd.read_csv(args.train, index_col=0)
    val = pd.read_csv(args.val, index_col=0)
    train_dataset = PeptideTCRMLMDataset(train)
    val_dataset = PeptideTCRMLMDataset(val)

    model = BertForMaskedLM(BERT_CONFIG)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_frac
    )
    training_args = get_training_args(output_dir)

    trainer = Trainer(
        model=model,  # BERT for MLM
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        data_collator=data_collator,  # data collator
    )

    trainer.train()
