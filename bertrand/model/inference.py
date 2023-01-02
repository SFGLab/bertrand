import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch

from bertrand.model.model import BERTrand
from bertrand.training.dataset import PeptideTCRDataset
from training.evaluate import get_trainer


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script performs inference using a previously trained model"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to dataset",
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to model checkpoint",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output file with predictions",
    )

    return parser.parse_args()


def single_obs_to_tensor(input: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    Converts a single observation to a tensor with a single observation
    :param input: dictionary of the observation - input_ids, token_type_ids, attention_mask,
    :return: dictionary with tensors
    """
    return {k: torch.tensor([v]) for k, v in input.items()}


def get_single_prediction(model: BERTrand, pep: str, cdr3b: str) -> float:
    """
    Inference for a single peptide:TCR observation
    :param model: BERTrand model
    :param pep: peptide amino acid sequence
    :param cdr3b: TCR amino acid sequence
    :return:
    """
    input = PeptideTCRDataset.encode_peptide_cdr3b(pep, cdr3b)
    input = single_obs_to_tensor(input)
    output = model(**input, return_dict=False)
    logits = output[0]
    with torch.no_grad():
        probs = torch.nn.functional.softmax(logits, 1).numpy()[0, 1]
    return probs


def get_dataset_predictions(
    model: BERTrand, dataset: pd.DataFrame, batch_size: int = 64
) -> np.ndarray:
    """
    Inference for a set of peptide:TCR observations
    :param model: BERTrand model
    :param dataset: dataframe with `peptide_seq` and `CDR3b` columns
    :param batch_size: batch size for inference
    :return: array of predictions for every observation
    """
    dataset = PeptideTCRDataset(dataset, 0, subset="inference")
    trainer = get_trainer(model, dataset, batch_size=batch_size)
    predictions = trainer.predict(dataset)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(
            torch.tensor(predictions.predictions), 1
        ).numpy()[:, 1]
    return probs


if __name__ == "__main__":
    args = parse_args()
    dataset = pd.read_csv(args.input)

    model = BERTrand.from_pretrained(args.model)
    model.eval()

    probs = get_dataset_predictions(model, dataset)
    probs = pd.Series(probs)
    probs.columns = ["BERTrand"]
    probs.to_csv(args.output, index=False)
