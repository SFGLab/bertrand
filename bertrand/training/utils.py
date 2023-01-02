import json
import os
from glob import glob
from typing import Union

import pandas as pd


def load_metrics_df(ckpt: str) -> pd.DataFrame:
    """
    Loads the metrics from trainer state json
    :param ckpt: checkpoint folder
    :return: dataframe with train and eval metrics
    """
    with open(os.path.join(ckpt, "trainer_state.json"), "r") as fp:
        metrics_json = json.load(fp)
    metrics_df = pd.DataFrame(metrics_json["log_history"])
    return metrics_df


def get_last_ckpt(input_dir: str) -> Union[str, None]:
    """
    Return the last (in terms of steps) checkpoint from a folder
    :param input_dir: folder with checkpoints
    :return: last checkpoint
    """
    checkpoint_dirs_by_step = {
        int(os.path.basename(x).split("-")[1]): x
        for x in glob(os.path.join(input_dir, "checkpoint*"))
    }
    if len(checkpoint_dirs_by_step) == 0:
        return None
    last_checkpoint = checkpoint_dirs_by_step[max(checkpoint_dirs_by_step.keys())]
    return last_checkpoint


def compute_most_popular_pep(examples):
    most_popular_pep = ""
    peptides = examples.peptide_cluster.drop_duplicates().str.split("_").str[0]
    for pos in range(9):
        aa = peptides.str[pos].value_counts().index[0]
        most_popular_pep += aa
    return most_popular_pep