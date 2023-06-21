import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from bertrand.training.cv import split_train_val_test
from bertrand.model.tokenization import tokenizer
from typing import Dict, Literal


class PeptideTCRDataset(Dataset):
    """
    This class represents peptide:TCR dataset for model training
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        cv_seed: int,
        subset: Literal[
            "train", "val", "test", "cancer", "val+test", "val+test+cancer", "inference"
        ],
    ):
        """
        Class constructor
        :param dataset: peptide:TCR dataset
        :param cv_seed: cross-validation split random state
        :param subset: one of  ['train', 'val', 'test', 'cancer', 'val+test']
        """
        if subset == "inference":
            self.examples = dataset.copy().drop(columns=["y"])
        else:
            self.examples = self.split_dataset(cv_seed, dataset, subset)
            self.calc_weights()

    def split_dataset(self, cv_seed: int, dataset: pd.DataFrame, subset: str):
        """
        Produces a desired dataset split
        :param cv_seed: cross-validation seed
        :param dataset: peptide:TCR dataset
        :param subset: one of ['train', 'val', 'test', 'val+test', 'cancer', 'val+test+cancer']
        :return: a subset of the dataset
        """
        if "subset" in dataset.columns:
            examples = dataset[dataset.subset.isin(subset.split("+"))].copy()
        else:
            if subset != "cancer":
                train, val, test = split_train_val_test(dataset, cv_seed)

            if subset == "train":
                examples = train.copy()
            elif subset == "val":
                examples = val.copy()
            elif subset == "test":
                examples = test.copy()
            elif subset == "val+test":
                val = val.copy()
                test = test.copy()
                val.loc[:, "split"] = "val"
                test.loc[:, "split"] = "test"
                examples = pd.concat([val, test])
            elif subset == "cancer":
                examples = dataset[dataset.is_cancer].copy()
                examples.loc[:, "split"] = "cancer"
            elif subset == "val+test+cancer":
                cancer = dataset[dataset.is_cancer].copy()
                cancer.loc[:, "split"] = "cancer"
                val = val.copy()
                test = test.copy()
                val.loc[:, "split"] = "val"
                test.loc[:, "split"] = "test"
                examples = pd.concat([val, test, cancer])
            else:
                raise NotImplementedError()
        # examples = examples.sample(512, random_state=42)
        examples.reset_index(drop=True, inplace=True)
        examples.y = examples.y.astype("int")
        return examples

    def calc_weights(self) -> None:
        """
        Calculates weights for peptide:TCR prediction problem.
        The weight of an observation depends on it's peptide and TCR abundance in the dataset
        Popular peptide and TCR clusters are downweighted
        """
        # pep_count = self.examples.peptide_cluster.value_counts()
        # pep_w = pep_count.loc[self.examples.peptide_cluster].values
        # # 1 / pep_w gives very low weights for popular peptide clusters
        # pep_w = 1 / np.log(2 + pep_w)

        tcr_count = self.examples.tcr_cluster.value_counts()
        tcr_w = tcr_count.loc[self.examples.tcr_cluster].values
        # 1 / tcr_w gives very low weights for popular TCR clusters
        tcr_w = 1 / np.log(2 + tcr_w)  #

        weights = tcr_w
        self.examples.loc[:, "weight"] = weights

        # from matplotlib import pyplot as plt
        # plt.scatter(pep_w[self.examples.y == 0], tcr_w[self.examples.y == 0], c='blue', alpha=0.5)
        # plt.scatter(pep_w[self.examples.y == 1], tcr_w[self.examples.y == 1], c='green')
        # plt.show()
        # plt.hist(weights, bins=40)
        # plt.show()
        # raise Exception()

    def __len__(self) -> int:
        """
        :return: length of the dataset
        """
        return len(self.examples)

    @staticmethod
    def encode_peptide_cdr3b(peptide, CDR3b):
        """
        helper function to encode a single peptide:TCR pair
        :param peptide: peptide amino acid sequence
        :param CDR3b: TCR amino acid sequence
        :return: dictionary with `input_ids`, `token_type_ids`, and `attention_mask`
        """
        return tokenizer.encode_plus(" ".join(list(peptide)), " ".join(list(CDR3b)))

    def __getitem__(self, i: int) -> Dict:
        """
        Produces a numerical representation of a single observation in the dataset.
        Peptide and TCR are tokenized and concatenated,
        with CLS token at the start and PAD tokens in between peptide and TCR and at the end.
        :param i: index of the observation
        :return: dictionary with `input_ids`, `token_type_ids`, `attention_mask`, and optionally `labels` and `weights`
        """
        row = self.examples.iloc[i]
        input = PeptideTCRDataset.encode_peptide_cdr3b('', row.CDR3b)
        if "y" in self.examples.columns:
            input.data["labels"] = row.y
        if "weight" in self.examples.columns:
            input.data["weights"] = row.weight
        return input
