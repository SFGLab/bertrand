import pandas as pd
from torch.utils.data import Dataset

from bertrand.model.tokenization import tokenizer


class PeptideTCRMLMDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        self.examples = dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        row = self.examples.iloc[i]
        encode_plus = tokenizer.encode_plus(
            " ".join(list(row.Peptide)), " ".join(list(row.CDR3b))
        )
        return encode_plus
