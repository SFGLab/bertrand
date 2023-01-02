import os

from transformers import BertTokenizer

AA_list = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
AA_dict = {v: (k + 1) for k, v in enumerate(AA_list)}
AA_dict["-"] = 0


tokenizer = BertTokenizer(
    os.path.join(os.path.split(os.path.realpath(__file__))[0], "vocab.txt"),
    do_lower_case=False,
)
