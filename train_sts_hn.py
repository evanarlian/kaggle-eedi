from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch import nn
from torch import optim
import torch
from torch import Tensor

@dataclass
class Args:
    dataset_path: Path
    model: str


def hn_mine(model: nn.Module) -> :


def main(args: Args):
    # load model
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # load data
    df = pd.read_csv(args.dataset_path)
    # need to separate the data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=Path, required=True, help="Paraphrased dataset"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Embedding model to train"
    )
    args = parser.parse_args()
    args = Args(dataset_path=args.dataset_path, model=args.model)
    main(args)
