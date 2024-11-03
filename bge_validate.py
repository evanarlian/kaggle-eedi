import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from utils import map_at_k, rank_dist


@dataclass
class Args:
    dataset_dir: Path
    emb: str


def main(args: Args):
    # load dataset
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_val = pd.read_csv(args.dataset_dir / "bge_dataset" / "val.csv")

    # NOTE: since the model does not use the answer, these triplet below will form duplicates. No biggie
    misconceptions = ("MISCONCEPTION: " + df_mis["MisconceptionName"]).tolist()
    questions = (
        "QUESTION: "
        + df_val["SubjectName"]
        + ". "
        + df_val["ConstructName"]
        + ". "
        + df_val["QuestionText"]
    ).tolist()
    model = SentenceTransformer(args.emb)

    # calculate embeddings
    sent_embs = model.encode(questions, convert_to_tensor=True)
    mis_embs = model.encode(misconceptions, convert_to_tensor=True)
    print("sent_emb shape:", sent_embs.size())
    print("mis_emb shape:", mis_embs.size())

    # calculate the embedding similarities
    similarities = model.similarity(sent_embs, mis_embs).cpu()
    print("similarities shape:", similarities.size())

    # evaluate preds on train
    labels = torch.tensor(df_val["MisconceptionIdLabel"].tolist())
    map_at_25 = map_at_k(labels, similarities, k=25)
    rank_distributions = rank_dist(labels, similarities, k=25)
    print(f"mAP@25 is {map_at_25}")
    print("=============")
    for rank, count in rank_distributions.items():
        print(f"rank {rank}: {count} ({count/labels.size(0):.2%})")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder."
    )
    parser.add_argument(
        "--emb", required=True, type=str, help="Embedding model path or name from HF."
    )
    args = parser.parse_args()
    args = Args(dataset_dir=args.dataset_dir, emb=args.emb)
    print(args)
    t0 = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f} secs.")
