import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from eedi.metrics import map_at_k, rank_dist
from eedi.my_datasets import make_nice_df


@dataclass
class Args:
    dataset_dir: Path
    emb: str
    private: bool


def main(args: Args):
    # load dataset
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_train = pd.read_csv(args.dataset_dir / "train.csv")
    df_test = pd.read_csv(args.dataset_dir / "test.csv")
    if args.private:
        df = df_test.copy()
    else:
        df = df_train.copy()
    df_valid = make_nice_df(df)

    # embed
    sentences = (
        df_valid["SubjectName"]
        + ". "
        + df_valid["ConstructName"]
        + ". "
        + df_valid["QuestionText"]
        + ". "
        + df_valid["AnswerText"]
    ).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    model = SentenceTransformer(args.emb)

    # calculate embeddings
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    mis_embs = model.encode(misconceptions, convert_to_tensor=True)
    print("sent_emb shape:", sent_embs.size())
    print("mis_emb shape:", mis_embs.size())

    # calculate the embedding similarities
    similarities = model.similarity(sent_embs, mis_embs).cpu()
    print("similarities shape:", similarities.size())

    if args.private:
        # make submission
        val, idx = similarities.topk(k=25)
        df_valid["MisconceptionId"] = [
            " ".join(str(m) for m in top25) for top25 in idx.tolist()
        ]
        df_ans = df_valid[["QuestionId_Answer", "MisconceptionId"]]
        df_ans.to_csv("submission.csv", index=False)
    else:
        # evaluate preds on train
        labels = torch.tensor(df_valid["MisconceptionIdLabel"].tolist())
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
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether code running with private test dataset or not.",
    )
    args = parser.parse_args()
    args = Args(dataset_dir=args.dataset_dir, emb=args.emb, private=args.private)
    print(args)
    t0 = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f} secs.")
