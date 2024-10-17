from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import make_valid_train


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
    df_sub = pd.read_csv(args.dataset_dir / "sample_submission.csv")
    df = df_test.copy() if args.private else df_train

    # embed
    sentences = (
        df["SubjectName"] + ". " + df["ConstructName"] + ". " + df["QuestionText"]
    ).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    model = SentenceTransformer(args.emb)

    # calculate embeddings
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    mis_embs = model.encode(misconceptions, convert_to_tensor=True)
    print("sent_emb shape:", sent_embs.size())
    print("mis_emb shape:", sent_embs.size())

    # calculate the embedding similarities
    similarities = model.similarity(sent_embs, mis_embs)
    print(similarities.size())
    val, idx = similarities.topk(k=25, dim=1)

    # make submission
    d = {
        "QuestionId_Answer": [],
        "MisconceptionId": [],
    }
    for (i, row), top25 in zip(df_train.iterrows(), idx.tolist()):
        correct = row["CorrectAnswer"]
        for letter in "ABCD":
            if letter == correct:
                continue
            d["QuestionId_Answer"].append(f"{row['QuestionId']}_{letter}")
            # for now, assign the same top25 regardless of user wrong answer
            d["MisconceptionId"].append(" ".join(str(m) for m in top25))
    df_ans = pd.DataFrame.from_dict(d)
    df_ans.to_csv("submission.csv", index=False)


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
    main(args)
