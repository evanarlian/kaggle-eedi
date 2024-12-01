import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from eedi.datasets import make_nice_df


@dataclass
class Args:
    dataset_dir: Path
    output_dir: Path


def make_jsonl(df: pd.DataFrame) -> list[dict]:
    flag_emb_dataset = []
    for i, row in df.iterrows():
        subject = row["SubjectName"]
        construct = row["ConstructName"]
        question = row["QuestionText"]
        mis = [f"MISCONCEPTION: {mis}" for mis in row["MisconceptionName"]]
        anchor = f"QUESTION: {subject}. {construct}. {question}"
        flag_row = {"query": anchor, "pos": mis}
        flag_emb_dataset.append(flag_row)
    return flag_emb_dataset


def save_flag_emb_dataset(jsonl: list[dict], savepath: Path) -> None:
    with open(savepath, "w") as f:
        for row in jsonl:
            f.write(json.dumps(row))
            f.write("\n")


def main(args: Args):
    # prepare dataset
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_full = pd.read_csv(args.dataset_dir / "train.csv")
    df_full = make_nice_df(df_full)
    df_full = pd.merge(
        left=df_full,
        right=df_mis,
        left_on="MisconceptionIdLabel",
        right_on="MisconceptionId",
    )

    # split dataset to train and val
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df_full, groups=df_full["SubjectName"]))
    df_train = df_full.iloc[train_idx]
    df_val = df_full.iloc[val_idx]

    # save the complete csv for evaluation later
    bge_folder = args.output_dir / "bge_dataset"
    bge_folder.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(bge_folder / "full.csv", index=False)
    df_train.to_csv(bge_folder / "train.csv", index=False)
    df_val.to_csv(bge_folder / "val.csv", index=False)

    df_full_g = df_full.groupby(
        ["SubjectName", "ConstructName", "QuestionText"], as_index=False
    )[["MisconceptionName"]].agg(set)
    df_train_g = df_train.groupby(
        ["SubjectName", "ConstructName", "QuestionText"], as_index=False
    )[["MisconceptionName"]].agg(set)
    df_val_g = df_val.groupby(
        ["SubjectName", "ConstructName", "QuestionText"], as_index=False
    )[["MisconceptionName"]].agg(set)

    # make 3 version of flag embedding dataset
    jsonl_full = make_jsonl(df_full_g)
    save_flag_emb_dataset(jsonl_full, bge_folder / "full.jsonl")
    jsonl_train = make_jsonl(df_train_g)
    save_flag_emb_dataset(jsonl_train, bge_folder / "train.jsonl")
    jsonl_val = make_jsonl(df_val_g)
    save_flag_emb_dataset(jsonl_val, bge_folder / "val.jsonl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Cleaned BGE dataset for training and validation.",
    )
    args = parser.parse_args()
    args = Args(dataset_dir=args.dataset_dir, output_dir=args.output_dir)
    print(args)
    t0 = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f} secs.")
