import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from utils import late_interaction, make_valid_df, map_at_k, rank_dist


@dataclass
class Args:
    dataset_dir: Path
    model: str
    private: bool
    nth_hidden_state: int


@torch.no_grad()
def get_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    nth_hidden: int,
    batch_size: int,
    device: torch.device,
) -> list[Tensor]:
    model.to(device)
    all_hidden_states = []  # each size (n_tok, emb_sz)
    for i in tqdm(range(0, len(texts), batch_size)):
        subset = texts[i : i + batch_size]
        encoded = tokenizer(subset, padding=True, return_tensors="pt").to(device)
        model_output = model(**encoded, output_hidden_states=True)
        hidden_state = model_output["hidden_states"][nth_hidden].cpu()
        all_hidden_states += list(hidden_state)
    return all_hidden_states


@torch.no_grad()
def calculate_colbert_score():
    # hoolly shit this funciton will be quite messy because of memory usage of pure batch
    # but instead we need to calculate loop D and Q in batched fashion
    # TODO implement tiling colbert score
    pass


def main(args: Args):
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    # load dataset
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_train = pd.read_csv(args.dataset_dir / "train.csv")
    df_test = pd.read_csv(args.dataset_dir / "test.csv")
    if args.private:
        df = df_test.copy()
    else:
        df = df_train.copy()
    df_valid = make_valid_df(df)

    # embed + late interaction
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
    Q = get_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=sentences,
        nth_hidden=args.nth_hidden_state,
        batch_size=128,
        device=device,
    )
    D = get_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=misconceptions,
        nth_hidden=args.nth_hidden_state,
        batch_size=128,
        device=device,
    )
    li = late_interaction(
        Q, D, encoded_sents.attention_mask, encoded_mis.attention_mask
    )
    print("late interaction shape:", li.size())

    if args.private:
        # make submission
        val, idx = li.topk(k=25)
        df_valid["MisconceptionId"] = [
            " ".join(str(m) for m in top25) for top25 in idx.tolist()
        ]
        df_ans = df_valid[["QuestionId_Answer", "MisconceptionId"]]
        df_ans.to_csv("submission.csv", index=False)
    else:
        # evaluate preds on train
        labels = torch.tensor(df_valid["MisconceptionIdLabel"].tolist())
        map_at_25 = map_at_k(labels, li, k=25)
        rank_distributions = rank_dist(labels, li, k=25)
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
        "--model", required=True, type=str, help="Embedding model path or name from HF."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether code running with private test dataset or not.",
    )
    parser.add_argument(
        "--nth-hidden-state",
        required=True,
        type=int,
        help="Nth hidden state to use. 0 means nn.Embedding, -1 means last hidden state",
    )
    args = parser.parse_args()
    args = Args(
        dataset_dir=args.dataset_dir,
        model=args.model,
        private=args.private,
        nth_hidden_state=args.nth_hidden_state,
    )
    print(args)
    t0 = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f} secs.")
