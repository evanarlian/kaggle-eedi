import time
from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import late_interaction, make_valid_df, map_at_k, rank_dist


@dataclass
class Args:
    dataset_dir: Path
    model: str
    private: bool
    nth_hidden_state: int
    batch_size: int


@torch.no_grad()
def get_normalized_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    nth_hidden: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[Tensor], list[Tensor]]:
    """Get hidden states along with attn mask, batched style."""
    model.to(device)
    all_hidden_states = []  # each size (bs, n_tok, emb_sz)
    all_attn_masks = []  # # each size (bs, n_tok)
    for i in tqdm(range(0, len(texts), batch_size), desc="hidden_states"):
        subset = texts[i : i + batch_size]
        encoded = tokenizer(subset, padding=True, return_tensors="pt").to(device)
        model_output = model(**encoded, output_hidden_states=True)
        normed_hidden = F.normalize(
            model_output["hidden_states"][nth_hidden]
            * encoded["attention_mask"][..., None],
            dim=-1,
        )  # we multiply with attn mask to prevent pad token to mess with L2 norm
        all_hidden_states.append(normed_hidden)
        all_attn_masks.append(encoded["attention_mask"])
    assert len(all_hidden_states) == len(all_attn_masks)
    return all_hidden_states, all_attn_masks


@torch.no_grad()
def calculate_colbert_score(
    model,
    tokenizer,
    queries: list[str],
    documents: list[str],
    nth_hidden: int,
    batch_size: int,
    device: torch.device,
):
    """Calculate colbert score using late interaction. Use "tiling" operation to save memory."""
    Qs, Q_attns = get_normalized_hidden_states(
        model, tokenizer, queries, nth_hidden, batch_size, device
    )
    Ds, D_attns = get_normalized_hidden_states(
        model, tokenizer, documents, nth_hidden, batch_size, device
    )
    colbert_score = torch.empty(len(queries), len(documents))
    n_loops = ceil(len(queries) / batch_size) * ceil(len(documents) / batch_size)
    with tqdm(total=n_loops, desc="colbert_score") as pbar:
        for i, Q, Q_attn in zip(range(0, len(queries), batch_size), Qs, Q_attns):
            for j, D, D_attn in zip(range(0, len(documents), batch_size), Ds, D_attns):
                li = late_interaction(Q, D, Q_attn, D_attn)
                colbert_score[i : i + batch_size, j : j + batch_size] = li
                pbar.update(1)
    return colbert_score


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
    colbert_score = calculate_colbert_score(
        model=model,
        tokenizer=tokenizer,
        queries=sentences,
        documents=misconceptions,
        nth_hidden=args.nth_hidden_state,
        batch_size=args.batch_size,
        device=device,
    )
    print("late interaction shape:", colbert_score.size())

    if args.private:
        # make submission
        val, idx = colbert_score.topk(k=25)
        df_valid["MisconceptionId"] = [
            " ".join(str(m) for m in top25) for top25 in idx.tolist()
        ]
        df_ans = df_valid[["QuestionId_Answer", "MisconceptionId"]]
        df_ans.to_csv("submission.csv", index=False)
    else:
        # evaluate preds on train
        labels = torch.tensor(df_valid["MisconceptionIdLabel"].tolist())
        map_at_25 = map_at_k(labels, colbert_score, k=25)
        rank_distributions = rank_dist(labels, colbert_score, k=25)
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
        help="Nth hidden state to use. 0 means nn.Embedding, -1 means last hidden state.",
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        help="Batch size during colbert computation.",
    )
    args = parser.parse_args()
    args = Args(
        dataset_dir=args.dataset_dir,
        model=args.model,
        private=args.private,
        nth_hidden_state=args.nth_hidden_state,
        batch_size=args.batch_size,
    )
    print(args)
    t0 = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f} secs.")
