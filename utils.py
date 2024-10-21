import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor


def merge_with_misconceptions(
    df_train: pd.DataFrame, df_mis: pd.DataFrame
) -> pd.DataFrame:
    mis_dict = df_mis["MisconceptionName"].to_dict()
    df_merged = df_train.copy()
    for letter in "ABCD":
        df_merged[f"Misconception{letter}Name"] = df_merged[
            f"Misconception{letter}Id"
        ].apply(lambda x: None if np.isnan(x) else mis_dict[int(x)])
    return df_merged


def make_valid_df(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a valid train or test dataframe. Valid means the misconception id must be not null
    and the dataset is melted to ease row-by-row inference. For train, we are need to melt 2 times
    because we want to melt both answer and miconceptions in unison.
    """
    # 1. melt answers
    df_melted_ans = pd.melt(
        df,
        id_vars=[  # what column to keep
            "QuestionId",
            "ConstructId",
            "ConstructName",
            "SubjectId",
            "SubjectName",
            "CorrectAnswer",
            "QuestionText",
        ],
        value_vars=[  # what columns to transform to rows (melted)
            "AnswerAText",
            "AnswerBText",
            "AnswerCText",
            "AnswerDText",
        ],
        var_name="WrongChoice",  # rename the column that holds melted-column's headers
        value_name="AnswerText",  # rename the column that holds melted-column's content
    )
    df_melted_ans["WrongChoice"] = df_melted_ans["WrongChoice"].str[6]
    df_melted_ans = df_melted_ans[
        df_melted_ans["WrongChoice"] != df_melted_ans["CorrectAnswer"]
    ]
    df_melted_ans = df_melted_ans.sort_values(["QuestionId", "WrongChoice"])
    try:
        # 2. melt misconceptions (only available at train dataset)
        df_melted_mis = pd.melt(
            df,
            id_vars=["QuestionId"],
            value_vars=[
                "MisconceptionAId",
                "MisconceptionBId",
                "MisconceptionCId",
                "MisconceptionDId",
            ],
            var_name="_melted_mis_header",
            value_name="MisconceptionIdLabel",
        )
        df_melted_mis = df_melted_mis.sort_values(["QuestionId", "_melted_mis_header"])
        df_melted_mis = df_melted_mis.drop(columns=["_melted_mis_header"])
        # 3. combine and cleam
        df_melted_mis = df_melted_mis.drop(columns="QuestionId")
        df_valid = pd.concat([df_melted_ans, df_melted_mis], axis=1)
        df_valid = df_valid[df_valid["MisconceptionIdLabel"].notna()]
        df_valid["MisconceptionIdLabel"] = df_valid["MisconceptionIdLabel"].astype(int)
    except KeyError:
        # test set does not have misconceptions
        df_valid = df_melted_ans
    df_valid = df_valid.reset_index(drop=True)
    df_valid["QuestionId_Answer"] = (
        df_valid["QuestionId"].astype(str) + "_" + df_valid["WrongChoice"]
    )
    return df_valid


def map_at_k(labels: Tensor, similarities: Tensor, k: int) -> float:
    """Calculates mAP@k metric.

    Args:
        labels (Tensor): ground truth tensor, size (n_rows,).
        similarities (Tensor): similarity score tensor, size (n_rows, n_misconceptions).
        k (int): k for mAP@k.

    Returns:
        float: mAP@k score
    """
    min_k = min(similarities.size(-1), k)
    val, idx = similarities.topk(min_k)
    mask = labels[:, None] == idx
    denominator = torch.arange(min_k) + 1
    mapk = 1 / denominator[None, :] * mask
    return mapk.sum(-1).mean().item()


def rank_dist(labels: Tensor, similarities: Tensor, k: int) -> dict[int, int]:
    """Calculates rank distributions from similarities.

    Args:
        labels (Tensor): ground truth tensor, size (n_rows,).
        similarities (Tensor): similarity score tensor, size (n_rows, n_misconceptions).
        k (int): k for mAP@k.

    Returns:
        Counter:
            Rank distributions, ranging from 1 to k inclusive.
            Will also count rank -1, this indicates rank above k.
    """
    d = dict.fromkeys([-1] + list(range(1, k + 1)), 0)
    min_k = min(similarities.size(-1), k)
    val, idx = similarities.topk(min_k)
    mask = labels[:, None] == idx
    aranger = torch.arange(1, min_k + 1).expand(labels.size(0), min_k)
    ranks = aranger[mask].tolist()
    d[-1] = labels.size(0) - len(ranks)
    for rank in ranks:
        d[rank] += 1
    assert sum(d.values()) == labels.size(0)
    return d


def late_interaction(
    queries: Tensor, docs: Tensor, query_mask: Tensor, doc_mask: Tensor
) -> Tensor:
    """Apply batched colbert late interaction with cosine similarity.

    Args:
        queries (Tensor): Queries, size (nq, n_q_tok, emb_sz)
        docs (Tensor): Documents, size (nd, n_d_tok, emb_sz)
        query_mask (Tensor): Query attn mask, size (nq, n_q_tok)
        doc_mask (Tensor): Document attn mask, size (nd, n_d_tok)

    Returns:
        Tensor: Late interaction, maxsim applied. Size (nq, nd)
    """
    queries = F.normalize(queries, dim=-1)
    docs = F.normalize(docs, dim=-1)
    #   (nq, 1, n_q_tok, emb_sz)
    #   (1, nd, emb_sz, n_d_tok)
    # = (nq, nd, n_q_tok, n_d_tok)
    li = queries[:, None] @ docs[None, :].transpose(-1, -2)
    #   (nq, 1, n_q_tok, 1)
    #   (1, nd, 1, n_d_tok)
    # = (nq, nd, n_q_tok, n_d_tok)
    mask = query_mask[:, None, :, None] * doc_mask[None, :, None, :]
    # temporarily reduce the padding value to under -1, to make padding impossible to win max operation
    # -2.0 will work since after L2 norm, max possible value is 1.0
    li += (mask - 1.0) * 2
    li = li.max(-1).values
    # bring back padding to 0.0 just before sum
    li = (li * mask.max(-1).values).sum(-1)
    return li


def manual_late_interaction(queries: list[Tensor], docs: list[Tensor]) -> Tensor:
    """Loopy version of colbert late interaction with cosine similarity.
    Don't use! This version is only for debugging and correctness check.

    Args:
        queries (list[Tensor]): Queries, each size (n_q_tok, emb_sz)
        docs (Tensor): Documents, each size (n_d_tok, emb_sz)

    Returns:
        Tensor: Late interaction, maxsim applied. Size (nq, nd)
    """
    li = torch.zeros(len(queries), len(docs))
    for i, q in enumerate(queries):
        for j, d in enumerate(docs):
            q = F.normalize(q, dim=-1)
            d = F.normalize(d, dim=-1)
            li[i, j] = (q @ d.T).max(-1).values.sum(-1)
    return li
