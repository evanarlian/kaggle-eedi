import numpy as np
import pandas as pd
import torch
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
    df_melted_ans = df_melted_ans.sort_values(["QuestionId", "WrongChoice"])
    try:
        # 2. melt misconceptions
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
        k (int, optional): k for mAP@k. Defaults to 25.

    Returns:
        float: mAP@k score
    """
    min_k = min(similarities.size(-1), k)
    val, idx = similarities.topk(min_k)
    mask = labels[:, None] == idx
    denominator = torch.arange(min_k) + 1
    mapk = 1 / denominator[None, :] * mask
    return mapk.sum(-1).mean().item()
