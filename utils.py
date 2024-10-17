import numpy as np
import pandas as pd


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


def make_valid_train(df_train: pd.DataFrame) -> pd.DataFrame:
    """Creates a valid train dataframe. Valid means the misconception id must be not null
    and the dataset is melted to ease row-by-row inference. We are need to melt 2 times
    because we want to melt both answer and miconceptions in unison.
    """
    # 1. melt answers
    df_melted_ans = pd.melt(
        df_train,
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
        var_name="_melted_ans_header",  # rename the column that holds melted-column's headers
        value_name="_melted_ans",  # rename the column that holds melted-column's content
    )
    df_melted_ans = df_melted_ans.sort_values(["QuestionId", "_melted_ans_header"])
    # 2. melt misconceptions
    df_melted_mis = pd.melt(
        df_train,
        id_vars=["QuestionId"],
        value_vars=[
            "MisconceptionAId",
            "MisconceptionBId",
            "MisconceptionCId",
            "MisconceptionDId",
        ],
        var_name="_melted_mis_header",
        value_name="_melted_mis_id",
    )
    df_melted_mis = df_melted_mis.sort_values(["QuestionId", "_melted_mis_header"])
    # 3. combine and cleam
    df_melted_mis = df_melted_mis.drop(columns="QuestionId")
    df_valid = pd.concat([df_melted_ans, df_melted_mis], axis=1)
    df_valid = df_valid[df_valid["_melted_mis_id"].notna()]
    df_valid["_melted_mis_id"] = df_valid["_melted_mis_id"].astype(int)
    df_valid = df_valid.reset_index(drop=True)
    return df_valid

