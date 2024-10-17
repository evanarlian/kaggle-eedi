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
