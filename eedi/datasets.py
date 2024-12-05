import random
from collections import defaultdict
from typing import Literal

import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from eedi.helpers import batched_inference


def make_nice_df(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a valid train or test dataframe. Valid means the misconception id must be not null
    and the dataset is melted to ease row-by-row inference. For train, we are need to melt 2 times
    because we want to melt both answer and miconceptions in unison.
    """
    # 1. duplicate correct answer text to its own column
    df = df.copy()
    df = df.rename(columns={"CorrectAnswer": "CorrectChoice"})
    df["CorrectText"] = df.apply(lambda x: x[f"Answer{x['CorrectChoice']}Text"], axis=1)
    # 2. melt answers
    df_melted_ans = pd.melt(
        df,
        id_vars=[  # what column to keep
            "QuestionId",
            "ConstructId",
            "ConstructName",
            "SubjectId",
            "SubjectName",
            "CorrectChoice",
            "CorrectText",
            "QuestionText",
        ],
        value_vars=[  # what columns to transform to rows (melted)
            "AnswerAText",
            "AnswerBText",
            "AnswerCText",
            "AnswerDText",
        ],
        var_name="WrongChoice",  # rename the column that holds melted-column's headers
        value_name="WrongText",  # rename the column that holds melted-column's content
    )
    df_melted_ans["WrongChoice"] = df_melted_ans["WrongChoice"].str[6]
    df_melted_ans = df_melted_ans.sort_values(["QuestionId", "WrongChoice"])
    df_melted_ans = df_melted_ans.reset_index(drop=True)
    try:
        # 3. melt misconceptions (only available at train dataset)
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
            value_name="MisconceptionId",
        )
        df_melted_mis = df_melted_mis.sort_values(["QuestionId", "_melted_mis_header"])
        df_melted_mis = df_melted_mis.drop(columns=["QuestionId", "_melted_mis_header"])
        df_melted_mis = df_melted_mis.reset_index(drop=True)
        # 4. combine
        assert len(df_melted_ans) == len(df_melted_mis)
        df_nice = pd.concat([df_melted_ans, df_melted_mis], axis=1)
    except KeyError:
        # test set does not have misconceptions
        df_nice = df_melted_ans
    # 5. clean
    df_nice = df_nice[(df_nice["WrongChoice"] != df_nice["CorrectChoice"])]
    try:
        df_nice = df_nice[df_nice["MisconceptionId"].notna()]
        df_nice["MisconceptionId"] = df_nice["MisconceptionId"].astype(int)
    except KeyError:
        pass
    df_nice = df_nice.reset_index(drop=True)
    df_nice["QuestionId_Answer"] = (
        df_nice["QuestionId"].astype(str) + "_" + df_nice["WrongChoice"]
    )
    return df_nice


def make_complete_query(row: pd.Series) -> str:
    template = "SUBJECT: {}\n\nCONSTRUCT: {}\n\nQUESTION: {}\n\nCORRECT ANSWER: {}\n\nWRONG ANSWER: {}"
    return template.format(
        row["SubjectName"],
        row["ConstructName"],
        row["QuestionText"],
        row["CorrectText"],
        row["WrongText"],
    )


def hn_mine_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q_texts: list[str],
    q_mis_ids: list[int],
    mis_texts: list[str],
    mis_ids: list[int],
    k: int,
    bs: int,
    token_pool: Literal["first", "last"],
    device: torch.device,
    tqdm: bool,
) -> list[list[int]]:
    """Hard negative mining, but different from: https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.mine_hard_negatives.
    Sentence Transformers' version assumes different rows are always negatives, but that is not the case if we use paraphrased data.

    Args:
        model (PreTrainedModel): Huggingface model.
        tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer.
        q_texts (list[str]): Question texts.
        q_mis_ids (list[int]): Ground truth misconception ids for the questions.
        mis_texts (list[str]): Misconception texts.
        mis_ids (list[int]): Misconception ids.
        k (int): Top k hard misconception ids per question.
        bs (int): Batch size.
        token_pool (str, {"first", "last"}): Token considered for sentence embedding.
        device (torch.device): Device.
        tqdm (bool): Show tqdm bar or not.

    Returns:
        list[list[int]]:
            Hard misconceptions for each question.
            This is NOT misconception ids, but the actual list index.
    """
    assert len(q_texts) == len(q_mis_ids)
    assert len(mis_texts) == len(mis_ids)
    m_embeds = batched_inference(
        model,
        tokenizer,
        mis_texts,
        bs=bs,
        token_pool=token_pool,
        device=device,
        desc="hn mine mis" if tqdm else None,
    ).numpy()
    # +10 compensate for same ids
    nn = NearestNeighbors(n_neighbors=k + 10, algorithm="brute", metric="cosine")
    nn.fit(m_embeds)
    q_embeds = batched_inference(
        model,
        tokenizer,
        q_texts,
        bs=bs,
        token_pool=token_pool,
        device=device,
        desc="hn mine q" if tqdm else None,
    ).numpy()
    ranks = nn.kneighbors(q_embeds, return_distance=False)
    hards = []
    for i, top_n_miscons in enumerate(ranks):
        hards.append(top_n_miscons[top_n_miscons != q_mis_ids[i]][:k].tolist())
    assert len(hards) == len(q_texts)
    return hards


class TrainDataset(Dataset):
    def __init__(
        self,
        q_texts: list[str],
        q_mis_ids: list[int],
        mis_texts: list[str],
        mis_ids: list[int],
        hards: list[list[int]],
        n_negatives: int,
    ) -> None:
        """Create dataset suitable for MultipleNegativesRankingLoss. Each index will
        output anchor, random positive, and many random negatives.
        """
        assert len(q_texts) == len(q_mis_ids) == len(hards)
        assert len(mis_texts) == len(mis_ids)
        assert all(n_negatives <= len(hard) for hard in hards)
        self.q_texts = q_texts
        self.q_mis_ids = q_mis_ids
        self.mis_texts = mis_texts
        self.mis_ids = mis_ids
        self.hards = hards
        self.n_negatives = n_negatives
        # create reverse mapping
        self.mis_id_to_mis_idx = defaultdict(list)
        for i, mis_id in enumerate(mis_ids):
            self.mis_id_to_mis_idx[mis_id].append(i)

    def replace_hards(self, new_hards: list[list[int]]) -> None:
        """Setter to allow hard negative replacement for iterative hard negative mining."""
        assert len(self.q_texts) == len(self.q_mis_ids) == len(new_hards)
        assert all(self.n_negatives <= len(hard) for hard in new_hards)
        self.hards = new_hards

    def __len__(self) -> int:
        return len(self.q_texts)

    def __getitem__(self, i: int) -> dict:
        rand_pos = random.choice(self.mis_id_to_mis_idx[self.q_mis_ids[i]])
        rand_negs = random.sample(self.hards[i], k=self.n_negatives)
        return {
            "anchor": self.q_texts[i],
            "pos": self.mis_texts[rand_pos],
            "negs": [self.mis_texts[x] for x in rand_negs],
        }


class EvalDataset(Dataset):
    def __init__(
        self, q_texts: list[str], q_mis_ids: list[int], mis_texts: list[str]
    ) -> None:
        """Create dataset suitable for evaluating retrieval. Each index will
        output an anchor and a positive text.
        """
        assert len(q_texts) == len(q_mis_ids)
        assert len(mis_texts) == 2587
        self.q_texts = q_texts
        self.q_mis_ids = q_mis_ids
        self.mis_texts = mis_texts

    def __len__(self) -> int:
        return len(self.q_texts)

    def __getitem__(self, i: int) -> dict:
        return {
            "anchor": self.q_texts[i],
            "pos": self.mis_texts[self.q_mis_ids[i]],
        }


class MyCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        anchors = [row["anchor"] for row in batch]
        positives = [row["pos"] for row in batch]
        negatives = [item for row in batch for item in row["negs"]]
        encoded_anchor = self.tokenizer(
            anchors,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded_pos_neg = self.tokenizer(
            positives + negatives,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {"anchor": encoded_anchor, "pos_neg": encoded_pos_neg}
