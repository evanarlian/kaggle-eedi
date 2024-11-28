import random
from collections import defaultdict
from typing import Iterable, Literal

import pandas as pd
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import PreTrainedModel, PreTrainedTokenizer

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


def hn_mine_sbert(
    model: SentenceTransformer,
    q_texts: list[str],
    q_mis_ids: list[int],
    mis_texts: list[str],
    mis_ids: list[int],
    k: int,
    bs: int,
) -> list[list[int]]:
    """Hard negative mining, but different from: https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.mine_hard_negatives.
    Sentence Transformers' version assumes different rows are always negatives, but that is not the case if we use paraphrased data.

    Args:
        model (SentenceTransformer): Sentence transformer model.
        q_texts (list[str]): Question texts.
        q_mis_ids (list[int]): Ground truth misconception ids for the questions.
        mis_texts (list[str]): Misconception texts.
        mis_ids (list[int]): Misconception ids.
        k (int): Top k hard misconception ids per question.
        bs (int): Batch size.

    Returns:
        list[list[int]]:
            Hard misconceptions for each question.
            This is NOT misconception ids, but the actual list index.
    """
    assert len(q_texts) == len(q_mis_ids)
    assert len(mis_texts) == len(mis_ids)
    m_embeds = model.encode(
        mis_texts,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=True,
        device="cuda",
    )
    # +10 compensate for same ids
    nn = NearestNeighbors(n_neighbors=k + 10, algorithm="brute", metric="cosine")
    nn.fit(m_embeds)
    q_embeds = model.encode(
        q_texts,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=True,
        device="cuda",
    )
    ranks = nn.kneighbors(q_embeds, return_distance=False)
    hards = []
    for i, top_n_miscons in enumerate(ranks):
        hards.append(top_n_miscons[top_n_miscons != q_mis_ids[i]][:k].tolist())
    assert len(hards) == len(q_texts)
    return hards


# TODO unused again lol
def hn_mine_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    q_texts: list[str],
    q_mis_ids: list[int],
    mis_texts: list[str],
    mis_ids: list[int],
    k: int,
    bs: int,
    token_pool: Literal["first", "last"],
    device,
) -> list[list[int]]:
    """Hard negative mining, but different from: https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.mine_hard_negatives.
    Sentence Transformers' version assumes different rows are always negatives, but that is not the case if we use paraphrased data.

    Args:
        model (PreTrainedModel): Huggingface model.
        tokenizer (PreTrainedTokenizer): Huggingface tokenizer.
        q_texts (list[str]): Question texts.
        q_mis_ids (list[int]): Ground truth misconception ids for the questions.
        mis_texts (list[str]): Misconception texts.
        mis_ids (list[int]): Misconception ids.
        k (int): Top k hard misconception ids per question.
        bs (int): Batch size.

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
        desc="Misconceptions",
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
        desc="Questions",
    ).numpy()
    ranks = nn.kneighbors(q_embeds, return_distance=False)
    hards = []
    for i, top_n_miscons in enumerate(ranks):
        hards.append(top_n_miscons[top_n_miscons != q_mis_ids[i]][:k].tolist())
    assert len(hards) == len(q_texts)
    return hards


class TrainDatasetProxy(HFDataset):
    def __init__(
        self,
        q_texts: list[str],
        q_mis_ids: list[int],
        mis_texts: list[str],
        mis_ids: list[int],
        hards: list[list[int]],
        n_negatives: int,
    ) -> None:
        """Create SentenceTransformer dataset suitable for MultipleNegativesRankingLoss.
        The format is (anchor, positive, negative_1, …, negative_n).
        Example: https://huggingface.co/datasets/tomaarsen/gooaq-hard-negatives

        This custom class will override the default huggingface dataset behavior to allow
        random sampling over the positives and hard negatives.
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
        # make a fake huggingface dataset, we wont use the content anyway
        self.colnames = ["anchor", "pos"] + [
            f"neg_{i}" for i in range(1, n_negatives + 1)
        ]
        d = {colname: [] for colname in self.colnames}
        for i, (q_text, q_mis_id) in enumerate(zip(q_texts, q_mis_ids)):
            rand_pos = random.choice(self.mis_id_to_mis_idx[q_mis_id])
            rand_negs = random.sample(hards[i], k=n_negatives)
            d["anchor"].append(q_text)
            d["pos"].append(mis_texts[rand_pos])
            for j, rand_neg in enumerate(rand_negs, 1):
                d[f"neg_{j}"].append(mis_texts[rand_neg])
        self.original = HFDataset.from_dict(d)

    def replace_hards(self, new_hards: list[list[int]]) -> None:
        """Setter to allow hard negative replacement for iterative hard negative mining."""
        assert len(self.q_texts) == len(self.q_mis_ids) == len(new_hards)
        assert all(self.n_negatives <= len(hard) for hard in new_hards)
        self.hards = new_hards

    def __len__(self) -> int:
        return len(self.original)

    def __getattr__(self, name):
        # delegate getting missing attribute name to original hf dataset
        return getattr(self.original, name)

    def __delattr__(self, name):
        # delegate deleting missing attribute name to original hf dataset
        return delattr(self.original, name)

    def get_one_item(self, i: int) -> dict:
        rand_pos = random.choice(self.mis_id_to_mis_idx[self.q_mis_ids[i]])
        rand_negs = random.sample(self.hards[i], k=self.n_negatives)
        return {
            "anchor": self.q_texts[i],
            "pos": self.mis_texts[rand_pos],
            **{f"neg_{j}": self.mis_texts[x] for j, x in enumerate(rand_negs, 1)},
        }

    def __getitem__(self, key: int | slice | Iterable[int]) -> dict:  # type: ignore
        # custom behavior for item access
        if isinstance(key, int):
            return self.get_one_item(key)
        if isinstance(key, slice):
            indices = list(range(*key.indices(len(self))))
        else:
            indices = list(key)
        d = {colname: [] for colname in self.colnames}
        batch = [self.get_one_item(i) for i in indices]
        for row in batch:
            for k, v in row.items():
                d[k].append(v)
        return d


def make_ir_evaluator_dataset(
    df_val: pd.DataFrame, orig_mis: list[str]
) -> tuple[dict, dict, dict]:
    """Create dicts required by SentenceTransformer's InformationRetrievalEvaluator."""
    assert len(orig_mis) == 2587
    mapping = (
        df_val[["QuestionId_Answer", "MisconceptionId"]]
        .set_index("QuestionId_Answer")["MisconceptionId"]
        .apply(lambda x: [x])
        .to_dict()
    )
    q = (
        df_val[["QuestionId_Answer", "QuestionComplete"]]
        .set_index("QuestionId_Answer")["QuestionComplete"]
        .to_dict()
    )
    mis = {i: mis_text for i, mis_text in enumerate(orig_mis)}
    return q, mis, mapping
