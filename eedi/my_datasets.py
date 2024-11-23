import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm.auto import tqdm
from usearch.index import Index


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


@torch.inference_mode()
def batched_inference(model, tokenizer, texts: list[str], bs: int, desc: str) -> Tensor:
    """Basically SentenceTransformer.encode, but consume less vram."""
    embeddings = []
    for i in tqdm(range(0, len(texts), bs), desc=desc):
        # max_length=256 comes from plotting the complete question text, and 256 covers 99%
        encoded = tokenizer(
            texts[i : i + bs],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = model(**encoded)
        emb = outputs.last_hidden_state[:, 0]  # cls token
        emb = F.normalize(emb, p=2, dim=-1)
        embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings)
    return embeddings


def hn_mine_st(
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
        q_texts (list[str]): Question texts.
        q_mis_ids (list[int]): Ground truth misconception ids for the questions.
        mis_texts (list[str]): Misconception texts.
        mis_ids (list[int]): Misconception ids.
        k (int): Top k hard misconception ids per question (at max).
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
    index = Index(ndim=m_embeds.shape[-1], metric="ip")
    index.add(np.arange(m_embeds.shape[0]), m_embeds)
    q_embeds = model.encode(
        q_texts,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=True,
        device="cuda",
    )
    batch_matches = index.search(q_embeds, count=k + 10)  # +10 compensate for same ids
    hards = []
    for i, matches in enumerate(batch_matches):  # type: ignore
        nth_miscons = [m.key for m in matches]
        hard_miscons = [
            nth.item() for nth in nth_miscons if mis_ids[nth] != q_mis_ids[i]
        ][:k]
        hards.append(hard_miscons)
    assert len(hards) == len(q_texts)
    return hards


def make_mnr_dataset(
    q_texts: list[str],
    q_mis_ids: list[int],
    mis_texts: list[str],
    mis_ids: list[int],
    hards: list[list[int]],
    n_negatives: int,
) -> Dataset:
    """Create SentenceTransformer dataset suitable for MultipleNegativesRankingLoss.
    The format is (anchor, positive, negative_1, …, negative_n).
    Example: https://huggingface.co/datasets/tomaarsen/gooaq-hard-negatives
    """
    assert len(q_texts) == len(q_mis_ids) == len(hards)
    assert len(mis_texts) == len(mis_ids)
    assert all(n_negatives <= len(hard) for hard in hards)
    # create reverse mapping
    mis_id_to_mis_idx = defaultdict(list)
    for i, mis_id in enumerate(mis_ids):
        mis_id_to_mis_idx[mis_id].append(i)
    # make hf dataset
    d = {}
    d["q"], d["mis"] = [], []
    for i in range(1, n_negatives + 1):
        d[f"neg_{i}"] = []
    for i, (q_text, q_mis_id) in enumerate(zip(q_texts, q_mis_ids)):
        rand_pos = random.choice(mis_id_to_mis_idx[q_mis_id])
        rand_negs = random.sample(hards[i], k=n_negatives)
        d["q"].append(q_text)
        d["mis"].append(mis_texts[rand_pos])
        for j, rand_neg in enumerate(rand_negs, 1):
            d[f"neg_{j}"].append(mis_texts[rand_neg])
    return Dataset.from_dict(d)


def make_cosent_dataset(
    q_texts: list[str],
    q_mis_ids: list[int],
    mis_texts: list[str],
    mis_ids: list[int],
    hards: list[list[int]],
    n_negatives: int,
) -> Dataset:
    """Create SentenceTransformer dataset suitable for CoSENTLoss.
    The format is (sentence_A, sentence_B).
    Example: https://sbert.net/docs/sentence_transformer/training_overview.html#loss-function
    """
    assert len(q_texts) == len(q_mis_ids) == len(hards)
    assert len(mis_texts) == len(mis_ids)
    assert all(n_negatives <= len(hard) for hard in hards)
    # create reverse mapping
    mis_id_to_mis_idx = defaultdict(list)
    for i, mis_id in enumerate(mis_ids):
        mis_id_to_mis_idx[mis_id].append(i)
    # make hf dataset
    d = {"q": [], "mis": [], "label": []}
    for i, (q_text, q_mis_id) in enumerate(zip(q_texts, q_mis_ids)):
        # insert positive
        rand_pos = random.choice(mis_id_to_mis_idx[q_mis_id])
        d["q"].append(q_text)
        d["mis"].append(mis_texts[rand_pos])
        d["label"].append(1.0)
        # insert negatives
        rand_negs = random.sample(hards[i], k=n_negatives)
        for j, rand_neg in enumerate(rand_negs, 1):
            d["q"].append(q_text)
            d["mis"].append(mis_texts[rand_neg])
            d["label"].append(-1.0)
    return Dataset.from_dict(d)


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
