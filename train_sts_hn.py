import random
import string
import time
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from usearch.index import Index


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
    batch_matches = index.search(q_embeds, count=k)
    hards = []
    for i, matches in enumerate(batch_matches):  # type: ignore
        nth_miscons: list[int] = [m.key for m in matches]
        hard_miscons = [nth for nth in nth_miscons if mis_ids[nth] != q_mis_ids[i]]
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
) -> HFDataset:
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
    return HFDataset.from_dict(d)


def make_cosent_dataset(
    q_texts: list[str],
    q_mis_ids: list[int],
    mis_texts: list[str],
    mis_ids: list[int],
    hards: list[list[int]],
    n_negatives: int,
) -> HFDataset:
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
    return HFDataset.from_dict(d)


def make_ir_evaluator_dataset(
    df: pd.DataFrame, all_mis_texts: list[str]
) -> tuple[dict, dict, dict]:
    temp = (
        df[
            [
                "QuestionId_Answer",
                "QuestionComplete",
                "MisconceptionId",
                "MisconceptionText",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    mapping = (
        temp[["QuestionId_Answer", "MisconceptionId"]]
        .set_index("QuestionId_Answer")["MisconceptionId"]
        .apply(lambda x: [x])
        .to_dict()
    )
    q = (
        temp[["QuestionId_Answer", "QuestionComplete"]]
        .set_index("QuestionId_Answer")["QuestionComplete"]
        .to_dict()
    )
    mis = {i: mis_text for i, mis_text in enumerate(all_mis_texts)}
    return q, mis, mapping


def main():
    # 1. load data
    all_mis_texts = pd.read_csv("data/misconception_mapping.csv")[
        "MisconceptionName"
    ].tolist()
    df = pd.read_csv("data/eedi-paraphrased/train.csv")
    df["QuestionComplete"] = (
        "Subject: "
        + df["SubjectName"]
        + ". Construct: "
        + df["ConstructName"]
        + ". Question: "
        + df["QuestionText"]
        + ". Correct answer: "
        + df["CorrectText"]
        + ". Wrong answer: "
        + df["WrongText"]
        + "."
    )
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["QuestionId"]))
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_val = df_val[
        ~df_val["QuestionAiCreated"] & ~df_val["MisconceptionAiCreated"]
    ].reset_index(drop=True)
    df_q = (
        df_train[["MisconceptionId", "QuestionComplete"]]
        .sort_values("MisconceptionId")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # NOTE: this is not a bug, we look for misconceptions from the full dataset, not train dataset
    df_m = (
        temp[["MisconceptionId", "MisconceptionText"]]
        .sort_values("MisconceptionId")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # load model
    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # mine negatives
    hards_st = hn_mine_st(
        model,
        q_texts=df_q["QuestionComplete"].tolist(),
        q_mis_ids=df_q["MisconceptionId"].tolist(),
        mis_texts=df_m["MisconceptionText"].tolist(),
        mis_ids=df_m["MisconceptionId"].tolist(),
        k=100,
        bs=4,
    )

    # build dataset according to sentence transformer's format
    ds = make_mnr_dataset(
        q_texts=df_q["QuestionComplete"].tolist(),
        q_mis_ids=df_q["MisconceptionId"].tolist(),
        mis_texts=df_m["MisconceptionText"].tolist(),
        mis_ids=df_m["MisconceptionId"].tolist(),
        hards=hards_st,
        n_negatives=10,
    )
    # ds = make_cosent_dataset(
    #     q_texts=df_q["QuestionComplete"].tolist(),
    #     q_mis_ids=df_q["MisconceptionId"].tolist(),
    #     mis_texts=df_m["MisconceptionText"].tolist(),
    #     mis_ids=df_m["MisconceptionId"].tolist(),
    #     hards=hards_st,
    #     n_negatives=10,
    # )

    q, mis, mapping = make_ir_evaluator_dataset(df_val, all_mis_texts)
    # 1. Load a model to finetune with 2. (Optional) model card data
    # -- done upper

    # 3. Load a dataset to finetune on
    dataset = load_dataset("sentence-transformers/all-nli", "triplet")
    # -- done upper

    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/gte-large-en-1",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="gte-large-en-1",  # Will be used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    # dev_evaluator = TripletEvaluator(
    #     anchors=eval_dataset["anchor"],
    #     positives=eval_dataset["positive"],
    #     negatives=eval_dataset["negative"],
    #     name="all-nli-dev",
    # )
    # dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds2,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # (Optional) Evaluate the trained model on the test set
    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="all-nli-test",
    )
    test_evaluator(model)

    # 8. Save the trained model
    model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

    # 9. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub("mpnet-base-all-nli-triplet")


if __name__ == "__main__":
    main()
