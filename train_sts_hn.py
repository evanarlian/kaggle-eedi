import json
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

from my_datasets import Dataset as HFDataset
from my_datasets import (
    hn_mine_st,
    load_dataset,
    make_ir_evaluator_dataset,
    make_mnr_dataset,
)


@dataclass
class Args:
    dataset_path: Path
    model: Path
    output_dir: Path


def main():
    # 1. load model
    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # 2. load dataset
    # load paraphrased misconception
    df_mis = pd.read_csv("data/eedi-paraphrased/misconception_mapping.csv")
    orig_mis = (
        df_mis[~df_mis["MisconceptionAiCreated"]]
        .sort_values("MisconceptionId")["MisconceptionText"]
        .tolist()
    )
    assert len(orig_mis) == 2587
    # load paraphrased train
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
    # split to train (w/ miscons) and val (w/o miscons)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["QuestionId"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx]
    df_val = df_val[~df_val["QuestionAiCreated"]].reset_index(drop=True)

    # mine hard negatives (TODO this must be changed later to iterative)
    cache = Path("hards.json")
    if cache.exists():
        print("loading from cache")
        with open(cache, "r") as f:
            hards_st = json.load(f)
    else:
        print("no cache, precomputing")
        hards_st = hn_mine_st(
            model,
            q_texts=df_train["QuestionComplete"].tolist(),
            q_mis_ids=df_train["MisconceptionId"].tolist(),
            mis_texts=df_mis["MisconceptionText"].tolist(),
            mis_ids=df_mis["MisconceptionId"].tolist(),
            k=100,
            bs=4,
        )
        with open(cache, "w") as f:
            json.dump(hards_st, f)

    # TODO numerize all
    # make hf dataset suitable for sentence transformers
    train_dataset = make_mnr_dataset(
        q_texts=df_train["QuestionComplete"].tolist(),
        q_mis_ids=df_train["MisconceptionId"].tolist(),
        mis_texts=df_mis["MisconceptionText"].tolist(),
        mis_ids=df_mis["MisconceptionId"].tolist(),
        hards=hards_st,
        n_negatives=10,  # TODO parametrize ?
    )

    # make evaluator
    q, mis, mapping = make_ir_evaluator_dataset(df_val, orig_mis)
    evaluator = InformationRetrievalEvaluator(
        queries=q,
        corpus=mis,
        relevant_docs=mapping,
        map_at_k=[1, 3, 5, 10, 25],
        batch_size=4,
        show_progress_bar=True,
    )

    loss = MultipleNegativesRankingLoss(model)
    training_args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/gte-large-en-1-outdir",  # TODO what is the difference vs save pretrained?
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
        run_name="gte-large-en-1-wandb",  # Will be used in W&B if `wandb` is installed
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # TODO dont need eval dataset for now, really?
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    evaluator(model)

    # 8. Save the trained model
    model.save_pretrained("models/gte-large-en-1-local")

    # 9. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub("gte-large-en-1-hfpush")


if __name__ == "__main__":
    main()
