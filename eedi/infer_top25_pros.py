
import gc
import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from eedi.datasets import  make_nice_df
from eedi.helpers import batched_inference


@dataclass
class Args:
    dataset_dir: Path
    model_paths: list[str]
    lora_paths: list[str]


def anhvth226_template(row: pd.Series) -> str:
    template = """<instruct>Given a math multiple-choice problem with a student's wrong answer, retrieve the math misconceptions
<query>Question: {question}
    
SubjectName: {subject}
ConstructName: {construct}
Correct answer: {correct}
Student wrong answer: {wrong}
<response>"""
    return template.format(
        question=row["QuestionText"],
        subject=row["SubjectName"],
        construct=row["ConstructName"],
        correct=row["CorrectText"],
        wrong=row["WrongText"],
    )


def mschoo_template(row: pd.Series) -> str:
    template = """Instruct: Given a math question with correct answer and a misconcepted incorrect answer, retrieve the most accurate misconception for the incorrect answer.
Query: ### SubjectName: {subject}
### ConstructName: {subject}
### Question: {question}
### Correct Answer: {correct}
### Misconcepte Incorrect answer: {wrong}
<response>"""
    return template.format(
        question=row["QuestionText"],
        subject=row["SubjectName"],
        construct=row["ConstructName"],
        correct=row["CorrectText"],
        wrong=row["WrongText"],
    )


def zuoyouzuo_template(row: pd.Series) -> str:
    template = """Instruct: Given a math question with correct answer and a misconcepted incorrect answer, retrieve the most accurate misconception for the incorrect answer.
Query: ### SubjectName: {subject}
### ConstructName: {subject}
### Question: {question}
### Correct Answer: {correct}
### Misconcepte Incorrect answer: {wrong}"""
    return template.format(
        question=row["QuestionText"],
        subject=row["SubjectName"],
        construct=row["ConstructName"],
        correct=row["CorrectText"],
        wrong=row["WrongText"],
    )


def get_embeddings(
    model_path: str,
    lora_path: str,
    token_pool: Literal["first", "last"],
    q_texts: list[str],
    mis_texts: list[str],
    bs: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    assert len(mis_texts) == 2587
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map=device,
    )
    model.load_adapter(lora_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    q_embeds = batched_inference(
        model,
        tokenizer,
        q_texts,
        bs=bs,
        token_pool=token_pool,
        device=device,
        desc="infer q",
    ).numpy()
    m_embeds = batched_inference(
        model,
        tokenizer,
        mis_texts,
        bs=bs,
        token_pool=token_pool,
        device=device,
        desc="infer mis",
    ).numpy()
    return q_embeds, m_embeds


def main(args: Args):
    device = torch.device("cuda:0")

    # load dataset
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_test = pd.read_csv(args.dataset_dir / "test.csv")
    df_test = make_nice_df(df_test)
    df_test["anhvth226Prompt"] = df_test.apply(anhvth226_template, axis=1)
    df_test["mschooPrompt"] = df_test.apply(mschoo_template, axis=1)
    df_test["zuoyouzuoPrompt"] = df_test.apply(zuoyouzuo_template, axis=1)

    # load big model once
    # NOTE: dont do something stupid and just reload the model from scratch
    # dont try to optimize speed first

    model = 
    all_q_embeds = []
    all_m_embeds = []

    # do it manually
    for model_path, lora_path, token_pool in zip(
        args.model_paths, args.lora_paths, args.token_pools
    ):
        q_embeds, m_embeds = get_embeddings(
            model_path,
            lora_path,
            token_pool,
            q_texts=df_test["QuestionComplete"].tolist(),
            mis_texts=df_mis["MisconceptionName"].tolist(),
            bs=16,
            device=device,
        )
        assert len(q_embeds) == df_test.shape[0]
        all_q_embeds.append(q_embeds)
        all_m_embeds.append(m_embeds)
        gc.collect()
        torch.cuda.empty_cache()

    # concat sideways
    all_q_embeds = np.concatenate(all_q_embeds, axis=-1)
    all_m_embeds = np.concatenate(all_m_embeds, axis=-1)

    # calc
    nn = NearestNeighbors(n_neighbors=25, algorithm="brute", metric="cosine")
    nn.fit(all_m_embeds)
    dist, topk_mis = nn.kneighbors(all_q_embeds)

    # save
    savepath = "top25_miscons.json"
    with open(savepath, "w") as f:
        json.dump(topk_mis.tolist(), f)
    print(f"saved to {savepath}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder"
    )
    parser.add_argument(
        "--model-paths",
        required=True,
        type=str,
        nargs="+",
        help="Base model paths or names from HF (multiple allowed)",
    )
    parser.add_argument(
        "--lora-paths",
        required=True,
        type=str,
        nargs="+",
        help="Embedding model paths or names from HF (multiple allowed)",
    )
    args = parser.parse_args()
    args = Args(**vars(args))
    print(args)
    assert len(args.model_paths) == len(args.lora_paths)
    t0 = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f} secs.")
