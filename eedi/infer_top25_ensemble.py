import gc
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from eedi.datasets import make_nice_df
from eedi.helpers import last_token_pool


def get_embeddings_in_batches(
    model,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    batch_size: int,
    desc: str,
):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i : i + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"],  # type: ignore
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


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


def anhvth226_flow(
    df_test: pd.DataFrame,
    df_mis: pd.DataFrame,
    model_path: str,
    lora_path: str,
    tokenizer_path: str,
) -> tuple[Tensor, Tensor]:
    # load dataset
    queries = df_test.apply(anhvth226_template, axis=1).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    # load model and tokenizer
    model = AutoModel.from_pretrained(
        model_path, device_map=0, torch_dtype=torch.float16, load_in_4bit=False
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, lora_path)
    # batch infer both
    q_embeds = get_embeddings_in_batches(
        model, tokenizer, queries, max_length=320, batch_size=4, desc="anhvth226 Q"
    )
    m_embeds = get_embeddings_in_batches(
        model,
        tokenizer,
        misconceptions,
        max_length=320,
        batch_size=4,
        desc="anhvth226 M",
    )
    return q_embeds, m_embeds


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


def mschoo_flow(
    df_test: pd.DataFrame,
    df_mis: pd.DataFrame,
    model_path: str,
    lora_path: str,
    tokenizer_path: str,
) -> tuple[Tensor, Tensor]:
    # load dataset
    queries = df_test.apply(mschoo_template, axis=1).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    # load model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, lora_path)
    # batch infer both
    q_embeds = get_embeddings_in_batches(
        model, tokenizer, queries, max_length=512, batch_size=4, desc="mschoo Q"
    )
    m_embeds = get_embeddings_in_batches(
        model, tokenizer, misconceptions, max_length=512, batch_size=4, desc="mschoo M"
    )
    return q_embeds, m_embeds


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


def zuoyouzuo_flow(
    df_test: pd.DataFrame,
    df_mis: pd.DataFrame,
    model_path: str,
    lora_path: str,
    tokenizer_path: str,
) -> tuple[Tensor, Tensor]:
    # load dataset
    queries = df_test.apply(zuoyouzuo_template, axis=1).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    # load model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, config)
    d = torch.load(lora_path, map_location=model.device)
    model.load_state_dict(d, strict=False)
    model = model.merge_and_unload()
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # batch infer both
    q_embeds = get_embeddings_in_batches(
        model, tokenizer, queries, max_length=512, batch_size=4, desc="zuoyouzuo Q"
    )
    m_embeds = get_embeddings_in_batches(
        model,
        tokenizer,
        misconceptions,
        max_length=512,
        batch_size=4,
        desc="zuoyouzuo M",
    )
    return q_embeds, m_embeds


def main():
    # load dataset
    df_mis = pd.read_csv("data/misconception_mapping.csv")
    df_test = pd.read_csv("data/test.csv")
    df_test = make_nice_df(df_test)

    all_q_embeds = []
    all_m_embeds = []

    # (1) anhvth226
    q_embeds1, m_embeds1 = anhvth226_flow(
        df_test,
        df_mis,
        model_path="models/model_orig",
        lora_path="models/model_lora",
        tokenizer_path="models/model_orig",
    )
    all_q_embeds.append(q_embeds1)
    all_m_embeds.append(m_embeds1)
    gc.collect()
    torch.cuda.empty_cache()

    # (2) mschoo
    q_embeds2, m_embeds2 = mschoo_flow(
        df_test,
        df_mis,
        model_path="models/model_orig",
        lora_path="models/model_lora",
        tokenizer_path="models/model_orig",
    )
    all_q_embeds.append(q_embeds2)
    all_m_embeds.append(m_embeds2)
    gc.collect()
    torch.cuda.empty_cache()

    # (3) zuoyouzuo
    q_embeds3, m_embeds3 = zuoyouzuo_flow(
        df_test,
        df_mis,
        model_path="models/model_orig",
        lora_path="models/model_lora",
        tokenizer_path="models/model_orig",
    )
    all_q_embeds.append(q_embeds3)
    all_m_embeds.append(m_embeds3)
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
    main()
