import json
import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from html import parser
from pathlib import Path
from pprint import pprint
from typing import Optional

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertModel

from eedi.callbacks import IterativeHNMiningCallback
from eedi.my_datasets import (
    TrainDatasetProxy,
    hn_mine_sbert,
    make_complete_query,
    make_ir_evaluator_dataset,
)
from eedi.utils import local_rank, wib_now


@dataclass
class Args:
    paraphrased_path: Path
    model: str
    per_device_bs: int
    lr: float
    n_epochs: int
    lora_rank: Optional[int]
    run_name: str


def get_target_modules(model) -> list[str]:
    if isinstance(model, BertModel):
        return ["query", "key", "value", "dense"]
    elif re.search(r"Alibaba-NLP.+NewModel", str(type(model))):
        return ["qkv_proj", "o_proj", "up_gate_proj", "down_proj"]
    raise ValueError(
        f"Model with type {type(model)} is unsupported, please manually inspect and add lora modules."
    )


def main(args: Args):
    print("LOCAL_RANK", local_rank())
    torch.cuda.set_device(local_rank())  # prevent default cuda:0 in every .cuda() call
    # TODO how to load lora once in multi gpu trianing?
    # TODO how to do EVERYTHING in multi gpu training to avoid duplicate work?
    # 1. load model along with lora
    # good lora blog: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
    model = SentenceTransformer(args.model, trust_remote_code=True)
    print("🔥", next(model.parameters()).device)
    if args.lora_rank is not None:
        print("using lora")
        # in sentence transformers, model[0]._modules["auto_model"] is the location of original model
        lora_modules = get_target_modules(model[0]._modules["auto_model"])
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=lora_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,  # just set to 2 * alpha as a rule of thumb
            lora_dropout=0.2,
        )
        model[0]._modules["auto_model"] = get_peft_model(
            model[0]._modules["auto_model"],  # type: ignore
            peft_config,
        )
        model[0]._modules["auto_model"].print_trainable_parameters()
    else:
        print("not using lora")
    # 2. load dataset
    # load paraphrased misconception
    df_mis = pd.read_csv(args.paraphrased_path / "misconception_mapping.csv")
    orig_mis = (
        df_mis[~df_mis["MisconceptionAiCreated"]]
        .sort_values("MisconceptionId")["MisconceptionText"]
        .tolist()
    )
    assert len(orig_mis) == 2587
    # load paraphrased train
    df = pd.read_csv(args.paraphrased_path / "train.csv")
    df["QuestionComplete"] = df.apply(make_complete_query, axis=1)
    # split to train (w/ miscons) and val (w/o miscons)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["QuestionId"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx]
    df_val = df_val[~df_val["QuestionAiCreated"]].reset_index(drop=True)

    # TODO caching is not correct, this is just for fast dev iteration
    cache = Path(f"hards_{args.model}.json")
    if cache.exists():
        print("loading from cache")
        with open(cache, "r") as f:
            hards_st = json.load(f)
    else:
        print("no hard negative cache, precomputing")
        hards_st = hn_mine_sbert(
            model,
            q_texts=df_train["QuestionComplete"].tolist(),
            q_mis_ids=df_train["MisconceptionId"].tolist(),
            mis_texts=df_mis["MisconceptionText"].tolist(),
            mis_ids=df_mis["MisconceptionId"].tolist(),
            k=100,
            bs=4,
            tqdm=local_rank() == 0,
        )
        with open(cache, "w") as f:
            json.dump(hards_st, f)

    # TODO numerize all
    # make hf dataset suitable for sentence transformers and iterative hard negative mining
    train_dataset = TrainDatasetProxy(
        q_texts=df_train["QuestionComplete"].tolist(),
        q_mis_ids=df_train["MisconceptionId"].tolist(),
        mis_texts=df_mis["MisconceptionText"].tolist(),
        mis_ids=df_mis["MisconceptionId"].tolist(),
        hards=hards_st,
        n_negatives=10,  # TODO check what is the best
    )

    # make evaluator
    q, mis, mapping = make_ir_evaluator_dataset(df_val, orig_mis)
    val_evaluator = InformationRetrievalEvaluator(
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
        output_dir=f"models/{args.run_name}",
        # Optional training parameters:
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch TODO check is this is the slow and memory hog culprit
        dataloader_drop_last=True,
        dataloader_num_workers=1,  # we are only getting text so it will be fast
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        logging_steps=500,
        run_name=args.run_name,  # Will be used in W&B if `wandb` is installed
    )
    ihnm_callback = IterativeHNMiningCallback(
        4, 100
    )  # TODO 4 and 100 comes from the orignal hn mine call
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # TODO dont need eval dataset for now, really?
        loss=loss,
        evaluator=val_evaluator,
        callbacks=[ihnm_callback],
    )
    trainer.train()

    final_val_result = val_evaluator(model)
    print("=== FINAL VAL RESULT ===")
    pprint(final_val_result)
    # 8. Save the trained model
    # TODO test loading the lora model from sentence transformers
    model.save_pretrained(f"models/{args.run_name}/last")
    model.push_to_hub(args.run_name, private=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--paraphrased-path",
        type=Path,
        required=True,
        help="Path to paraphrased dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pretrained model name from huggingface or local path",
    )
    parser.add_argument(
        "--per-device-bs",
        type=int,
        required=True,
        help="Batch size per gpu for both training and validation",
    )
    parser.add_argument("--lr", type=float, required=True, help="Peak learning rate")
    parser.add_argument(
        "--n-epochs", type=int, required=True, help="Number of trianing epochs"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        required=False,
        help="LoRA rank. Good baseline is 8, try to keep doubling this value",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="<run_name>_<time>. Time will be auto added, used for saving locally, wandb tracking, and upload to hf repo",
    )
    args = parser.parse_args()
    args = Args(**vars(args))
    os.environ["WANDB_PROJECT"] = "kaggle-eedi"
    args.run_name = f"{args.run_name}_{wib_now()}"
    print(args)
    main(args)
