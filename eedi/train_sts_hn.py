import json
import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from html import parser
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional

import pandas as pd
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModel, AutoTokenizer, BertModel, TrainingArguments

from eedi.callbacks import IterativeHNMiningCallback
from eedi.datasets import EvalDataset, TrainDataset, hn_mine_hf, make_complete_query
from eedi.losses import MultipleNegativesRankingLoss
from eedi.trainer import MyTrainer
from eedi.utils import wib_now


@dataclass
class Args:
    paraphrased_path: Path
    model: str
    token_pool: Literal["first", "last"]
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
    ac = Accelerator()

    # 1. load model
    # good lora blog: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

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
    # cache hard negative mining, this is just for fast dev iteration
    with ac.main_process_first():
        cache = Path(f"hards_{args.model.replace('/', '_')}.json")
        if cache.exists():
            print("loading from cache")
            with open(cache, "r") as f:
                hards_st = json.load(f)
        else:
            print("no hard negative cache, precomputing")
            hards_st = hn_mine_hf(
                model,
                tokenizer,  # type: ignore
                q_texts=df_train["QuestionComplete"].tolist(),
                q_mis_ids=df_train["MisconceptionId"].tolist(),
                mis_texts=df_mis["MisconceptionText"].tolist(),
                mis_ids=df_mis["MisconceptionId"].tolist(),
                k=100,
                bs=4,
                token_pool=args.token_pool,
                device=ac.device,
            )
            with open(cache, "w") as f:
                json.dump(hards_st, f)

    # 3. apply lora AFTER hn mining to make sure the model prediction not random bc of lora
    if args.lora_rank is not None:
        print("using lora")
        target_modules = get_target_modules(model)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,  # just set to 2 * alpha as a rule of thumb
            lora_dropout=0.2,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("not using lora")

    # 4. make datasets
    train_dataset = TrainDataset(
        q_texts=df_train["QuestionComplete"].tolist(),
        q_mis_ids=df_train["MisconceptionId"].tolist(),
        mis_texts=df_mis["MisconceptionText"].tolist(),
        mis_ids=df_mis["MisconceptionId"].tolist(),
        hards=hards_st,
        n_negatives=10,
    )
    eval_dataset = EvalDataset(
        q_texts=df_train["QuestionComplete"].tolist(),
        q_mis_ids=df_train["MisconceptionId"].tolist(),
        mis_texts=df_mis["MisconceptionText"].tolist(),
    )

    # 6. loss
    loss = MultipleNegativesRankingLoss()

    # 7. train!
    # TODO check hf docs
    training_args = TrainingArguments(
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
    # TODO 4 and 100 comes from the orignal hn mine call
    # TODO the callback need to be changed to standard hf model and tokenizer
    ihnm_callback = IterativeHNMiningCallback(
        bs=4, top_k_negatives=100, token_pool=args.token_pool
    )
    # TODO fic
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,  # normally not needed if you use data collator, but i need it for iterative hn mining callback
        args=training_args,
        data_collator=None,  # TODO inject using the tokenizer
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[ihnm_callback],
    )
    trainer.train()

    # 8. print final evaluation
    print("=== FINAL VAL RESULT ===")
    pprint(final_val_result)

    # 9. save the trained model
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
        "--token-pool",
        type=str,
        required=True,
        choices=["first", "last"],
        help="What token used for pooling, for encoders usually first (CLS), for decoders usually last (EOS)",
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
