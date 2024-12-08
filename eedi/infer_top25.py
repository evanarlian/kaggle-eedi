import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from eedi.datasets import make_complete_query, make_nice_df
from eedi.helpers import batched_inference


@dataclass
class Args: 
    dataset_dir: Path
    model_path: str
    lora_path: str
    token_pool: Literal["first", "last"]


def get_topk_misconception(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q_texts: list[str],
    mis_texts: list[str],
    k: int,
    bs: int,
    token_pool: Literal["first", "last"],
    device: torch.device,
) -> list[list[int]]:
    """Get topk misconceptions using brute force.

    Args:
        model (PreTrainedModel): Huggingface model.
        tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer.
        q_texts (list[str]): Question texts.
        q_mis_ids (list[int]): Ground truth misconception ids for the questions.
        mis_texts (list[str]): Misconception texts.
        mis_ids (list[int]): Misconception ids.
        k (int): Top k hard misconception ids per question.
        bs (int): Batch size.

    Returns:
        list[list[int]]:
            Hardest k misconception ids for each question.
    """
    assert len(mis_texts) == 2587
    m_embeds = batched_inference(
        model,
        tokenizer,
        mis_texts,
        bs=bs,
        token_pool=token_pool,
        device=device,
        desc="infer mis",
    ).numpy()
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="cosine")
    nn.fit(m_embeds)
    q_embeds = batched_inference(
        model,
        tokenizer,
        q_texts,
        bs=bs,
        token_pool=token_pool,
        device=device,
        desc="infer q",
    ).numpy()
    ranks = nn.kneighbors(q_embeds, return_distance=False)
    return ranks.tolist()  # type: ignore


def main(args: Args):
    # load model, TODO make sure bnb_config is the same during training
    device = torch.device("cuda:0")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map=device,
    )
    model.load_adapter(args.lora_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # load dataset
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_test = pd.read_csv(args.dataset_dir / "test.csv")
    df_test = make_nice_df(df_test)
    df_test["QuestionComplete"] = df_test.apply(make_complete_query, axis=1)

    # calc miscons
    topk_mis = get_topk_misconception(
        model,
        tokenizer,
        q_texts=df_test["QuestionComplete"].tolist(),
        mis_texts=df_mis["MisconceptionName"].tolist(),
        k=25,
        bs=4,
        token_pool=args.token_pool,
        device=device,
    )
    assert len(topk_mis) == df_test.shape[0]

    # save
    savepath = "top25_miscons.json"
    with open(savepath, "w") as f:
        json.dump(topk_mis, f)
    print(f"saved to {savepath}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder"
    )
    parser.add_argument(
        "--model-path", required=True, type=str, help="Base model from path or HF"
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        type=str,
        help="Embedding model path or name from HF",
    )
    parser.add_argument(
        "--token-pool",
        type=str,
        required=True,
        choices=["first", "last"],
        help="What token used for pooling, for encoders usually first (CLS), for decoders usually last (EOS)",
    )
    args = parser.parse_args()
    args = Args(**vars(args))
    print(args)
    t0 = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f} secs.")
