import re
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from transformers import (
    BertModel,
    MistralModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def get_lora_target_modules(model) -> list[str]:
    if isinstance(model, BertModel):
        return ["query", "key", "value", "dense"]
    elif re.search(r"Alibaba-NLP.+NewModel", str(type(model))):
        return ["qkv_proj", "o_proj", "up_gate_proj", "down_proj"]
    elif isinstance(model, MistralModel):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # fmt: off
    raise ValueError(
        f"Model with type {type(model)} is unsupported, please manually inspect and add lora modules."
    )


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # copied from: https://huggingface.co/Salesforce/SFR-Embedding-2_R
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


@torch.inference_mode()
def batched_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    bs: int,
    token_pool: Literal["first", "last"],
    device: torch.device,
    desc: Optional[str],
) -> Tensor:
    """Basically SentenceTransformer.encode, but consume less vram."""
    embeddings = []
    for i in tqdm(range(0, len(texts), bs), desc=desc, disable=desc is None):
        # max_length=256 comes from plotting the complete question text, and 256 covers 99%
        encoded = tokenizer(
            texts[i : i + bs],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**encoded)
        if token_pool == "first":
            emb = outputs["last_hidden_state"][:, 0]  # cls token
        elif token_pool == "last":
            emb = last_token_pool(
                outputs["last_hidden_state"],
                encoded["attention_mask"],  # type: ignore
            )
        emb = F.normalize(emb, p=2, dim=-1)
        embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings)
    return embeddings
