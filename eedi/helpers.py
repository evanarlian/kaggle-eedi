from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # TODO i dont like this
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
    model,
    tokenizer,
    texts: list[str],
    bs: int,
    token_pool: Literal["first", "last"],
    device,  # TODO what is the accelerator device name?
    desc: str,
) -> Tensor:
    """Basically SentenceTransformer.encode, but consume less vram."""
    # TODO add token pool, review the last_token_pool code because i dont like it
    embeddings = []
    for i in tqdm(range(0, len(texts), bs), desc=desc):
        # max_length=256 comes from plotting the complete question text, and 256 covers 99%
        # TODO check again abt this statement!
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
                outputs["last_hidden_state"], encoded["attention_mask"]
            )
        emb = F.normalize(emb, p=2, dim=-1)
        embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings)
    return embeddings
