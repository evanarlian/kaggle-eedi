from typing import Literal

from transformers import PreTrainedModel, Trainer

from eedi.datasets import EvalDataset
from eedi.helpers import batched_inference, last_token_pool
from eedi.losses import MultipleNegativesRankingLoss


class MyTrainer(Trainer):
    def __init__(self, token_pool: Literal["first", "last"], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = MultipleNegativesRankingLoss()
        self.token_pool: Literal["first", "last"] = token_pool

    def compute_loss(
        self, model: PreTrainedModel, inputs: dict, return_outputs: bool = False
    ):
        out_anchor = model(**inputs["anchor"])
        out_pos_neg = model(**inputs["pos_neg"])
        if self.token_pool == "first":
            out_anchor_pooled = out_anchor["last_hidden_state"][:, 0]
            out_pos_neg_pooled = out_pos_neg["last_hidden_state"][:, 0]
        elif self.token_pool == "last":
            out_anchor_pooled = last_token_pool(
                out_anchor["last_hidden_state"], inputs["anchor"]["attention_mask"]
            )
            out_pos_neg_pooled = last_token_pool(
                out_pos_neg["last_hidden_state"], inputs["pos_neg"]["attention_mask"]
            )
        else:
            assert False, "impossible during Trainer.compute_loss"
        return self.loss(out_anchor_pooled, out_pos_neg_pooled)

    def evaluate(
        self,
        eval_dataset: EvalDataset,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        device = next(self.model.parameters()).device
        batched_inference(
            self.model,
            self.tokenizer,
            texts=eval_dataset.q_texts,
            bs=4,
            token_pool="first",
            device=device,
            desc="eval",
        )
        
        return {"_fake": 0.7}
