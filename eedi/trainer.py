from typing import Literal

import torch
from transformers import PreTrainedModel, Trainer

from eedi.datasets import EvalDataset
from eedi.helpers import batched_inference, last_token_pool
from eedi.losses import MultipleNegativesRankingLoss
from eedi.metrics import map_at_k, rank_dist


class MyTrainer(Trainer):
    def __init__(
        self, token_pool: Literal["first", "last"], bs: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss = MultipleNegativesRankingLoss()
        self.token_pool: Literal["first", "last"] = token_pool
        self.bs = bs

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

    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        self.eval_dataset: EvalDataset
        is_training_state = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        q_embs = batched_inference(
            self.model,  # type: ignore
            self.tokenizer,  # type: ignore
            texts=self.eval_dataset.q_texts,
            bs=self.bs,
            token_pool=self.token_pool,
            device=device,
            desc="eval q",
        )
        mis_embs = batched_inference(
            self.model,  # type: ignore
            self.tokenizer,  # type: ignore
            texts=self.eval_dataset.mis_texts,
            bs=self.bs,
            token_pool=self.token_pool,
            device=device,
            desc="eval mis",
        )
        similarities = q_embs @ mis_embs.T
        labels = torch.tensor(self.eval_dataset.q_mis_ids)
        map_at_25 = map_at_k(labels, similarities, k=25)
        map_at_5 = map_at_k(labels, similarities, k=5)
        map_at_1 = map_at_k(labels, similarities, k=1)
        output_metrics = {
            "eval/cosine_map@25": map_at_25,
            "eval/cosine_map@5": map_at_5,
            "eval/cosine_map@1": map_at_1,
        }
        self.log(output_metrics)
        rank_distributions = rank_dist(labels, similarities, k=25)
        print("====== RANK DIST =======")
        for rank, count in rank_distributions.items():
            print(f"rank {rank}: {count} ({count/labels.size(0):.2%})")
        self.model.train(is_training_state)
        return output_metrics
