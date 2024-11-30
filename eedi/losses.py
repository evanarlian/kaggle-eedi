import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0) -> None:
        """MultipleNegativesRankingLoss modified from sentence transformer. This version
        only operates on tensors and does not require the model.
        """
        super().__init__()
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, anchor_emb: Tensor, pos_negs_emb: Tensor) -> Tensor:
        scores = (
            F.cosine_similarity(anchor_emb[:, None], pos_negs_emb, dim=-1) * self.scale
        )
        # Example a[i] should match with b[i]
        range_labels = torch.arange(0, scores.size(0), device=scores.device)
        return self.cross_entropy_loss(scores, range_labels)


class MultiplePositiveNegativeRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0) -> None:
        """My idea of another loss function that can handle multiple positives and negatives
        per anchor. This loss also handles assumption from MultipleNegativesRankingLoss,
        jth batch's negatives is also ith batch's negatives (which is not true).
        """
        raise NotImplementedError()
        super().__init__()
        # self.scale = scale
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, anchor_emb: Tensor, pos_negs_emb: Tensor) -> Tensor:
        raise NotImplementedError()
        # scores = (
        #     F.cosine_similarity(anchor_emb[:, None], pos_negs_emb, dim=-1) * self.scale
        # )
        # # Example a[i] should match with b[i]
        # range_labels = torch.arange(0, scores.size(0), device=scores.device)
        # return self.cross_entropy_loss(scores, range_labels)


def main():
    anchor_emb = torch.randn(17, 100)  # anchor
    pos_negs_emb = torch.randn(17 * 4, 100)  # pos, neg, neg, neg
    mnrl = MultipleNegativesRankingLoss()
    loss = mnrl(anchor_emb, pos_negs_emb)
    print(loss)


if __name__ == "__main__":
    main()
