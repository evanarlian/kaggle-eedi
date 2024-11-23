import torch
from torch import Tensor


def map_at_k(labels: Tensor, similarities: Tensor, k: int) -> float:
    """Calculates mAP@k metric.

    Args:
        labels (Tensor): ground truth tensor, size (n_rows,).
        similarities (Tensor): similarity score tensor, size (n_rows, n_misconceptions).
        k (int): k for mAP@k.

    Returns:
        float: mAP@k score
    """
    min_k = min(similarities.size(-1), k)
    val, idx = similarities.topk(min_k)
    mask = labels[:, None] == idx
    denominator = torch.arange(min_k) + 1
    mapk = 1 / denominator[None, :] * mask
    return mapk.sum(-1).mean().item()


def rank_dist(labels: Tensor, similarities: Tensor, k: int) -> dict[int, int]:
    """Calculates rank distributions from similarities.

    Args:
        labels (Tensor): ground truth tensor, size (n_rows,).
        similarities (Tensor): similarity score tensor, size (n_rows, n_misconceptions).
        k (int): k for mAP@k.

    Returns:
        Counter:
            Rank distributions, ranging from 1 to k inclusive.
            Will also count rank -1, this indicates rank above k.
    """
    d = dict.fromkeys([-1] + list(range(1, k + 1)), 0)
    min_k = min(similarities.size(-1), k)
    val, idx = similarities.topk(min_k)
    mask = labels[:, None] == idx
    aranger = torch.arange(1, min_k + 1).expand(labels.size(0), min_k)
    ranks = aranger[mask].tolist()
    d[-1] = labels.size(0) - len(ranks)
    for rank in ranks:
        d[rank] += 1
    assert sum(d.values()) == labels.size(0)
    return d
