import torch
import torch.nn.functional as F
from torch import Tensor


def late_interaction(
    queries: Tensor, docs: Tensor, query_mask: Tensor, doc_mask: Tensor
) -> Tensor:
    """Apply batched colbert late interaction with cosine similarity.
    Both queries and docs must be L2-normalized beforehand for performance reasons.

    Args:
        queries (Tensor): Queries, size (nq, n_q_tok, emb_sz).
        docs (Tensor): Documents, size (nd, n_d_tok, emb_sz)
        query_mask (Tensor): Query attn mask, size (nq, n_q_tok)
        doc_mask (Tensor): Document attn mask, size (nd, n_d_tok)

    Returns:
        Tensor: Late interaction, maxsim applied. Size (nq, nd)
    """
    # convert both input tensors and masks to 4d tensor
    #   (nq, 1, n_q_tok, emb_sz)
    #   (1, nd, emb_sz, n_d_tok)
    # = (nq, nd, n_q_tok, n_d_tok)
    li = queries[:, None] @ docs[None, :].transpose(-2, -1)
    mask = query_mask[:, None, :, None] * doc_mask[None, :, None, :]
    # temporarily reduce the padding value to under -1, to make padding impossible to win max operation
    # -2.0 will work since after L2-norm/cossim, max possible value is 1.0
    li += (mask - 1.0) * 2
    li = li.max(-1).values
    # bring back padding to 0.0 just before sum
    li = (li * mask.max(-1).values).sum(-1)
    return li


def manual_late_interaction(queries: list[Tensor], docs: list[Tensor]) -> Tensor:
    """Loopy version of colbert late interaction with cosine similarity.
    Don't use! This version is only for debugging and correctness check.

    Args:
        queries (list[Tensor]): Queries, each size (n_q_tok, emb_sz)
        docs (list[Tensor]): Documents, each size (n_d_tok, emb_sz)

    Returns:
        Tensor: Late interaction, maxsim applied. Size (nq, nd)
    """
    li = torch.zeros(len(queries), len(docs))
    for i, q in enumerate(queries):
        for j, d in enumerate(docs):
            q = F.normalize(q, dim=-1)
            d = F.normalize(d, dim=-1)
            li[i, j] = (q @ d.T).max(-1).values.sum(-1)
    return li
