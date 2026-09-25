"""Ranking metrics."""

import torch


def metrics(topk, targets, ks=(5, 10)):
    """Recall@K and NDCG@K for one target item per user.

    topk: [n_users, max(ks)] ranked item ids. targets: [n_users] true next item.
    """
    hits = (topk == targets[:, None]).float()
    discount = 1 / torch.log2(torch.arange(2, hits.shape[1] + 2, dtype=torch.float))
    out = {}
    for k in ks:
        out[f"recall@{k}"] = hits[:, :k].sum(1).mean().item()
        out[f"ndcg@{k}"] = (hits[:, :k] * discount[:k]).sum(1).mean().item()
    return out
