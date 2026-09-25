"""Popularity baseline and SASRec, a small causal transformer that reads a user's item history.

Run:
    uv run python sasrec.py --model pop
    uv run python sasrec.py --model sasrec --loss bce   # original paper: one random negative item per step
    uv run python sasrec.py --model sasrec --loss ce    # softmax over all items
Add --split time to use the global time split instead of the per-user last-item split.
"""

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import leave_one_out, load, time_cutoffs, time_split
from evaluate import metrics

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
K = 10


def pad(seqs, max_len):
    """Keep the last max_len items of each sequence and pad on the right with 0."""
    out = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        out[i, : len(s)] = torch.tensor(s)
    return out


class SASRec(nn.Module):
    def __init__(self, n_items, d=64, n_layers=2, n_heads=1, max_len=50, dropout=0.5):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, n_heads, d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)

    def forward(self, seq):
        """seq: [B, L] right-padded item ids. Returns [B, L, d], where position t only sees items 0..t.

        Right padding means real items never attend to padding, so the causal mask is the only mask needed.
        """
        L = seq.shape[1]
        x = self.item_emb(seq) * self.item_emb.embedding_dim**0.5 + self.pos_emb.weight[:L]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=seq.device)
        return self.encoder(self.drop(x), mask=mask, is_causal=True)

    def scores(self, h):
        s = h @ self.item_emb.weight.T
        s[:, 0] = -torch.inf
        return s


@torch.no_grad()
def evaluate(model, pairs, max_len, batch=1024):
    """Rank every item for each (history, target) pair and return Recall/NDCG at 5 and 10."""
    model.eval()
    tops = []
    for i in range(0, len(pairs), batch):
        seq = pad([h for h, _ in pairs[i : i + batch]], max_len).to(DEVICE)
        last = model(seq)[torch.arange(len(seq)), (seq > 0).sum(1) - 1]
        tops.append(model.scores(last).topk(K).indices.cpu())
    return metrics(torch.cat(tops), torch.tensor([t for _, t in pairs]))


def train_sasrec(train, valid, n_items, loss, max_len=50, epochs=1000, patience=20, batch=128, lr=1e-3):
    torch.manual_seed(0)
    model = SASRec(n_items, max_len=max_len).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    data = pad(train, max_len + 1)  # input is items 0..n-2, target is items 1..n-1
    best, best_state, best_epoch = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(data)).split(batch):
            x, y = data[idx, :-1].to(DEVICE), data[idx, 1:].to(DEVICE)
            h, y = model(x)[y > 0], y[y > 0]
            if loss == "ce":
                step_loss = F.cross_entropy(model.scores(h), y)
            else:
                # ponytail: negatives are uniform over all items and may hit the user's history (about 0.1% chance).
                neg = torch.randint(1, n_items + 1, y.shape, device=DEVICE)
                logits = torch.cat([(h * model.item_emb(y)).sum(-1), (h * model.item_emb(neg)).sum(-1)])
                labels = torch.cat([torch.ones_like(y), torch.zeros_like(y)]).float()
                step_loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad()
            step_loss.backward()
            opt.step()
            total += step_loss.item()
        if valid is None:  # fixed number of epochs, no early stopping
            print(f"epoch {epoch:3d}  loss {total:8.2f}", flush=True)
            continue
        score = evaluate(model, valid, max_len)["ndcg@10"]
        print(f"epoch {epoch:3d}  loss {total:8.2f}  valid ndcg@10 {score:.4f}", flush=True)
        if score > best:
            best, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_epoch


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["pop", "sasrec"], default="sasrec")
    p.add_argument("--loss", choices=["bce", "ce"], default="bce")
    p.add_argument("--split", choices=["loo", "time"], default="loo", help="loo matches the TIGER paper, time has no leakage across users")
    args = p.parse_args()

    seqs, times, items, _ = load()
    if args.split == "loo":
        train, valid, test = leave_one_out(seqs)
        final_train = train
    else:
        t_valid, t_test = time_cutoffs(times)
        train, valid, test = time_split(seqs, times, t_valid, t_test)
        # Settings are picked on validation, then the final model also learns from the validation period.
        final_train = time_split(seqs, times, t_test, t_test)[0]
    suffix = "" if args.split == "loo" else "_time"
    start = time.time()
    if args.model == "pop":
        name = "pop" + suffix
        top = [i for i, _ in Counter(i for s in final_train for i in s).most_common(K)]
        result = {"test": metrics(torch.tensor(top).expand(len(test), K), torch.tensor([t for _, t in test]))}
    else:
        name = f"sasrec_{args.loss}{suffix}"
        model, best_epoch = train_sasrec(train, valid, len(items), args.loss)
        result = {"best_epoch": best_epoch, "valid": evaluate(model, valid, 50)}
        if final_train is not train:
            model, _ = train_sasrec(final_train, None, len(items), args.loss, epochs=best_epoch + 1)
        result["test"] = evaluate(model, test, 50)
    result["minutes"] = round((time.time() - start) / 60, 1)

    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(json.dumps(result, indent=2))
    print(name, json.dumps(result, indent=2))
