"""Generative recommender: a small T5 reads a user's history as semantic ID tokens and writes the next item's ID.

Each item is 4 tokens, one per semantic ID level. Beam search may only follow token paths that spell
a real item's ID, using a trie (a prefix tree of all valid IDs), so every output is a real item.

Run: uv run python genrec.py --split loo
"""

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import T5Config, T5ForConditionalGeneration

from data import DATA, get_split, load
from evaluate import metrics
from sasrec import DEVICE, pad

PAD, EOS, FIRST_CODE = 0, 1, 2  # token ids
CODEBOOK = 256
HISTORY = 20  # TIGER keeps the last 20 items
K = 10


def tiny_t5(vocab_size):
    """TIGER's model size: 4 encoder and 4 decoder layers, 6 heads of size 64, d_model 128."""
    config = T5Config(
        vocab_size=vocab_size, d_model=128, d_kv=64, d_ff=1024, num_layers=4, num_decoder_layers=4, num_heads=6,
        dropout_rate=0.1, feed_forward_proj="relu", pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD,
    )
    return T5ForConditionalGeneration(config)


def id_tokens(ids):
    """Row i holds item i's 4 tokens. Row 0 is padding. Level l's code c becomes token 2 + 256 * l + c."""
    assert max(c for i in ids for c in i) < CODEBOOK
    rows = [[FIRST_CODE + CODEBOOK * level + c for level, c in enumerate(i)] for i in ids]
    return torch.tensor([[PAD] * len(ids[0])] + rows)


def build_trie(tokens):
    """Map every prefix of a real item's tokens to the tokens that may come next."""
    trie = defaultdict(set)
    for row in tokens[1:].tolist():
        for k in range(len(row)):
            trie[tuple(row[:k])].add(row[k])
    return {prefix: sorted(nxt) for prefix, nxt in trie.items()}


@torch.no_grad()
def generate_topk(model, history, trie, k=K, beams=20):
    """Return, for each history row, the k best item token tuples found by trie-constrained beam search."""
    n_levels = max(map(len, trie)) + 1  # the longest prefix is one token short of a full ID
    out = model.generate(
        input_ids=history, attention_mask=history != PAD, num_beams=beams, num_return_sequences=k,
        max_new_tokens=n_levels, do_sample=False,
        prefix_allowed_tokens_fn=lambda _, prefix: trie[tuple(prefix[1:].tolist())],  # prefix[0] is the start token
    )
    return [list(map(tuple, out[i : i + k, 1:].tolist())) for i in range(0, len(out), k)]


def encode(histories, tokens):
    return tokens[pad(histories, HISTORY)].flatten(1)


@torch.no_grad()
def evaluate(model, pairs, tokens, trie, item_of, batch=256):
    model.eval()
    tops = []
    for i in range(0, len(pairs), batch):
        x = encode([h for h, _ in pairs[i : i + batch]], tokens).to(DEVICE)
        tops += [[item_of[t] for t in row] for row in generate_topk(model, x, trie)]
    return metrics(torch.tensor(tops), torch.tensor([t for _, t in pairs]))


def train_genrec(train, valid, tokens, trie, item_of, epochs=400, hours=None, eval_every=5, patience=4, batch=256, lr=1e-3):
    """Train on every (last 20 items, next item) pair in train. Stop when valid Recall@10 stops improving.

    If valid is None, train for exactly `epochs` epochs. Training also stops after `hours`, keeping the best model,
    so a run always finishes inside Kaggle's 12-hour session limit.
    """
    torch.manual_seed(0)
    model = tiny_t5(int(tokens.max()) + 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pairs = [(s[:t], s[t]) for s in train for t in range(1, len(s))]
    x, y = encode([h for h, _ in pairs], tokens), tokens[torch.tensor([t for _, t in pairs])]
    # ponytail: early stopping scores 2,000 validation users to keep beam search cheap; use all if choices look noisy.
    valid_sample = random.Random(0).sample(valid, min(2000, len(valid))) if valid else None
    best, best_state, best_epoch = -1.0, None, 0
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x)).split(batch):
            xb, yb = x[idx].to(DEVICE), y[idx].to(DEVICE)
            loss = model(input_ids=xb, attention_mask=xb != PAD, labels=yb).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        minutes = (time.time() - start) / 60
        if valid is None or (epoch + 1) % eval_every:
            print(f"epoch {epoch:3d}  loss {total:8.2f}  {minutes:6.1f} min", flush=True)
            continue
        score = evaluate(model, valid_sample, tokens, trie, item_of)["recall@10"]
        print(f"epoch {epoch:3d}  loss {total:8.2f}  {minutes:6.1f} min  valid recall@10 {score:.4f}", flush=True)
        if score > best:
            best, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience * eval_every:
            break
        if hours and minutes > hours * 60:
            print(f"stopping at the {hours}-hour limit", flush=True)
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_epoch


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["loo", "time"], default="loo")
    p.add_argument("--hours", type=float, help="stop picking the epoch count after this many hours")
    args = p.parse_args()

    seqs, times, items, _ = load()
    train, valid, test, final_train = get_split(args.split, seqs, times)
    tokens = id_tokens(json.loads((DATA / f"semantic_ids_{args.split}.json").read_text()))
    trie = build_trie(tokens)
    item_of = {tuple(row): i + 1 for i, row in enumerate(tokens[1:].tolist())}

    start = time.time()
    model, best_epoch = train_genrec(train, valid, tokens, trie, item_of, hours=args.hours)
    result = {"best_epoch": best_epoch, "valid": evaluate(model, valid, tokens, trie, item_of)}
    if final_train is not train:
        model, _ = train_genrec(final_train, None, tokens, trie, item_of, epochs=best_epoch + 1)
    result["test"] = evaluate(model, test, tokens, trie, item_of)
    result["minutes"] = round((time.time() - start) / 60, 1)

    name = "genrec" + ("" if args.split == "loo" else "_time")
    Path(f"results/{name}.json").write_text(json.dumps(result, indent=2))
    print(name, json.dumps(result, indent=2))
