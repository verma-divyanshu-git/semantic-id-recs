"""Generative recommender: a small T5 reads a user's history as semantic ID tokens and writes the next item's ID.

Each item is 4 tokens, one per semantic ID level. Beam search may only follow token paths that spell
a real item's ID, using a trie (a prefix tree of all valid IDs), so every output is a real item.

Run: uv run python genrec.py --split loo   # or --split cold / time, --ids image / random
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

from data import DATA, get_split, load, unseen_pairs
from evaluate import metrics
from rqvae import add_collision_token
from sasrec import DEVICE, pad

PAD, EOS, FIRST_CODE = 0, 1, 2  # token ids
CODEBOOK = 256
HISTORY = 20  # TIGER keeps the last 20 items
K = 10
EPS = (0, 0.1, 0.2, 0.3, 0.5)  # share of the top 10 kept for unseen items, picked on validation


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


def reserve_slots(ranked, unseen, k, eps):
    """Top k of ranked, but at least round(eps * k) slots go to the best-ranked items never seen in training.

    The model only learned to write IDs of training items, so it ranks new items low even when their IDs
    share a prefix with what it wrote. This is the cold-start knob from the TIGER paper.
    """
    new = [i for i in ranked if i in unseen][: round(eps * k)]
    rest = [i for i in ranked if i not in new][: k - len(new)]
    return sorted(new + rest, key=ranked.index)


def encode(histories, tokens):
    return tokens[pad(histories, HISTORY)].flatten(1)


@torch.no_grad()
def rank(model, pairs, tokens, trie, item_of, beams=20, batch=256):
    """Return each pair's beam search items, best first."""
    model.eval()
    ranked = []
    for i in range(0, len(pairs), batch):
        x = encode([h for h, _ in pairs[i : i + batch]], tokens).to(DEVICE)
        ranked += [[item_of[t] for t in row] for row in generate_topk(model, x, trie, k=beams, beams=beams)]
    return ranked


def score(ranked, pairs, unseen=frozenset(), eps=0):
    tops = [reserve_slots(r, unseen, K, eps) for r in ranked]
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
        recall = score(rank(model, valid_sample, tokens, trie, item_of), valid_sample)["recall@10"]
        print(f"epoch {epoch:3d}  loss {total:8.2f}  {minutes:6.1f} min  valid recall@10 {recall:.4f}", flush=True)
        if recall > best:
            best, best_state, best_epoch = recall, copy.deepcopy(model.state_dict()), epoch
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
    p.add_argument("--split", choices=["loo", "cold", "time"], default="loo")
    p.add_argument("--ids", choices=["text", "image", "random"], default="text", help="semantic IDs from text, from text + image, or random codes")
    p.add_argument("--hours", type=float, help="stop picking the epoch count after this many hours")
    args = p.parse_args()

    seqs, times, items, _ = load()
    train, valid, test, final_train = get_split(args.split, seqs, times)
    if args.ids == "random":  # ablation: same ID shape, but codes carry no meaning
        rng = random.Random(0)
        ids = add_collision_token([[rng.randrange(CODEBOOK) for _ in range(3)] for _ in items])
    else:
        ids = json.loads((DATA / f"semantic_ids_{args.split}{'_image' if args.ids == 'image' else ''}.json").read_text())
    tokens = id_tokens(ids)
    trie = build_trie(tokens)
    item_of = {tuple(row): i + 1 for i, row in enumerate(tokens[1:].tolist())}
    unseen_items = lambda train_seqs: set(range(1, len(items) + 1)) - {i for s in train_seqs for i in s}

    start = time.time()
    model, best_epoch = train_genrec(train, valid, tokens, trie, item_of, hours=args.hours)
    valid_ranked = rank(model, valid, tokens, trie, item_of)
    by_eps = {eps: score(valid_ranked, valid, unseen_items(train), eps)["recall@10"] for eps in EPS}
    eps = max(by_eps, key=by_eps.get)
    result = {"best_epoch": best_epoch, "eps": eps, "valid_recall@10_by_eps": by_eps, "valid": score(valid_ranked, valid, unseen_items(train), eps)}
    if final_train is not train:
        model, _ = train_genrec(final_train, None, tokens, trie, item_of, epochs=best_epoch + 1)
    unseen, test_unseen = unseen_items(final_train), unseen_pairs(test, final_train)
    result["test"] = score(rank(model, test, tokens, trie, item_of), test, unseen, eps)
    result["test_unseen"] = {"cases": len(test_unseen), **score(rank(model, test_unseen, tokens, trie, item_of), test_unseen, unseen, eps)}
    result["minutes"] = round((time.time() - start) / 60, 1)

    name = "genrec" + ("" if args.split == "loo" else f"_{args.split}") + ("" if args.ids == "text" else f"_{args.ids}")
    Path(f"results/{name}.json").write_text(json.dumps(result, indent=2))
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{name}.pt")
    print(name, json.dumps(result, indent=2))
