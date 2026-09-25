"""Download Amazon 2014 Beauty 5-core, build one time-ordered item list per user, and split it.

Item ids run from 1 to n_items. Id 0 is reserved for padding.
"""

import ast
import gzip
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
DATA = Path("data")
OUT = DATA / "beauty.json"
EXPECTED = {"users": 22363, "items": 12101, "reviews": 198502}  # counts reported in the TIGER paper
META_FIELDS = ("title", "brand", "categories", "price", "imUrl")


def download(name):
    path = DATA / "raw" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(BASE_URL + name, part)
        part.rename(path)
    return path


def build():
    with gzip.open(download("reviews_Beauty_5.json.gz"), "rt") as f:
        reviews = [json.loads(line) for line in f]
    by_user = defaultdict(list)
    for r in reviews:
        by_user[r["reviewerID"]].append((r["unixReviewTime"], r["asin"]))
    item_counts = Counter(r["asin"] for r in reviews)
    assert min(item_counts.values()) >= 5 and min(map(len, by_user.values())) >= 5, "not a 5-core file"

    items = sorted(item_counts)
    item_id = {asin: i + 1 for i, asin in enumerate(items)}
    # sorted() is stable, so reviews with the same timestamp keep their file order.
    seqs = [[item_id[a] for _, a in sorted(by_user[u], key=lambda r: r[0])] for u in sorted(by_user)]
    stats = {"users": len(seqs), "items": len(items), "reviews": len(reviews)}
    assert stats == EXPECTED, f"got {stats}, expected {EXPECTED}"

    meta = {}
    with gzip.open(download("meta_Beauty.json.gz"), "rt") as f:
        for line in f:
            m = ast.literal_eval(line)  # the 2014 metadata is Python dict syntax, not JSON
            if m["asin"] in item_id:
                meta[m["asin"]] = {k: m.get(k) for k in META_FIELDS}

    OUT.write_text(json.dumps({"items": items, "seqs": seqs, "meta": [meta.get(a, {}) for a in items]}))


def load():
    """Return (seqs, items, meta). seqs[u] is user u's item ids, oldest first."""
    if not OUT.exists():
        build()
    d = json.loads(OUT.read_text())
    return d["seqs"], d["items"], d["meta"]


def leave_one_out(seqs):
    """Last item is the test target, second to last is the validation target, the rest is training."""
    train = [s[:-2] for s in seqs]
    valid = [(s[:-2], s[-2]) for s in seqs]
    test = [(s[:-1], s[-1]) for s in seqs]
    return train, valid, test


def pick_cold_items(n_items, frac=0.1, seed=0):
    return set(random.Random(seed).sample(range(1, n_items + 1), round(frac * n_items)))


def cold_start_split(seqs, cold_items):
    """Remove cold items from every sequence.

    Returns the warm sequences (at least 3 items, so leave_one_out still works) and
    (history, target) pairs for users whose last item is cold.
    """
    warm = [w for s in seqs if len(w := [i for i in s if i not in cold_items]) >= 3]
    cold_test = [
        (h, s[-1]) for s in seqs if s[-1] in cold_items and (h := [i for i in s[:-1] if i not in cold_items])
    ]
    return warm, cold_test


if __name__ == "__main__":
    seqs, items, meta = load()
    lengths = sorted(map(len, seqs))
    print(f"{len(seqs)} users, {len(items)} items, {sum(lengths)} reviews, median length {lengths[len(lengths) // 2]}")
    print(f"items with a title: {sum(bool(m.get('title')) for m in meta)}, with an image URL: {sum(bool(m.get('imUrl')) for m in meta)}")
    warm, cold_test = cold_start_split(seqs, pick_cold_items(len(items)))
    print(f"cold-start split: {len(warm)} warm users, {len(cold_test)} cold test cases")
