"""Download Amazon 2014 Beauty 5-core, build one time-ordered item list per user, and split it.

Item ids run from 1 to n_items. Id 0 is reserved for padding.
"""

import ast
import gzip
import json
import random
import urllib.request
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date
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
    # Timestamps are whole days and the file is sorted by ASIN, so a stable sort would put same-day
    # reviews in ASIN order, a pattern models can learn. Shuffle first so same-day order is random.
    rng = random.Random(0)
    ordered = []
    for u in sorted(by_user):
        rng.shuffle(by_user[u])
        ordered.append(sorted(by_user[u], key=lambda r: r[0]))
    seqs = [[item_id[a] for _, a in rs] for rs in ordered]
    times = [[t for t, _ in rs] for rs in ordered]
    same_day = [(p, n) for s, ts in zip(seqs, times) for p, n, tp, tn in zip(s, s[1:], ts, ts[1:]) if tp == tn]
    rising = sum(n > p for p, n in same_day) / len(same_day)
    assert 0.45 < rising < 0.55, f"same-day reviews still follow item order ({rising:.2f} rising)"
    stats = {"users": len(seqs), "items": len(items), "reviews": len(reviews)}
    assert stats == EXPECTED, f"got {stats}, expected {EXPECTED}"

    meta = {}
    with gzip.open(download("meta_Beauty.json.gz"), "rt") as f:
        for line in f:
            m = ast.literal_eval(line)  # the 2014 metadata is Python dict syntax, not JSON
            if m["asin"] in item_id:
                meta[m["asin"]] = {k: m.get(k) for k in META_FIELDS}

    out = {"items": items, "seqs": seqs, "times": times, "meta": [meta.get(a, {}) for a in items]}
    OUT.write_text(json.dumps(out))


def load():
    """Return (seqs, times, items, meta). seqs[u] is user u's item ids, oldest first, reviewed at times[u]."""
    if not OUT.exists():
        build()
    d = json.loads(OUT.read_text())
    return d["seqs"], d["times"], d["items"], d["meta"]


def leave_one_out(seqs):
    """Last item is the test target, second to last is the validation target, the rest is training."""
    train = [s[:-2] for s in seqs]
    valid = [(s[:-2], s[-2]) for s in seqs]
    test = [(s[:-1], s[-1]) for s in seqs]
    return train, valid, test


def time_cutoffs(times, valid_frac=0.1, test_frac=0.1):
    """Return (t_valid, t_test): the last test_frac of reviews fall on or after t_test, the valid_frac before them on or after t_valid."""
    all_times = sorted(t for ts in times for t in ts)
    t_valid = all_times[int(len(all_times) * (1 - valid_frac - test_frac))]
    t_test = all_times[int(len(all_times) * (1 - test_frac))]
    return t_valid, t_test


def time_split(seqs, times, t_valid, t_test):
    """Split every user at the same two dates, so no model ever trains on a review from after a cutoff.

    Training has only reviews before t_valid. A validation case predicts a user's first item in
    [t_valid, t_test) from their earlier items. A test case predicts their first item on or after t_test.
    """
    train, valid, test = [], [], []
    for s, ts in zip(seqs, times):
        n_valid, n_test = bisect_left(ts, t_valid), bisect_left(ts, t_test)
        if n_valid >= 2:
            train.append(s[:n_valid])
        if 0 < n_valid < n_test:
            valid.append((s[:n_valid], s[n_valid]))
        if 0 < n_test < len(s):
            test.append((s[:n_test], s[n_test]))
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
    seqs, times, items, meta = load()
    lengths = sorted(map(len, seqs))
    print(f"{len(seqs)} users, {len(items)} items, {sum(lengths)} reviews, median length {lengths[len(lengths) // 2]}")
    print(f"items with a title: {sum(bool(m.get('title')) for m in meta)}, with an image URL: {sum(bool(m.get('imUrl')) for m in meta)}")
    warm, cold_test = cold_start_split(seqs, pick_cold_items(len(items)))
    print(f"cold-start split: {len(warm)} warm users, {len(cold_test)} cold test cases")
    t_valid, t_test = time_cutoffs(times)
    train, valid, test = time_split(seqs, times, t_valid, t_test)
    day = lambda t: date.fromtimestamp(t).isoformat()
    print(f"time split at {day(t_valid)} and {day(t_test)}: {len(train)} training users, {len(valid)} valid cases, {len(test)} test cases")
