import math

import torch

from data import cold_start_split, leave_one_out, time_split
from evaluate import metrics
from genrec import build_trie, generate_topk, tiny_t5
from rqvae import add_collision_token, residual_quantize
from sasrec import SASRec, pad

SEQS = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7], [1, 2, 4, 3], [3, 1, 3]]


def test_leave_one_out_has_no_leakage():
    train, valid, test = leave_one_out(SEQS)
    for seq, t, (vh, vt), (th, tt) in zip(SEQS, train, valid, test):
        assert t == seq[:-2] and vh == t and vt == seq[-2]
        assert th == seq[:-1] and tt == seq[-1]


def test_time_split_never_trains_on_the_future():
    seqs = [[1, 2, 3, 4], [5, 6, 7], [8, 9]]
    times = [[1, 2, 5, 8], [3, 6, 9], [7, 10]]
    train, valid, test = time_split(seqs, times, t_valid=5, t_test=8)
    assert train == [[1, 2]]  # only items reviewed before day 5, users with at least 2 of them
    assert valid == [([1, 2], 3), ([5], 6)]  # first item on or after day 5, before day 8
    assert test == [([1, 2, 3], 4), ([5, 6], 7), ([8], 9)]  # first item on or after day 8


def test_cold_items_never_reach_training():
    warm, cold_test = cold_start_split(SEQS, cold_items={3})
    assert all(3 not in s for s in warm)
    assert warm == [[1, 2, 4, 5], [2, 4, 5, 6, 7], [1, 2, 4]]  # last user has under 3 warm items
    assert cold_test == [([1, 2, 4], 3), ([1], 3)]


def test_metrics_on_toy_case():
    topk = torch.tensor([[3, 1, 2], [5, 6, 7]])
    m = metrics(topk, torch.tensor([1, 9]), ks=(1, 3))
    assert m["recall@1"] == 0 and m["recall@3"] == 0.5
    assert math.isclose(m["ndcg@3"], 0.5 / math.log2(3), rel_tol=1e-6)


def test_residual_quantize_codes_what_earlier_levels_missed():
    codebooks = torch.tensor([[[0.0, 0.0], [10.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]])
    codes, residuals, chosen = residual_quantize(torch.tensor([[11.0, 1.0], [0.2, -0.1]]), codebooks)
    assert codes.tolist() == [[1, 1], [0, 0]]
    assert torch.equal(sum(chosen)[0], torch.tensor([11.0, 1.0]))  # the sum of chosen codes rebuilds the vector
    assert torch.equal(residuals[1][0], torch.tensor([1.0, 1.0]))  # level 2 sees what level 1 left over


def test_collision_token_makes_every_id_unique():
    assert add_collision_token([[1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3]]) == [
        (1, 2, 3, 0),
        (1, 2, 3, 1),
        (4, 5, 6, 0),
        (1, 2, 3, 2),
    ]


def test_constrained_beam_search_only_returns_real_items():
    torch.manual_seed(0)
    id_tokens = torch.tensor([[0, 0, 0, 0], [2, 258, 514, 770], [2, 259, 514, 770], [3, 300, 600, 770]])
    trie = build_trie(id_tokens)
    model = tiny_t5(vocab_size=1026).eval()  # untrained, so without the trie it would emit random tokens
    history = torch.tensor([[2, 258, 514, 770, 0, 0, 0, 0]])
    top = generate_topk(model, history, trie, k=3, beams=3)
    assert sorted(top[0]) == sorted(map(tuple, id_tokens[1:].tolist()))


def test_sasrec_cannot_see_future_items():
    torch.manual_seed(0)
    model = SASRec(n_items=10, max_len=5).eval()
    a, b = pad([[1, 2, 3, 4]], 5), pad([[1, 2, 9, 8]], 5)
    assert torch.allclose(model(a)[0, :2], model(b)[0, :2], atol=1e-6)
