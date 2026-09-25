# Progress

Plan: [PLAN.md](PLAN.md).
Rules: [AGENTS.md](AGENTS.md) and [CODING-RULES.md](CODING-RULES.md).

## Overall

Status: in progress.
Setup and data are done.
Baselines are done.
The next step is Week 2: item embeddings and semantic IDs.

## Modules

| Module | File | Status | Notes |
| --- | --- | --- | --- |
| Repo guardrails and rules | `scripts/guard.sh`, `AGENTS.md` | done | One-account guard in git hooks, agent hooks and shell wrappers |
| Setup | `pyproject.toml` | done | Python 3.12, torch 2.14 on MPS, numpy, pytest. `.venv` is 646 MB |
| Data | `data.py` | done | 22,363 users, 12,101 items, 198,502 reviews, same as the TIGER paper |
| Baselines | `sasrec.py` | done | Recall@10: popularity 0.0114, SASRec binary loss 0.0414, SASRec cross-entropy 0.0810 |
| Semantic IDs | `embed.py`, `rqvae.py` | not started | |
| Generative model | `genrec.py` | not started | Target: Recall@10 about 0.065 on Beauty |
| Cold-start and image study | `embed.py`, `evaluate.py` | not started | Split is ready: 2,488 cold test cases |
| Core check | `test_core.py` | in progress | 4 tests pass: split leakage, cold items kept out of training, metrics, SASRec causal mask |
| Demo and latency | `app.py`, `evaluate.py` | not started | |
| Write-up | `README.md` | not started | |

## Data

- Source: Amazon 2014 Beauty 5-core reviews and metadata from the SNAP mirror of the McAuley lab files.
- `uv run python data.py` downloads 144 MB and builds `data/beauty.json` in about 80 seconds.
- Each user's items are sorted by review time.
The last item is the test target, the second to last is the validation target, and the rest is training.
This is called a leave-one-out split.
- Median history length is 6 items.
- 12,094 of 12,101 items have a title and an image URL.
- Cold-start split: 10% of items (1,210, seed 0) are removed from every sequence.
The test cases are the 2,488 users whose last item is one of those cold items.

## Baseline results on the test set

All metrics rank the full catalog of 12,101 items.
Recall@10 is the share of users whose true next item is in the top 10.
NDCG@10 also rewards putting it higher in the top 10.

| Model | Recall@10 | NDCG@10 | Train time | Note |
| --- | --- | --- | --- | --- |
| Most popular items | 0.0114 | 0.0053 | 0 s | |
| SASRec, binary loss | 0.0414 | 0.0176 | 12.5 min | early stop, best epoch 309 |
| SASRec, cross-entropy loss | 0.0810 | 0.0457 | 23.5 min | early stop, best epoch 379 |
| TIGER paper's SASRec | 0.0605 | 0.0318 | | |
| TIGER paper's TIGER | 0.0648 | 0.0384 | | |

SASRec uses the original paper's settings: 2 layers, 1 head, hidden size 64, dropout 0.5, learning rate 0.001, batch 128, history of 50 items.
The binary loss is the original paper's loss, with one random negative item per step.
The cross-entropy loss scores every item at every step.

Training stops after 20 epochs with no gain in validation NDCG@10.
Binary-loss SASRec lands below the paper's 0.0605, so this setup does not reproduce the paper's SASRec number.
Cross-entropy SASRec already beats the TIGER paper's reported TIGER numbers.
That matters for the resume claim "improved Recall@10 by X% over SASRec", which may only hold against the binary-loss SASRec.
The write-up should report both.

## Next

- Decide which SASRec is the headline baseline.
- Week 2: `embed.py` and `rqvae.py`.
This needs `sentence-transformers`, about 1 GB more disk.
Only 22 GB of disk is free, so free space before adding the image models.

## Log

- 2026-09-24: Added the one-account guard, agent rules, this progress file and the plan.
- 2026-09-25: Set up uv with Python 3.12. Wrote `data.py`, `evaluate.py`, `sasrec.py` and `test_core.py` test-first. Data counts match the TIGER paper. Popularity Recall@10 0.0114. SASRec Recall@10 0.0284 with binary loss and 0.0753 with cross-entropy, both capped at 200 epochs. Retraining without the cap.
- 2026-09-25: Retrained SASRec with early stopping only. Test Recall@10 0.0414 with binary loss (12.5 min) and 0.0810 with cross-entropy (23.5 min). Baselines done.
