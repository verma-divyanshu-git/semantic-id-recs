# Progress

Plan: [PLAN.md](PLAN.md).
Rules: [AGENTS.md](AGENTS.md) and [CODING-RULES.md](CODING-RULES.md).

## Overall

Status: in progress.
Setup, data, baselines and semantic IDs are done.
The generative model is written and is training on Kaggle's free GPUs.
Heavy training now runs on Kaggle, because running it on the Mac made the Mac too hot and slow to use.

## Modules

| Module | File | Status | Notes |
| --- | --- | --- | --- |
| Repo guardrails and rules | `scripts/guard.sh`, `AGENTS.md` | done | One-account guard in git hooks, agent hooks and shell wrappers |
| Setup | `pyproject.toml` | done | Python 3.12, torch 2.14, sentence-transformers 6.1. `.venv` is 842 MB |
| Data | `data.py` | done | 22,363 users, 12,101 items, 198,502 reviews, same as the TIGER paper |
| Baselines | `sasrec.py` | in progress | Rerunning on shuffled same-day order, 2 of 6 runs done, the rest on Kaggle |
| Semantic IDs | `embed.py`, `rqvae.py` | done | 211, 254 and 248 of 256 codes used, 4.4% of items share 3 codes |
| Generative model | `genrec.py` | in progress | Training on Kaggle. Early check on the Mac: valid Recall@10 0.038 after 5 epochs |
| Kaggle runner | `scripts/kaggle.py` | done | Runs repo commands on Kaggle's free T4 x2, one command per GPU |
| Cold-start and image study | `embed.py`, `evaluate.py` | not started | Split is ready: 2,481 cold test cases |
| Core check | `test_core.py` | in progress | 8 tests pass: splits, cold items, metrics, SASRec mask, residual quantization, collision token, trie decoding |
| Demo and latency | `app.py`, `evaluate.py` | not started | |
| Write-up | `README.md` | not started | |

## Data

- Source: Amazon 2014 Beauty 5-core reviews and metadata from the SNAP mirror of the McAuley lab files.
- `uv run python data.py` downloads 144 MB and builds `data/beauty.json` in about 80 seconds.
It holds 4 lists: `items` (Amazon product codes), `seqs` (each user's item numbers, oldest first), `times` (review dates) and `meta` (title, brand, categories, price, image URL).
- Median history length is 6 items.
- 12,094 of 12,101 items have a title and an image URL.
- Dropped on purpose: ratings and review text (not used by TIGER), and the `related` and `salesRank` fields, because Amazon computed them from all purchases including future ones.

### Same-day reviews

Review times are whole days, and 43.9% of consecutive reviews fall on the same day.
The raw file is sorted by product code, so a plain sort put same-day reviews in product-code order every time.
That is a pattern that does not exist in real life, and models can learn it.
`data.py` now shuffles same-day reviews with seed 0 before sorting by date.
Results on the old order are kept in `results/paper_order/` for comparison with the paper.

## Leakage

The rules are in the "No leakage rules" section of [PLAN.md](PLAN.md).

- Leave-one-out split: each user's last item is the test target, the second to last is validation.
This matches the TIGER paper.
- Time split: every user is cut at 2014-03-07 (validation) and 2014-05-13 (test), so no model trains on a review written after a test review.
20,200 training users, 5,881 validation cases, 5,585 test cases.
The final time-split model retrains on everything before 2014-05-13 with the epoch count picked on validation.
- 19.1% of time-split test targets are items that never appear in training, because they launched after 2014-03-07.
On the leave-one-out split that share is 0.6%.

## Results

All metrics rank the full catalog of 12,101 items.
Recall@10 is the share of users whose true next item is in the top 10.
NDCG@10 also rewards putting it higher in the top 10.

Leave-one-out split, shuffled same-day order:

| Model | Recall@10 | NDCG@10 | Train time | Note |
| --- | --- | --- | --- | --- |
| Most popular items | 0.0163 | 0.0078 | 0 s | |
| SASRec, cross-entropy loss | 0.0709 | 0.0346 | Mac | best epoch 239 |
| SASRec, binary loss | pending | | Kaggle | |
| Generative model | pending | | Kaggle | |

Leave-one-out split, paper's same-day order (old runs):

| Model | Recall@10 | NDCG@10 |
| --- | --- | --- |
| Most popular items | 0.0114 | 0.0053 |
| SASRec, binary loss | 0.0414 | 0.0176 |
| SASRec, cross-entropy loss | 0.0810 | 0.0457 |
| TIGER paper's SASRec | 0.0605 | 0.0318 |
| TIGER paper's TIGER | 0.0648 | 0.0384 |

Shuffling same-day reviews dropped cross-entropy SASRec from 0.0810 to 0.0709 Recall@10, and from 0.0457 to 0.0346 NDCG@10.
So the paper's order inflated SASRec by 12% to 24%.

Time split: all runs are pending on Kaggle.
The old runs, before the final retrain step, got Recall@10 0.0098 for popularity and 0.0131 for cross-entropy SASRec.

SASRec settings follow the original paper: 2 layers, 1 head, hidden size 64, dropout 0.5, learning rate 0.001, batch 128, history of 50 items.
Training stops after 20 epochs with no gain in validation NDCG@10.

## Semantic IDs

- `embed.py`: the frozen `sentence-t5-base` model turns "Title. Brand. Categories. Price." into 768 numbers per item, in 46 seconds on the Mac.
- `rqvae.py`: an encoder shrinks 768 numbers to 32, then 3 levels each pick one of 256 codes.
Codebooks start at k-means centres.
A 4th number separates items that share all 3 codes.
It trains in 3 to 4 minutes on the Mac and only sees items that appear in training.

| Split | Items fit on | Codes used per level | Items sharing 3 codes | Largest group |
| --- | --- | --- | --- | --- |
| Leave-one-out | 12,087 | 211, 254, 248 | 4.4% | 5 |
| Time | 11,773 | 142, 256, 247 | 6.0% | 5 |

Check: a wig's first code covers hair accessories, and its first two codes hold only wigs.
For 67% of items, the nearest item by text shares its first code.

## Generative model

`genrec.py` follows TIGER's size: a T5 with 4 encoder and 4 decoder layers, 6 heads, d_model 128.
The input is the last 20 items as 4 tokens each, and the output is the next item's 4 tokens.
Beam search with 20 beams is restricted by a trie of real item IDs, so every output is a real item.
User tokens from the paper are left out.

## Running on Kaggle

- Needs the `kaggle` tool (`uv tool install kaggle`) and a Kaggle API token in `~/.kaggle/access_token`.
- This Mac's network inspects HTTPS with a Cisco root certificate, so Python needs `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` pointing at a bundle that includes it (`~/.config/ssl/ca-bundle.pem`, outside the repo).
- `uv run python scripts/kaggle.py data` uploads `data/` as the private dataset `semantic-id-recs-data`.
- `uv run python scripts/kaggle.py push NAME "cmd" ...` runs commands at the current pushed commit.
- Kaggle shows logs only after a job ends, and ends jobs at 12 hours.
So `genrec.py --hours 3` stops picking the epoch count after 3 hours and keeps the best model.

## Next

- Pull Kaggle job `run1` when it finishes (at most about 7 hours): generative model on both splits and the 4 missing baselines.
- Decide which SASRec is the headline baseline.
- Week 5: cold-start study and image embeddings.
The image models need more disk, and 17 GB is free.

## Log

- 2026-09-24: Added the one-account guard, agent rules, this progress file and the plan.
- 2026-09-25: Set up uv with Python 3.12. Wrote `data.py`, `evaluate.py`, `sasrec.py` and `test_core.py` test-first. Data counts match the TIGER paper.
- 2026-09-25: Trained SASRec with early stopping only. Test Recall@10 0.0414 with binary loss and 0.0810 with cross-entropy.
- 2026-09-25: Added the no-leakage rules to PLAN.md and a global time split. 19.1% of time-split test targets are items missing from training.
- 2026-09-25: Found same-day reviews sorted by product code. Shuffled them, which drops cross-entropy SASRec to Recall@10 0.0709. Added the final retrain step for the time split.
- 2026-09-25: Week 2 done. Text embeddings with sentence-t5-base, RQ-VAE semantic IDs with 4.4% shared 3-code IDs.
- 2026-09-25: Wrote `genrec.py` with trie-constrained beam search. Mac training was too slow and too hot, about 6 minutes per epoch while sharing the GPU. Moved heavy training to Kaggle with `scripts/kaggle.py` and started job `run1`.
