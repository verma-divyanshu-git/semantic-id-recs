# Progress

Plan: [PLAN.md](PLAN.md).
Rules: [AGENTS.md](AGENTS.md) and [CODING-RULES.md](CODING-RULES.md).

## Overall

Status: in progress.
Setup, data, baselines and semantic IDs are done.
The generative model has its first leave-one-out result, Recall@10 0.0533, below cross-entropy SASRec at 0.0709.
The cold-start and image study code is done, and its training runs wait for a free Kaggle slot.
Heavy training now runs on Kaggle, because running it on the Mac made the Mac too hot and slow to use.

## Modules

| Module | File | Status | Notes |
| --- | --- | --- | --- |
| Repo guardrails and rules | `scripts/guard.sh`, `AGENTS.md` | done | One-account guard in git hooks, agent hooks and shell wrappers |
| Setup | `pyproject.toml` | done | Python 3.12, torch 2.14, sentence-transformers 6.1. `.venv` is 842 MB |
| Data | `data.py` | done | 22,363 users, 12,101 items, 198,502 reviews, same as the TIGER paper |
| Baselines | `sasrec.py` | in progress | Popularity and text similarity done on all 3 splits. SASRec runs finishing on Kaggle |
| Semantic IDs | `embed.py`, `rqvae.py` | done | 211, 254 and 248 of 256 codes used, 4.4% of items share 3 codes |
| Generative model | `genrec.py` | in progress | Leave-one-out test Recall@10 0.0533. Time split finishing on Kaggle job `run1` |
| Kaggle runner | `scripts/kaggle.py` | done | Runs repo commands on Kaggle's free T4 x2, one command per GPU, and brings back checkpoints |
| Cold-start and image study | `data.py`, `embed.py`, `rqvae.py`, `genrec.py` | in progress | Code, image embeddings and cold-split semantic IDs done. Training queued as Kaggle job `run2` |
| Core check | `test_core.py` | in progress | 9 tests pass: splits, cold items, metrics, SASRec mask, residual quantization, collision token, trie decoding, cold slots |
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
- 19.1% of time-split test targets are items that never appear in training before 2014-03-07, because they launched later.
Against the final training data, everything before 2014-05-13, the share is 9.4% (525 cases).
On the leave-one-out split that share is 0.6%.

## Results

All metrics rank the full catalog of 12,101 items.
Recall@10 is the share of users whose true next item is in the top 10.
NDCG@10 also rewards putting it higher in the top 10.

Leave-one-out split, shuffled same-day order:

| Model | Recall@10 | NDCG@10 | Train time | Note |
| --- | --- | --- | --- | --- |
| Most popular items | 0.0163 | 0.0078 | 0 s | |
| Text similarity to the last item | 0.0496 | 0.0307 | 0 s | no training |
| SASRec, cross-entropy loss | 0.0709 | 0.0346 | Mac | best epoch 239 |
| SASRec, binary loss | pending | | Kaggle | |
| Generative model, text semantic IDs | 0.0533 | 0.0272 | 116 min on a T4 | best epoch 54 |

The generative model is 25% below cross-entropy SASRec on Recall@10, and only 7% above plain text similarity.
Its validation Recall@10 was 0.0624 and test 0.0533.
The TIGER paper reports 0.0648, on the paper's same-day order and with user tokens, which we leave out.

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

## Cold-start and image study

- Cold split: 10% of items (1,210, seed 0) never appear in training or in any history.
Validation and test targets are still each user's real last items, so 2,544 of 22,362 test cases ask for an item the model never saw.
- Every run now also reports an "unseen" slice: test cases whose target never appears in the final training data.
That is 67 cases on leave-one-out, 2,544 on the cold split and 525 on the time split.
- Cold slots: the generative model only learned to write IDs of training items, so new items rank low.
Following the TIGER paper, a share of the top 10 (0, 10%, 20%, 30% or 50%) is kept for the best unseen items that beam search found.
The share is picked on validation Recall@10 over all users.
- Only 39% of cold items share their first 2 codes with any training item, and 4.6% share all 3.
So the model must often write a code path it never saw as a target, which may keep cold recall low.
- Images: all 12,094 image URLs from 2014 still work (116 MB).
`embed.py --images` embeds them with the frozen CLIP ViT-B/32 model in 4.7 minutes on the Mac.
For 39% of items, the nearest item by image is also one of the 10 nearest by text.
- `rqvae.py --images` puts the text and image embeddings side by side (768 + 512 numbers).
With images, all 256 codes are used at every level, against 201, 242 and 231 for text only.

| Model | Cold split, all test cases, Recall@10 | Unseen items only, Recall@10 | NDCG@10 |
| --- | --- | --- | --- |
| Most popular items | 0.0163 | 0.0000 | 0.0000 |
| Text similarity to the last item | 0.0479 | 0.0483 | 0.0287 |
| SASRec | pending | | |
| Generative model, text IDs | pending | | |
| Generative model, text + image IDs | pending | | |

Text similarity needs no training, so it treats new items like any other.
It is the number the generative model has to beat on unseen items.
On the time split's 525 unseen cases it gets only 0.0057.

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
- The free tier allows 2 GPU sessions at a time, and one T4 x2 job uses both, so jobs run one after another.
- `genrec.py` saves the final model to `checkpoints/`, 18 MB each, and `pull` copies them here.

## Next

- Job `run1` ends around 23:30 on 2026-09-25 with the time-split results and the 4 missing baselines.
A background loop on the Mac pulls it and then starts `run2`.
- Job `run2`, about 5 hours: generative model on the cold split with text IDs and with text + image IDs, leave-one-out again to save a checkpoint, random IDs as an ablation, and SASRec on the cold split.
- Decide which SASRec is the headline baseline.
- Still to do for Week 5: codebook depth 2 and 4, and beam size against recall and speed using the saved checkpoint.
- The time-split generative model has no unseen-slice numbers yet, because `run1` started before that code.

## Log

- 2026-09-24: Added the one-account guard, agent rules, this progress file and the plan.
- 2026-09-25: Set up uv with Python 3.12. Wrote `data.py`, `evaluate.py`, `sasrec.py` and `test_core.py` test-first. Data counts match the TIGER paper.
- 2026-09-25: Trained SASRec with early stopping only. Test Recall@10 0.0414 with binary loss and 0.0810 with cross-entropy.
- 2026-09-25: Added the no-leakage rules to PLAN.md and a global time split. 19.1% of time-split test targets are items missing from training.
- 2026-09-25: Found same-day reviews sorted by product code. Shuffled them, which drops cross-entropy SASRec to Recall@10 0.0709. Added the final retrain step for the time split.
- 2026-09-25: Week 2 done. Text embeddings with sentence-t5-base, RQ-VAE semantic IDs with 4.4% shared 3-code IDs.
- 2026-09-25: Wrote `genrec.py` with trie-constrained beam search. Mac training was too slow and too hot, about 6 minutes per epoch while sharing the GPU. Moved heavy training to Kaggle with `scripts/kaggle.py` and started job `run1`.
- 2026-09-25: Generative model on leave-one-out: test Recall@10 0.0533, below SASRec's 0.0709. Added the cold split, the unseen slice, cold slots, a text similarity baseline (Recall@10 0.0496) and CLIP image embeddings for all 12,094 images. Queued job `run2`.
