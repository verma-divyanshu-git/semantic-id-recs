# Progress

Plan: [PLAN.md](PLAN.md).
Rules: [AGENTS.md](AGENTS.md).

## Overall

Status: in progress.
Repo guardrails are done.
The next step is setup: free disk space, then create the Python environment.

## Modules

| Module | File | Status | Notes |
| --- | --- | --- | --- |
| Repo guardrails and rules | `scripts/guard.sh`, `AGENTS.md` | done | One-account guard in git hooks, agent hooks and shell wrappers |
| Setup | `pyproject.toml` | not started | Needs at least 20 GB free disk space first |
| Data | `data.py` | not started | |
| Baselines | `sasrec.py` | not started | Target: SASRec Recall@10 about 0.060 on Beauty |
| Semantic IDs | `embed.py`, `rqvae.py` | not started | |
| Generative model | `genrec.py` | not started | Target: Recall@10 about 0.065 on Beauty |
| Cold-start and image study | `embed.py`, `evaluate.py` | not started | |
| Core check | `test_core.py` | not started | |
| Demo and latency | `app.py`, `evaluate.py` | not started | |
| Write-up | `README.md` | not started | |

## Log

- 2026-09-24: Added the one-account guard, agent rules, this progress file and the plan.
