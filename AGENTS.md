# Rules for anyone working in this repo

These rules apply to every agent and every person.
The build plan is in [PLAN.md](PLAN.md).
Current status is in [progress.md](progress.md).

## 1. One GitHub account only

Only the GitHub account `verma-divyanshu-git` may run git or gh commands here.
Commits must use the name `Divyanshu` and the email `bautocrats@gmail.com`.

[scripts/guard.sh](scripts/guard.sh) checks this.
It runs from four places:

- the git hooks in `.githooks/`, before every commit, branch or tag update, and push
- the Cursor agent hook in `.cursor/hooks.json`, before every agent shell command
- the Claude Code hook in `.claude/settings.json`, before every agent shell command
- the `git` and `gh` shell wrappers in `~/.zshenv` on the owner's Mac

After a fresh clone, run `scripts/guard.sh --setup` once.

If the guard blocks you:

- Stop and tell the user what the guard printed.
- Never work around it.
That means no `--no-verify`, no `command git`, no `git -c user.*`, no `GH_TOKEN` for another account, no editing or deleting the guard files, and no `gh auth switch` to another account.
- Never change the git identity or the `origin` remote.
- Never add a co-author line for an agent to commit messages.

## 2. Plain writing

The README, docs, comments and commit messages must be easy to read for an interviewer skimming the repo.

- Use plain, everyday words.
Write "use" instead of "leverage" and "help" instead of "facilitate".
- No buzzwords or hype.
Avoid words like "cutting-edge", "state-of-the-art", "robust", "seamless", "powerful", "revolutionary" and "game-changing".
- Explain each term the first time it appears.
For example: "a semantic ID is a short code of 3-4 numbers that describes an item, where similar items get similar codes".
- Use numbers instead of adjectives.
Write "Recall@10 went from 0.060 to 0.065", not "big improvement".
- Put each sentence on its own line in Markdown files.
- Never use the em dash.

## 3. Simple diagrams

- Use Mermaid diagrams so they render on GitHub.
- One idea per diagram, with at most about 10 boxes.
- Label boxes with plain words that someone new to recommender systems would understand.
- Every diagram gets one or two sentences under it saying what it shows.

## 4. Track progress in progress.md

After every work session, and every time a module changes status, update [progress.md](progress.md):

- the module's status: `not started`, `in progress`, `done` or `blocked`
- what was done, with key numbers such as metrics, run time and model size
- what comes next, and any blockers
- a dated line in the log

Read progress.md before starting work, so you continue from where the last session stopped.
