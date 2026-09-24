#!/usr/bin/env bash
# Exits non-zero unless git and gh in this repo act as the one allowed GitHub account.
# Called by the git hooks, the agent hooks and the git/gh shell wrappers.
# Run `scripts/guard.sh --setup` once after cloning.

GH_USER=verma-divyanshu-git
NAME=Divyanshu
EMAIL=bautocrats@gmail.com

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
g() { command git -C "$ROOT" "$@"; }

if [[ "${1:-}" == --setup ]]; then
  g config user.name "$NAME"
  g config user.email "$EMAIL"
  g config user.useConfigOnly true
  g config core.hooksPath .githooks
  g config credential.https://github.com.username "$GH_USER"
fi

errors=()
check() { [[ "$2" == "$3" ]] || errors+=("$1 is '$2', expected '$3'. Fix: $4"); }

# `git var` includes GIT_AUTHOR_* / GIT_COMMITTER_* env overrides, not just config.
ident() { g var "$1" 2>/dev/null | sed -E 's/ [0-9]+ [-+][0-9]{4}$//'; }
id_fix="scripts/guard.sh --setup, and unset any GIT_AUTHOR_* or GIT_COMMITTER_* env vars"
check "git author" "$(ident GIT_AUTHOR_IDENT)" "$NAME <$EMAIL>" "$id_fix"
check "git committer" "$(ident GIT_COMMITTER_IDENT)" "$NAME <$EMAIL>" "$id_fix"
check "git core.hooksPath" "$(g config core.hooksPath)" ".githooks" "scripts/guard.sh --setup"
check "git credential username" "$(g config credential.https://github.com.username)" "$GH_USER" "scripts/guard.sh --setup"

url="$(g remote get-url origin 2>/dev/null)"
[[ -z "$url" || "$url" =~ github\.com[:/]$GH_USER/ ]] || errors+=("origin is '$url', expected a repo under github.com/$GH_USER")

# GH_TOKEN and GITHUB_TOKEN override the logged-in account, so ask GitHub who the token belongs to.
if [[ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]]; then
  active="$(command gh api user --jq .login 2>/dev/null)" || active="unknown (token rejected)"
else
  active="$(command gh config get -h github.com user 2>/dev/null)"
fi
check "active gh account" "$active" "$GH_USER" "gh auth switch --hostname github.com --user $GH_USER"

[[ ${#errors[@]} -eq 0 ]] && exit 0
{
  echo "Blocked: only the GitHub account $GH_USER may run git or gh in this repo."
  printf '  - %s\n' "${errors[@]}"
  echo "Agents: stop and ask the user. Do not work around this check."
} >&2
exit 1
