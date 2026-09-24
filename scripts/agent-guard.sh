#!/usr/bin/env bash
# Cursor (beforeShellExecution) and Claude Code (PreToolUse) run this before an agent's shell command.
# Blocks any git or gh command when scripts/guard.sh fails. `gh auth` stays open so the account can be switched back.

cmd="$(jq -r '.command // .tool_input.command // empty')"

if [[ "$cmd" =~ (^|[^[:alnum:]_.-])(git|gh)([[:space:]]|$) && ! "$cmd" =~ gh[[:space:]]+auth ]]; then
  if ! msg="$("$(dirname "$0")/guard.sh" 2>&1)"; then
    jq -n --arg m "$msg" '{permission: "deny", user_message: $m, agent_message: $m}'
    echo "$msg" >&2
    exit 2
  fi
fi
echo '{"permission": "allow"}'
