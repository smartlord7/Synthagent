#!/bin/bash
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
while pgrep -f supervise_claude.sh >/dev/null; do
  sleep 1800
  git add src/TabuLLM/results/claude >/dev/null 2>&1
  if ! git diff --cached --quiet; then
    git commit -q -m "Checkpoint partial results of the Claude runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DraChAkQmVFsCcL5wiGUNW" >/dev/null 2>&1
    for i in 1 2 3 4; do git push -q origin claude/friendly-johnson-11679g && break; sleep $((2**i)); done
  fi
done
