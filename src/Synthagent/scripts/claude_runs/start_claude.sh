#!/bin/bash
# Idempotent: (re)starts any missing Claude supervisor and the pusher. Needs TABULLM_ANTHROPIC_API_KEY. State and logs go to $CLAUDE_RUN_STATE (default /tmp/synthagent_claude).
S=${CLAUDE_RUN_STATE:-/tmp/synthagent_claude}; mkdir -p $S; D=$(cd "$(dirname "$0")" && pwd); R=results/claude
[ -n "$TABULLM_ANTHROPIC_API_KEY$ANTHROPIC_API_KEY" ] || { echo "NO TABULLM_ANTHROPIC_API_KEY in this container"; exit 1; }
[ -f $S/pause_claude ] && { echo "paused (remove $S/pause_claude to resume)"; exit 0; }
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/src/Synthagent"; mkdir -p $R
for spec in "haiku55_full claude-haiku-5-5 full $R/haiku55_full" \
            "haiku55_raw claude-haiku-5-5 raw_prompting $R/haiku55_raw" \
            "haiku55_structured claude-haiku-5-5 structured_output $R/haiku55_structured" \
            "haiku45_full claude-haiku-4-5 full $R/haiku45_full" \
            "sonnet46_full claude-sonnet-4-6 full $R/sonnet46_full" \
            "sonnet55_full claude-sonnet-5-5 full $R/sonnet55_full"; do
  set -- $spec
  [ -f $S/done_$1 ] && continue
  mkdir -p $4
  (setsid nohup $D/supervise_claude.sh $1 $2 $3 $4 >/dev/null 2>&1 < /dev/null &)
done
pgrep -f pusher_claude.sh >/dev/null || (setsid nohup $D/pusher_claude.sh >> $S/pusher_claude.log 2>&1 < /dev/null &)
sleep 3; pgrep -fa "supervise_claude.sh" | grep -v pgrep | awk '{print $(NF-3)}'
