#!/bin/bash
# supervise_claude.sh NAME MODEL ARM DIR : run one Claude arm until all six datasets are complete.
NAME=$1; MODEL=$2; ARM=$3; DIR=$4; S=${CLAUDE_RUN_STATE:-/tmp/synthagent_claude}; mkdir -p $S
exec 9>$S/lock_$NAME; flock -n 9 || exit 0
[ -f $S/pause_claude ] && exit 0
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/src/Synthagent"
done_count() { python3 -I -c "
import json,sys
try: c=json.load(open('$DIR/checkpoint.json'))
except Exception: print(0); sys.exit()
print(sum(1 for k in c.get('completed',{}) if k.split('::')[0]=='$ARM'))"; }
while [ "$(done_count)" -lt 6 ] && [ ! -f $S/pause_claude ]; do
  echo "$(date -u) starting $NAME (completed: $(done_count))" >> $S/supervise_$NAME.log
  TABULLM_PROVIDER=anthropic TABULLM_MODEL=$MODEL TABULLM_ABLATION_ARMS=$ARM LOCAL_LLM_TIMEOUT=180 PYTHONUNBUFFERED=1 \
    python3 run_batch_pipeline.py --ablation --resume "$DIR" >> $S/run_$NAME.log 2>&1 < /dev/null
  sleep 30
done
[ "$(done_count)" -ge 6 ] && { echo "$(date -u) $NAME complete" >> $S/supervise_$NAME.log; touch $S/done_$NAME; }
