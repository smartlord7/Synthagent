# Claude runs for the Synthagent paper

Six runs, each in its own results folder under `src/Synthagent/results/claude/`:

| Name | Model | Arm |
|---|---|---|
| haiku55_full | claude-haiku-5-5 | full (reference Synthagent) |
| haiku55_raw | claude-haiku-5-5 | raw_prompting (direct prompting) |
| haiku55_structured | claude-haiku-5-5 | structured_output |
| haiku45_full | claude-haiku-4-5 | full |
| sonnet46_full | claude-sonnet-4-6 | full |
| sonnet55_full | claude-sonnet-5-5 | full |

- `start_claude.sh` starts one supervisor per unfinished run and a pusher (idempotent; safe to re-run after a restart). Needs `TABULLM_ANTHROPIC_API_KEY`.
- `supervise_claude.sh` resumes a run from its checkpoint until all six datasets are done, then writes `done_<name>` in `$CLAUDE_RUN_STATE` (default `/tmp/synthagent_claude`).
- `pusher_claude.sh` commits and pushes `results/claude` every 30 minutes while runs are active.
- `cost_claude.py` prints the spend so far and the projected total per run, from the checkpointed token counts.
- To pause: `touch $CLAUDE_RUN_STATE/pause_claude` and stop the `run_batch_pipeline.py` processes; remove the file and re-run `start_claude.sh` to resume.
