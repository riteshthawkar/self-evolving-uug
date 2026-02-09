# Unified Self-Evolving Experiments

- `00_x00_main_method.sh`: alternating understanding + generation co-evolution run.
- Strict main-only defaults:
- model: `BLIP3o/BLIP3o-NEXT-SFT`
- loader: `BLIP3O_REPO=$REPO_ROOT/BLIP3o`, `BLIP3O_USE_LOCAL_CLASSES=1`
- If you explicitly want original `BLIP3o/BLIP3o-Model-8B`, you must set `BLIP3O_REPO` to an external original BLIP3o checkout and keep `BLIP3O_USE_LOCAL_CLASSES=1`.
- Script enables `--allow_missing_generation_tokens` and `--generator_missing_trace_strategy proxy` so generator updates can still run via image-conditioned proxy captions when token traces are unavailable.

Logs/checkpoints are written under:
- `$OUTPUT_ROOT/X00_main_method/<run_name>/`
- `logs/proposer_prompts.jsonl`
- `logs/generation_candidates.jsonl`
- `logs/rewards.jsonl`
- `logs/policy_updates.jsonl`
- `iter_log.jsonl`
