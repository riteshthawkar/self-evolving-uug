# Unified Self-Evolving Experiments

- `00_x00_main_method.sh`: alternating understanding + generation co-evolution run.
- Default model: `BLIP3o/BLIP3o-Model-8B` (original BLIP3o).
- Default loader mode: original-safe (`BLIP3O_USE_LOCAL_CLASSES=0`, `BLIP3O_REPO=""`).
- If your transformers stack cannot load `blip3o_qwen` via remote code, set `BLIP3O_REPO` to an original BLIP3o `main` checkout and set `BLIP3O_USE_LOCAL_CLASSES=1`.
- Script enables `--allow_missing_generation_tokens` and `--generator_missing_trace_strategy proxy` so generator updates can still run via image-conditioned proxy captions when token traces are unavailable.

Logs/checkpoints are written under:
- `$OUTPUT_ROOT/X00_main_method/<run_name>/`
- `logs/proposer_prompts.jsonl`
- `logs/generation_candidates.jsonl`
- `logs/rewards.jsonl`
- `logs/policy_updates.jsonl`
- `iter_log.jsonl`
