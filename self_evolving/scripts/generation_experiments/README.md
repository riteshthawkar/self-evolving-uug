# Generation Self-Evolving Experiments

- `00_g00_main_method.sh`: main generation-only self-evolving run.
- Default model: `BLIP3o/BLIP3o-Model-8B` (original BLIP3o).
- Default loader mode: auto (`BLIP3O_USE_LOCAL_CLASSES=auto`, `BLIP3O_REPO=""`), which avoids forcing BLIP3o-NEXT local classes for original BLIP3o checkpoints.
- Script enables `--allow_missing_generation_tokens` because original BLIP3o diffusion decoder may not return token traces in all environments.

Logs/checkpoints are written under:
- `$OUTPUT_ROOT/G00_main_method/<run_name>/`
- `logs/proposer_prompts.jsonl`
- `logs/generation_candidates.jsonl`
- `logs/rewards.jsonl`
- `logs/policy_updates.jsonl`
- `iter_log.jsonl`
