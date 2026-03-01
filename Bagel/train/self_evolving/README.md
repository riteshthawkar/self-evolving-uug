# BAGEL Self-Evolving

This module now supports both rollout-only analysis and train-mode updates on
BAGEL with role-specific LoRA adapters.

## Supported modes

- `rollout`:
  proposer/solver/self-consistency diagnostics + JSONL logs only.
- `train`:
  rollout plus optimizer-backed proposer/solver/generator updates with
  `reinforce` or `grpo`-style normalized advantages.
- `unified_self_evolving`:
  alternating understanding/generation phases with generation→understanding
  replay mixing (buffer or folder-backed pool).

## Entrypoint

```bash
python3 train/train_self_evolving.py \
  --experiment understanding_self_evolving \
  --model_path /path/to/BAGEL-7B-MoT \
  --image_dir /path/to/images \
  --output_dir /path/to/outputs \
  --steps 500
```

## Train mode (LoRA + policy updates)

```bash
python3 train/train_self_evolving.py \
  --experiment understanding_self_evolving \
  --model_path /path/to/BAGEL-7B-MoT \
  --image_dir /path/to/images \
  --output_dir /path/to/outputs \
  --steps 500 \
  --enable_lora \
  --policy_updates_enabled \
  --policy_update_method grpo
```

## Unified mode

```bash
python3 train/train_self_evolving.py \
  --experiment unified_self_evolving \
  --model_path /path/to/BAGEL-7B-MoT \
  --image_dir /path/to/images \
  --output_dir /path/to/outputs \
  --steps 2000 \
  --suder_generation_enabled \
  --understanding_steps_per_cycle 3 \
  --generation_steps_per_cycle 2 \
  --gen_mix_source_mode buffer \
  --enable_lora \
  --policy_updates_enabled \
  --policy_update_method grpo
```

Generation-only loop (no U-phase):

```bash
python3 train/train_self_evolving.py \
  --experiment generation_self_evolving \
  --model_path /path/to/BAGEL-7B-MoT \
  --image_dir /path/to/images \
  --output_dir /path/to/outputs \
  --steps 1000 \
  --suder_generation_enabled \
  --generation_steps_per_cycle 1 \
  --understanding_steps_per_cycle 0
```

Key train-mode flags:

- `--lora_rank`, `--lora_alpha`, `--lora_dropout`
- `--lora_target_modules_csv`
- `--lora_role_adapters_csv` (default: `proposer,solver,generator`)
- `--understanding_skip_no_acceptable`
- `--understanding_require_acceptable_for_update`
- `--understanding_update_require_disagreement`
- `--proposer_reject_unsolvable`
- `--solver_skip_unsolvable_updates`
- `--policy_lr`, `--policy_grad_accum_steps`, `--policy_max_grad_norm`
- `--checkpoint_every`, `--resume_from`

## SUDER generation phase

Enable generation rollouts and generation-side proposer + generator updates:

```bash
python3 train/train_self_evolving.py \
  ... \
  --suder_generation_enabled \
  --train_generation_proposer \
  --train_generator \
  --proposer_gen_entropy_weight 0.7
```

Outputs:

- `rollouts.jsonl`: understanding phase traces + update diagnostics
- `generation_rollouts.jsonl`: SUDER generation traces + update diagnostics
- `metrics.jsonl`: periodic heartbeat records (`log_every`) + final summary/error rows
- `status.json`: latest run state snapshot (`running`/`completed`/`failed`) with progress + ETA
- `summary.json`: run-level metrics
- `checkpoints/step_*.pt`: trainer state + LoRA model state (resume-compatible)
- `checkpoints/step_*_lora/`: role-wise LoRA snapshots for inference parity:
  - `role_proposer.pt`
  - `role_solver.pt`
  - `role_generator.pt`
  - `adapter_roles.json` (role-to-adapter manifest)

To initialize runtime from a specific LoRA checkpoint:

```bash
python3 train/train_self_evolving.py \
  --experiment understanding_self_evolving \
  --model_path /path/to/BAGEL-7B-MoT \
  --image_dir /path/to/images \
  --output_dir /path/to/out \
  --enable_lora \
  --lora_checkpoint_path /path/to/checkpoints/step_001000_lora
```

Backfill role-wise LoRA exports from older `step_*.pt` checkpoints:

```bash
python3 train/self_evolving/export_role_lora_from_checkpoint.py \
  --checkpoint /path/to/checkpoints/step_001000.pt \
  --overwrite
```
