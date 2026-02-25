# BAGEL Self-Evolving

This module now supports both rollout-only analysis and train-mode updates on
BAGEL with role-specific LoRA adapters.

## Supported modes

- `rollout`:
  proposer/solver/self-consistency diagnostics + JSONL logs only.
- `train`:
  rollout plus optimizer-backed proposer/solver updates with
  `reinforce` or `grpo`-style normalized advantages.

## Entrypoint

```bash
python3 train/train_self_evolving.py \
  --model_path /path/to/BAGEL-7B-MoT \
  --image_dir /path/to/images \
  --output_dir /path/to/outputs \
  --steps 500
```

## Train mode (LoRA + policy updates)

```bash
python3 train/train_self_evolving.py \
  --model_path /path/to/BAGEL-7B-MoT \
  --image_dir /path/to/images \
  --output_dir /path/to/outputs \
  --steps 500 \
  --enable_lora \
  --policy_updates_enabled \
  --policy_update_method grpo
```

Key train-mode flags:

- `--lora_rank`, `--lora_alpha`, `--lora_dropout`
- `--lora_target_modules_csv`
- `--lora_role_adapters_csv` (default: `proposer,solver,generator`)
- `--policy_lr`, `--policy_grad_accum_steps`, `--policy_max_grad_norm`
- `--checkpoint_every`, `--resume_from`

## SUDER generation phase

Enable generation rollouts and proposer generation-side updates:

```bash
python3 train/train_self_evolving.py \
  ... \
  --suder_generation_enabled \
  --train_generation_proposer \
  --proposer_gen_entropy_weight 0.7
```

Outputs:

- `rollouts.jsonl`: understanding phase traces + update diagnostics
- `generation_rollouts.jsonl`: SUDER generation traces + update diagnostics
- `summary.json`: run-level metrics
- `checkpoints/step_*.pt`: adapter/update state checkpoints (train mode)
