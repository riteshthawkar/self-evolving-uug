# BAGEL Self-Evolving (Phase 1)

This module implements the first production-grade slice of the self-evolving
framework on BAGEL:

- understanding rollouts on real images
- proposer/solver self-consistency signals
- V-Zero style dual-track reward shaping
- structured JSONL logging for analysis and later policy updates

## Entrypoint

Run:

```bash
python3 train/train_self_evolving.py \
  --model_path /path/to/BAGEL-7B-MoT \
  --image_dir /path/to/images \
  --output_dir /path/to/outputs \
  --steps 500
```

## What this phase does

For each step:

1. proposer generates one objective visual question
2. solver answers the question with multi-temperature sampling
3. compute entropy / majority fraction from sampled answers
4. run one greedy intuitive solver pass
5. compute proposer reward with dual-track shaping
6. persist full record to `rollouts.jsonl`

## What this phase intentionally does not do yet

- policy gradient or GRPO optimization updates
- distributed training/FSDP for self-evolving loops
- generation-phase optimization loop

These are extension points for phase 2+ and the current code is organized to
add them cleanly without changing rollout correctness.

