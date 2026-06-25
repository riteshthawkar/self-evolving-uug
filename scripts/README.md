# BLIP3o Self-Evolving Launchers

This directory contains the public BLIP3o launchers used for self-evolving
training and ablations. Run them from the repository root and provide a local
unlabeled image directory through `DATA_DIR`.

## Launchers

| Script | Schedule | Trained components |
| --- | --- | --- |
| `E1_main_joint.sh` | 3 understanding + 2 generation | Solver, proposer, generator adapter, DiT LoRA |
| `E2_understanding_only.sh` | understanding only | Solver, proposer |
| `E3_generation_only.sh` | generation only | Generator adapter, DiT LoRA |
| `E4_no_dit_rwr.sh` | joint, no DiT RWR | Solver, proposer, generator adapter |
| `E5_synthetic_loop.sh` | imageless joint loop | Solver, proposer, generator adapter, DiT LoRA |
| `E6_single_step.sh` | generation-centered unified step | Solver, proposer, generator adapter, DiT LoRA |
| `E7_two_stage.sh` | understanding stage then generation stage | Stage-specific adapters |
| `E7_two_stage_10kU_10kG.sh` | fixed 10k + 10k two-stage schedule | Stage-specific adapters |

## Example

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
DATA_DIR=/path/to/images \
OUTPUT_DIR=$PWD/outputs/blip3o/E1_main_joint \
bash scripts/E1_main_joint.sh
```

For a short smoke test:

```bash
TOTAL_STEPS=20 \
ALLOW_SMALL_DATA=1 \
DATA_DIR=/path/to/small/image/folder \
OUTPUT_DIR=$PWD/outputs/smoke/blip3o \
bash scripts/E1_main_joint.sh
```

Remove the smoke-test overrides for full training. A full run expects at least
10,000 unlabeled images unless `ALLOW_SMALL_DATA=1` is set.

## Resume

Checkpoints are saved under the configured `OUTPUT_DIR` as `step_NNNNN`
directories. A checkpoint is complete when it contains `SAVE_OK`.

```bash
RESUME_FROM=/path/to/output_dir bash scripts/E1_main_joint.sh
RESUME_FROM=/path/to/output_dir/step_010000 bash scripts/E1_main_joint.sh
```

Resume restores trainable adapter weights, DiT LoRA weights, optimizer state,
KL coefficients, reward baselines, entropy/difficulty windows, and RNG state.
Replay buffers are rebuilt during training.

## Monitoring

Each run writes:

| Path | Contents |
| --- | --- |
| `status.json` | Latest progress, phase, checkpoint, and error state |
| `iter_log.jsonl` | Per-step structured records |
| `logs/training_watch.log` | Compact human-readable training trace |
| `logs/training_monitor.jsonl` | Structured monitor events |
| `logs/training_monitor.tsv` | Tabular monitor events |

## Evaluation

After training, pass a finished checkpoint to the BLIP3o evaluation scripts:

```bash
CHECKPOINT_DIR=/path/to/step_N bash BLIP3o/eval/understanding_eval_our.sh
CHECKPOINT_DIR=/path/to/step_N bash BLIP3o/eval/geneval/generation_our.sh
CHECKPOINT_DIR=/path/to/step_N bash BLIP3o/eval/dpg_bench/generate_dpg_our.sh
CHECKPOINT_DIR=/path/to/step_N bash BLIP3o/eval/wise/generate_wise_our.sh
```

WISE evaluation requires `OPENAI_API_KEY` in the environment.
