# Final Experiments — Self-Evolving Training for Unified Understanding & Generation

## Competitive Positioning

| Framework | Understanding | Generation | Architecture | Method | External Supervision |
|-----------|:------------:|:----------:|:------------:|--------|:--------------------:|
| **SUDER** | Maintained | +5% GenEval | AR only (Janus-Pro) | RL dual self-rewards | None |
| **UniCorn** | Maintained | +4% GenEval, +8.8% DPG | AR + Hybrid | SFT on self-generated data | None |
| **CoRL/ULM-R1** | +23% | +7% | AR only (Janus-Pro-1B) | GRPO + verifiable rewards | Ground-truth labels + T-I matching model |
| **Ours** | **Improves** | **Improves** | **Diffusion-based** (BLIP3o) | RL (GRPO + RWR) | **None** |

## Our Key Differentiators

1. **First self-evolving RL for diffusion-based UUG models** — all prior work uses AR-based generation
2. **Joint Generator conditioning + DiT LoRA training** — gradients flow from denoising loss through the Generator conditioning adapter
3. **Reward-Weighted Regression for diffusion** — continuous-action analogue of GRPO
4. **Fully self-supervised** — no external labels, reward models, or ground-truth answers
   (CoRL/ULM-R1 requires verifiable rewards = labeled data)
5. **RL-trained curriculum proposer** — learns to generate progressively harder questions
6. **Fully imageless self-evolving loop** (E5) — ZERO external images; proposer imagines scenes from text topics, generator creates them, solver learns from them

## Experiment Overview

| ID | Script | Cycle | Components Trained | What It Proves |
|----|--------|-------|-------------------|----------------|
| **E1** | `E1_main_joint.sh` | 3U + 2G | Solver + Proposer + Generator/DiT LoRA | **Main result**: both tasks improve |
| E2 | `E2_understanding_only.sh` | 5U + 0G | Solver + Proposer | Understanding improves in isolation |
| E3 | `E3_generation_only.sh` | 0U + 5G | Generator/DiT LoRA | Generation improves in isolation |
| E4 | `E4_no_dit_rwr.sh` | 3U + 2G | Solver + Proposer (no diffusion Generator update) | DiT LoRA/RWR is essential for generation |
| **E5** | `E5_synthetic_loop.sh` | 3U + 2G* | Solver‡ + Proposer + Generator/DiT LoRA | **Fully imageless**: understanding trains on self-generated images |
| E6 | `E6_single_step.sh` | 0U + 5G† | Solver + Proposer + Generator/DiT LoRA | Unified step: all components update simultaneously |
| **E7** | `E7_two_stage.sh` | 10k U then 10k G | Stage 1 Solver + Proposer, Stage 2 Generator/DiT LoRA | Rebuttal control: sequential training vs joint alternation |

\* E5 runs 3U+2G like E1, but in strict imageless mode it starts with generation bootstrap and uses generated-only understanding (no real-image fallback)
† E6 runs only G-steps, but solver + proposer also train every step via `gen_step_solver_update_enabled` + `proposer_gen_reward_enabled` — all 4 components update per step
‡ Solver in E5 trains primarily on self-generated images via replay buffer + gen_step_solver_update. Uses `--imageless_proposer_mode` + `gen_mix_ratio=1.0`

## Paper Setup

The paper/rebuttal setup is tracked under `scripts/self_evolving/paper/`.

```bash
# Build the paper's 6k five-source image pool.
bash scripts/self_evolving/paper/prepare_data_6k.sh

# Audit data, scripts, required environment variables, and run progress.
python scripts/self_evolving/paper/check_experiment_readiness.py

# Check that paper-table launch IDs and trainer knobs are implemented.
python scripts/self_evolving/paper/validate_protocol.py

# Launch a manifest experiment with paper defaults.
bash scripts/self_evolving/paper/run_experiment.sh blip3o_joint
bash scripts/self_evolving/paper/run_experiment.sh blip3o_two_stage

# Launch paper/supplement/rebuttal variants without editing scripts.
bash scripts/self_evolving/paper/run_experiment.sh blip3o_ablate_no_ste
bash scripts/self_evolving/paper/run_experiment.sh blip3o_ablate_no_prompt_perturbation
bash scripts/self_evolving/paper/run_experiment.sh blip3o_ste_mean
bash scripts/self_evolving/paper/run_experiment.sh blip3o_lora_r32
bash scripts/self_evolving/paper/run_experiment.sh blip3o_pps_n9
bash scripts/self_evolving/paper/run_experiment.sh blip3o_reward_cycle_0p30

# Parameter-strategy table.
bash scripts/self_evolving/paper/run_experiment.sh blip3o_strategy_lora
bash scripts/self_evolving/paper/run_experiment.sh blip3o_strategy_qlora
bash scripts/self_evolving/paper/run_experiment.sh blip3o_strategy_full_finetune
GENERATED_MIX_DIR=/path/to/generated_mix_pool \
  bash scripts/self_evolving/paper/run_experiment.sh blip3o_strategy_sft_self_generated

# Extract rebuttal answer-length and compute-cost tables from logs.
python scripts/self_evolving/paper/rebuttal_stats.py --output-md outputs/paper/rebuttal_stats.md

# Run generation benchmarks for a finished checkpoint.
CHECKPOINT_DIR=/path/to/step_010000 bash scripts/self_evolving/paper/run_generation_evals.sh blip3o
```

Defaults now point to `data/joint_6k/images`, matching the manuscript protocol.
Legacy partial runs under `runs/final/` remain discoverable by the readiness audit
but new paper runs are written under `outputs/`.
The two-stage rebuttal control is intentionally different: it defaults to
`data/joint_pool_10k/images`, runs 10k understanding-only steps, then resumes
from `step_010000` and runs 10k additional generation-only steps to
`step_020000`. It passes `MAX_IMAGES=10000` into both stages so they use the
same deterministic image subset. Override with
`TWO_STAGE_DATA_DIR=/path/to/10k/images` if the remote pool lives elsewhere.
All table-level BLIP3o variants are recorded in
`scripts/self_evolving/paper/paper_experiments.json` under
`blip3o_table_reproduction`.
The SFT self-generated-data baseline consumes BLIP3o generated-mix sidecars and
packages them as webdataset shards with
`scripts/self_evolving/paper/prepare_self_generated_sft_webdataset.py`.

## Ablation Matrix

| Comparison | What It Isolates | Expected Outcome |
|-----------|-----------------|------------------|
| E1 vs Baseline | Full framework effect | Both metrics improve |
| E1 vs E7 | Joint alternation vs sequential specialization | Joint should preserve both tasks better |
| E1 vs E2 | Generation's effect on understanding | Joint ≥ understanding-only |
| E1 vs E3 | Understanding's effect on generation | Joint ≥ generation-only |
| E1 vs E4 | DiT LoRA/RWR contribution | Diffusion Generator LoRA is essential for generation quality |
| **E1 vs E5** | **Real images vs fully imageless loop** | E5 still improves → TRUE self-evolution (zero external images)! |
| E1 vs E6 | Dedicated U-steps vs unified single-step | Reveals if separate understanding training is needed |
| E2 + E3 vs E1 | Synergy of joint training | Joint > sum of independent parts |

## The E5 Story (Key Differentiator — Fully Imageless)

E5 is the most novel experiment. In strict imageless mode, **ZERO external images**
are used by the training loop.
The proposer imagines scenes from text topics, the generator creates them, and
the solver learns to understand them — a fully autonomous learning loop:

```
┌──────────────────────────────────────────────────────────────────────┐
│  ┌──────────────── G-STEPS (2 per cycle) ──────────────────┐        │
│  │                                                          │        │
│  │  Text topic → Proposer (text-only) → prompt + QA spec   │        │
│  │  (75 themes)   (imagines scene)       (verifiable)       │        │
│  │                                                          │        │
│  │  prompt → Generator creates candidates                   │        │
│  │        → Solver scores candidates → GRPO reward          │        │
│  │        → DiT LoRA RWR update                             │        │
│  │        → Best image → REPLAY BUFFER (quality-gated)      │        │
│  │        → Solver LoRA update (gen_step_solver_update)      │        │
│  │        → Proposer dual reward → Proposer LoRA update     │        │
│  └──────────────────────────┬───────────────────────────────┘        │
│                             │                                        │
│                    replay buffer                                     │
│                    (generated imgs)                                   │
│                             │                                        │
│  ┌──────────────── U-STEPS (3 per cycle) ──────────────────┐        │
│  │                             ↓                            │        │
│  │  Sample image from replay buffer (100% generated)        │        │
│  │  → Proposer writes QA for generated image                │        │
│  │  → Solver answers QA → GRPO reward → Solver LoRA update  │        │
│  │  → Proposer reward → Proposer LoRA update                │        │
│  │                                                          │        │
│  │  (if buffer empty at start → U-step skipped, not real)   │        │
│  └──────────────────────────────────────────────────────────┘        │
│                                                                      │
│  ZERO real images at any point. True closed-loop self-evolution.     │
└──────────────────────────────────────────────────────────────────────┘
```

No competitor can claim this:
- **CoRL** needs ground-truth answer labels for understanding rewards
- **SUDER** uses real images for both understanding and generation
- **UniCorn** generates SFT data but from real image seeds with external quality
- **E5** proves understanding can improve from **purely self-generated** visual data
  with **no external image supervision whatsoever**

Key implementation details:
- `--imageless_proposer_mode` enables text-only proposer generation
- `--replay_buffer_size 500` stores best generated images for U-step consumption
- `--gen_mix_ratio_start 1.0 --gen_mix_ratio_max 1.0` forces 100% generated in U-steps
- 75 diverse topics cover counting, spatial relations, text, charts, color, objects, etc.
- Proposer creates QA pairs based on *expectations* of what the image should contain
- `--no_ref_answer_scoring` is enforced for imageless mode
- `--strict_imageless_mode` + `--understanding_generated_only` disallow real-image fallback
- `--cycle_starts_with_generation` + `--bootstrap_generated_pool_steps` prefill generated pool before U-steps

## Priority Order (Given 10-Day Constraint)

1. **E1** (must run) — Main result, ~2 days
2. **E5** (must run) — Key novelty claim, ~1.5 days
3. **E2 + E3** (important) — Joint training ablation, ~2-3 days
4. **E4** (important) — DiT ablation, ~1.5 days
5. **E6** (nice to have) — Cycle ratio study, ~1.5 days

If time is tight, drop E6 first, then E4. E1 + E5 + E2 + E3 are the minimum for a strong paper.

## Usage

```bash
# Main experiment (HIGHEST PRIORITY) — defaults to strict mode
bash E1_main_joint.sh

# Fully imageless self-evolving loop (KEY NOVELTY — zero external images)
bash E5_synthetic_loop.sh

# Component ablations
bash E2_understanding_only.sh
bash E3_generation_only.sh
bash E4_no_dit_rwr.sh

# Unified single-step ablation
bash E6_single_step.sh

# Override stage if needed (all scripts default to strict)
TRAIN_STAGE=warmup bash E1_main_joint.sh

# Resume from checkpoint (point to any step_NNNNN directory or its parent)
RESUME_FROM=/path/to/step_N bash E1_main_joint.sh

# Resume with proposer baseline reset (useful if reward distribution shifted)
RESUME_FROM=/path/to/step_N RESET_PROPOSER_BASELINE=1 bash E1_main_joint.sh
```

### Resuming Terminated Experiments

Checkpoints are saved every **50 steps** (`--save_every 50`) under the experiment's
`OUTPUT_DIR` (e.g. `$REPO_ROOT/runs/final/E1_main_joint/`).  Each checkpoint
is a directory named `step_NNNNN/` containing LoRA adapters, optimizer states,
baselines, and RNG states.

For live run health checks, use `logs/training_watch.log` in the run folder.
It keeps one short human-readable line per U/G step with reward, entropy or
spec quality, CE/KL, optimizer-step flags, skip reasons, and NaN/Inf fields.
For spreadsheet/debug use, `logs/training_monitor.tsv` and
`logs/training_monitor.jsonl` keep the same signal in structured form. Full
per-sample traces remain in `iter_log.jsonl` and `logs/policy_updates.jsonl`.

Each launch also mirrors lightweight metadata beside the training package at
`BLIP3o/blip3o/train/self_evolving/training_runs/<run_name>/`: `config.json`,
`git_info.json`, `environment.json`, and `output_dir.txt`. Override this with
`SELF_EVOLVING_CODE_RUN_REGISTRY=/path/to/registry` or disable it with
`--disable_code_run_registry`.

**Step 1 — Find the latest valid checkpoint:**
```bash
# List checkpoints (look for SAVE_OK marker = completed save)
ls -d "$REPO_ROOT"/runs/final/E1_main_joint/step_* | while read d; do
  [[ -f "$d/SAVE_OK" ]] && echo "OK: $d" || echo "INCOMPLETE: $d"
done
```

**Step 2 — Resume:**
```bash
# Point RESUME_FROM to the latest complete checkpoint
RESUME_FROM="$REPO_ROOT"/runs/final/E1_main_joint/step_00250 bash E1_main_joint.sh

# Or just point to the parent dir — the trainer auto-picks the newest valid checkpoint
RESUME_FROM="$REPO_ROOT"/runs/final/E1_main_joint bash E1_main_joint.sh
```

**What gets restored:** LoRA adapter weights (solver/proposer/generator), DiT LoRA
trainable params, optimizer states, KL coefficients, reward baselines,
entropy/difficulty windows, and RNG states for deterministic resumption.

**Note:** The replay buffer (E5) is **not** persisted to disk — it refills
naturally within the first few G-steps after resume.

## Evaluation Benchmarks

After training, evaluate each experiment on:

**Understanding:** MME-P, MME-C, MMMU, RealWorldQA, TextVQA, SEED
```bash
CHECKPOINT_DIR=/path/to/step_N bash ../../BLIP3o/eval/understanding_eval_our.sh
```

**Generation:** GenEval, DPG-Bench, WISE
```bash
CHECKPOINT_DIR=/path/to/step_N bash ../../BLIP3o/eval/geneval/generation_our.sh
CHECKPOINT_DIR=/path/to/step_N bash ../../BLIP3o/eval/dpg_bench/generate_dpg_our.sh
CHECKPOINT_DIR=/path/to/step_N bash ../../BLIP3o/eval/wise/generate_wise_our.sh
```
