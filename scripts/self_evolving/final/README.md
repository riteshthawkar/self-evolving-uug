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
2. **Joint LLM+DiT training** — gradients flow from denoising loss through LLM conditioning encoder
3. **Reward-Weighted Regression for diffusion** — continuous-action analogue of GRPO
4. **Fully self-supervised** — no external labels, reward models, or ground-truth answers
   (CoRL/ULM-R1 requires verifiable rewards = labeled data)
5. **RL-trained curriculum proposer** — learns to generate progressively harder questions
6. **Fully imageless self-evolving loop** (E5) — ZERO external images; proposer imagines scenes from text topics, generator creates them, solver learns from them

## Experiment Overview

| ID | Script | Cycle | Components Trained | What It Proves |
|----|--------|-------|-------------------|----------------|
| **E1** | `E1_main_joint.sh` | 3U + 2G | Solver + Generator + DiT + Proposer | **Main result**: both tasks improve |
| E2 | `E2_understanding_only.sh` | 5U + 0G | Solver + Proposer | Understanding improves in isolation |
| E3 | `E3_generation_only.sh` | 0U + 5G | Generator + DiT + Proposer | Generation improves in isolation |
| E4 | `E4_no_dit_rwr.sh` | 3U + 2G | Solver + Generator + Proposer (no DiT) | DiT training is essential for generation |
| **E5** | `E5_synthetic_loop.sh` | 3U + 2G* | Solver‡ + Generator + DiT + Proposer | **Fully imageless**: understanding trains on self-generated images |
| E6 | `E6_single_step.sh` | 0U + 5G† | Solver + Generator + DiT + Proposer | Unified step: all components update simultaneously |
| *E7* | *(X09 step 650)* | 3U + 2G | Same as E1, easy data | Data difficulty matters |

\* E5 runs 3U+2G like E1, but U-steps use 100% generated images from replay buffer (never real images)
† E6 runs only G-steps, but solver + proposer also train every step via `gen_step_solver_update_enabled` + `proposer_gen_reward_enabled` — all 4 components update per step
‡ Solver in E5 never sees real images — trains on self-generated images via replay buffer + gen_step_solver_update. Uses `--imageless_proposer_mode` + `--understanding_generated_only`

## Ablation Matrix

| Comparison | What It Isolates | Expected Outcome |
|-----------|-----------------|------------------|
| E1 vs Baseline | Full framework effect | Both metrics improve |
| E1 vs E7 (X09) | Data quality impact | Harder data → solver trains → understanding improves |
| E1 vs E2 | Generation's effect on understanding | Joint ≥ understanding-only |
| E1 vs E3 | Understanding's effect on generation | Joint ≥ generation-only |
| E1 vs E4 | DiT RWR contribution | DiT training is essential for generation quality |
| **E1 vs E5** | **Real images vs fully imageless loop** | E5 still improves → TRUE self-evolution (zero external images)! |
| E1 vs E6 | Dedicated U-steps vs unified single-step | Reveals if separate understanding training is needed |
| E2 + E3 vs E1 | Synergy of joint training | Joint > sum of independent parts |

## The E5 Story (Key Differentiator — Fully Imageless)

E5 is the most novel experiment. **ZERO external images** are used at any point.
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
│  │        → DiT RWR update                                  │        │
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
- `--understanding_generated_only` ensures U-steps NEVER fall back to real images
- `--replay_buffer_size 500` stores best generated images for U-step consumption
- `--gen_mix_ratio_start 1.0 --gen_mix_ratio_max 1.0` forces 100% generated in U-steps
- 75 diverse topics cover counting, spatial relations, text, charts, color, objects, etc.
- Proposer creates QA pairs based on *expectations* of what the image should contain
- `use_ref_answer_scoring` auto-disabled (no real image to generate reference answers)
- Startup: Cycle 1 U-steps skip (buffer empty) → G-steps fill buffer → Cycle 2+ runs full loop

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

# Resume from checkpoint
RESUME_FROM=/path/to/step_N bash E1_main_joint.sh
```

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
