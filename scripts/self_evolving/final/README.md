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
| **E5** | `E5_synthetic_loop.sh` | 0U + 5G* | Solver† + Generator + DiT + Proposer | **Fully imageless**: understanding improves from self-generated images |
| E6 | `E6_single_step.sh` | 1U + 1G | Solver + Generator + DiT + Proposer | Optimal U:G ratio for co-evolution |
| *E7* | *(X09 step 650)* | 3U + 2G | Same as E1, easy data | Data difficulty matters |

\* E5 runs only generation steps, but solver trains on generated images via `gen_step_solver_update_enabled`
† Solver in E5 never sees real images — only model-generated ones
‡ E5 uses `--imageless_proposer_mode`: proposer generates specs from text topics, never seeing any real image

## Ablation Matrix

| Comparison | What It Isolates | Expected Outcome |
|-----------|-----------------|------------------|
| E1 vs Baseline | Full framework effect | Both metrics improve |
| E1 vs E7 (X09) | Data quality impact | Harder data → solver trains → understanding improves |
| E1 vs E2 | Generation's effect on understanding | Joint ≥ understanding-only |
| E1 vs E3 | Understanding's effect on generation | Joint ≥ generation-only |
| E1 vs E4 | DiT RWR contribution | DiT training is essential for generation quality |
| **E1 vs E5** | **Real images vs fully imageless loop** | E5 still improves → TRUE self-evolution (zero external images)! |
| E1 vs E6 | Cycle ratio (3:2 vs 1:1) | Reveals optimal coupling frequency |
| E2 + E3 vs E1 | Synergy of joint training | Joint > sum of independent parts |

## The E5 Story (Key Differentiator — Fully Imageless)

E5 is the most novel experiment. **ZERO external images** are used at any point.
The proposer imagines scenes from text topics, the generator creates them, and
the solver learns to understand them — a fully autonomous learning loop:

```
┌─────────────────────────────────────────────────────────────────┐
│  Text topic ──→ Proposer (text-only) ──→ generation prompt + QA │
│  (sampled from     (imagines a scene,      (verifiable specs    │
│   75 themes)        writes prompt + QA)      for the generator) │
│                                                                 │
│  prompt ──→ Generator creates candidate images                  │
│         ──→ Solver picks best image (verification)              │
│                                                                 │
│  Generated image ──→ Solver answers QA ──→ GRPO reward          │
│       ↑                                        │                │
│       │              reward signal              ↓                │
│  DiT RWR update ←────────────────────→ Solver LoRA update       │
│       ↑                                        │                │
│       │          dual proposer reward           │                │
│       └──────── Proposer LoRA update ───────────┘                │
│                                                                 │
│  Better proposer → richer prompts → better images               │
│  Better generator → more faithful images → better solver        │
│  Better solver → better verification → better reward → loop     │
└─────────────────────────────────────────────────────────────────┘
```

No competitor can claim this:
- **CoRL** needs ground-truth answer labels for understanding rewards
- **SUDER** uses real images for both understanding and generation
- **UniCorn** generates SFT data but from real image seeds with external quality
- **E5** proves understanding can improve from **purely self-generated** visual data
  with **no external image supervision whatsoever**

Key implementation details:
- `--imageless_proposer_mode` enables text-only proposer generation
- 75 diverse topics cover counting, spatial relations, text, charts, color, objects, etc.
- Proposer creates QA pairs based on *expectations* of what the image should contain
- `use_ref_answer_scoring` auto-disabled (no real image to generate reference answers)
- Alignment scoring skipped (no source image to compare against)

## Priority Order (Given 10-Day Constraint)

1. **E1** (must run) — Main result, ~2 days
2. **E5** (must run) — Key novelty claim, ~1.5 days
3. **E2 + E3** (important) — Joint training ablation, ~2-3 days
4. **E4** (important) — DiT ablation, ~1.5 days
5. **E6** (nice to have) — Cycle ratio study, ~1.5 days

If time is tight, drop E6 first, then E4. E1 + E5 + E2 + E3 are the minimum for a strong paper.

## Usage

```bash
# Main experiment (HIGHEST PRIORITY)
TRAIN_STAGE=warmup bash E1_main_joint.sh

# Fully imageless self-evolving loop (KEY NOVELTY — zero external images)
TRAIN_STAGE=warmup bash E5_synthetic_loop.sh

# Component ablations
TRAIN_STAGE=warmup bash E2_understanding_only.sh
TRAIN_STAGE=warmup bash E3_generation_only.sh
TRAIN_STAGE=warmup bash E4_no_dit_rwr.sh

# Cycle ratio ablation (nice to have)
TRAIN_STAGE=warmup bash E6_single_step.sh

# Resume from checkpoint
RESUME_FROM=/path/to/step_N TRAIN_STAGE=warmup bash E1_main_joint.sh
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
