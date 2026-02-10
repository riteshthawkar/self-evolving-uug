# Experiments and Ablation Plan

> **Document Version**: 2.0
> **Last Updated**: 2026-02-10
> **Pipeline**: Self-Evolving Unified Understanding + Generation (BLIP3o)
> **Entry Point**: `BLIP3o/blip3o/train/train_self_evolving.py`
> **Scripts**: `scripts/self_evolving/`

---

## 0. Overview

This document defines all experiments and ablations needed to validate the
self-evolving unified training pipeline.  The core claim is:

> A single multimodal model can jointly improve its understanding and
> generation capabilities through fully unsupervised self-play, using
> only its own internal consistency signals — no annotations, no external
> reward models, no human labels.

Experiments are organized into four tiers by priority:

| Tier | Purpose | Required for |
|------|---------|-------------|
| **Tier 1** | Core method validation | Submission |
| **Tier 2** | Component ablations | Submission |
| **Tier 3** | Design-choice comparisons | Strengthens paper |
| **Tier 4** | Analysis and visualization | Figures / appendix |

---

## 1. Evaluation Benchmarks

All experiments are evaluated on external benchmarks the model never sees
during self-evolving training.

### 1.1 Understanding Benchmarks

| Benchmark | What it measures | Metric |
|-----------|-----------------|--------|
| VQAv2 (test-dev) | General visual question answering | Accuracy |
| TextVQA | OCR-grounded reasoning | Accuracy |
| MMBench | Comprehensive multimodal understanding | Accuracy |
| POPE | Hallucination tendency | F1 / Accuracy |
| SEED-Bench | Generative comprehension | Accuracy |
| ChartQA | Chart and diagram reasoning | Accuracy |
| MathVista | Mathematical visual reasoning | Accuracy |

### 1.2 Generation Benchmarks

| Benchmark | What it measures | Metric |
|-----------|-----------------|--------|
| GenEval | Compositional generation (objects, attributes, relations) | Overall score |
| DPG-Bench | Dense prompt-guided generation | Alignment score |
| CLIP-score | Image-text alignment (external CLIP) | Cosine similarity |
| FID | Image quality / distribution match | Lower is better |

### 1.3 Self-Judge Audit Metrics

Comparing internal training rewards against external evaluation to validate
that the model is not merely gaming its own reward signals.

| Internal Metric (training) | External Metric (evaluation) |
|---------------------------|------------------------------|
| solver_reward_mean | VQAv2 accuracy |
| spec_score | GenEval overall |
| cycle_score | CLIP-score (external CLIP model) |
| diversity_score | FID (distribution coverage) |

---

## 2. Tier 1 — Core Method Validation

These experiments establish the main result: unified self-evolving training
improves both understanding and generation.

### Table 1: Main Results

| ID | Experiment | Training Mode | Script | Seeds |
|----|-----------|--------------|--------|-------|
| E1.1 | Baseline (no self-evolving) | None — evaluate pretrained model | N/A | N/A |
| E1.2 | Understanding-only | `understanding` | `understanding/00_u00_main_method.sh` | 42, 123, 777 |
| E1.3 | Generation-only | `generation` | `generation/00_g00_main_method.sh` | 42, 123, 777 |
| E1.4 | **Unified (main method)** | `unified` | `unified/00_x00_main_method.sh` | 42, 123, 777 |

**Expected outcome**: E1.4 (Unified) > E1.2 ≈ E1.3 > E1.1 (Baseline) on
both understanding AND generation benchmarks.  This demonstrates the mutual
reinforcement between understanding and generation — the core contribution.

**Key configuration (E1.4)**:
```
--total_steps 10000
--kl_coef 0.01
--grad_accum_steps 4
--save_every 500
--num_solver_samples 5
--num_generations 4
--understanding_steps_per_cycle 3
--generation_steps_per_cycle 2
```

### What to report

- Mean ± std across 3 seeds for every benchmark
- Absolute improvement over baseline (E1.1)
- Understanding-only vs Unified delta (shows generation helps understanding)
- Generation-only vs Unified delta (shows understanding helps generation)

---

## 3. Tier 2 — Component Ablations

Each ablation removes or degrades exactly one component from the full unified
method (E1.4) to measure its individual contribution.

### 3A. Reward Signal Ablations (Generation side)

Ablations on the generation reward composition.  Default weights:
`spec=0.65, cycle=0.20, diversity=0.10, contradiction=0.20`.

| ID | Ablation | Config Change | Hypothesis |
|----|---------|--------------|-----------|
| A2.1 | No cycle consistency | `--reward_cycle_weight 0.0` | Generation quality drops — model cannot self-verify prompt faithfulness |
| A2.2 | No diversity reward | `--reward_diversity_weight 0.0` | Mode collapse — candidates converge to same image style |
| A2.3 | No contradiction penalty | `--reward_contradiction_weight 0.0` | Reward hacking — solver learns to say "yes" to everything |
| A2.4 | Spec-only reward | `--reward_cycle_weight 0.0 --reward_diversity_weight 0.0 --reward_contradiction_weight 0.0` | Worst generation — shows spec score alone is insufficient |
| A2.5 | Jaccard cycle consistency | Revert `_cycle_reward` to Jaccard overlap (token matching) | Shows embedding-based consistency > token overlap |

**Expected ordering**: Full method > A2.1 > A2.5 > A2.4.  A2.2 and A2.3
cause different failure modes (collapse vs hacking) visible in qualitative
samples.

### 3B. Architecture Ablations

| ID | Ablation | Config Change | Hypothesis |
|----|---------|--------------|-----------|
| A2.6 | No proposer evolution | Freeze proposer adapter (no gradient updates) | Weaker curriculum — fixed question/spec difficulty |
| A2.7 | No synthetic solver bridge | `--enable_solver_updates False` in unified mode | Generation phase does not help understanding — breaks mutual benefit claim |
| A2.8 | Single adapter (shared solver + proposer) | Use same LoRA adapter for both roles | Role interference — conflicting optimization objectives |
| A2.9 | No KL regularization | `--kl_coef 0.0` | Mode collapse and catastrophic drift from pretrained capabilities |

### How to report

**Table 2**: For each ablation, report all benchmarks.  Highlight the
specific metric that degrades most to identify what each component protects.

**Recommended format**:
```
| Ablation   | VQAv2 | TextVQA | GenEval | CLIP-score | Δ vs Full |
|-----------|-------|---------|---------|------------|-----------|
| Full (E1.4) | ...  | ...     | ...     | ...        | —         |
| A2.1       | ...  | ...     | ...     | ...        | -X.X      |
| ...        |      |         |         |            |           |
```

---

## 4. Tier 3 — Design Choice Comparisons

### 4A. Generator Update Rule: REINFORCE vs DPO

| ID | Method | Script |
|----|--------|--------|
| C3.1a | REINFORCE (default) | `unified/00_x00_main_method.sh` |
| C3.1b | DPO | `unified/01_x01_dpo_style.sh` |

**What to compare**: Training stability (reward variance over steps),
final benchmark performance, and sensitivity to `dpo_beta` / `dpo_label_smoothing`.

**Hypothesis**: DPO produces more stable gradients with large reward gaps
between best/worst candidates.  REINFORCE is more flexible when reward
signal is noisy or when reward landscape is non-stationary (early training).

### 4B. Hyperparameter Sensitivity (Understanding)

| ID | Parameter | Values | Script |
|----|----------|--------|--------|
| C3.2 | num_solver_samples | {3, 5, 7} | `understanding/01_u01_solver_samples.sh` |
| C3.3 | solver_soft_gamma | {0.5, 0.7, 0.9} | `understanding/02_u02_solver_gamma.sh` |
| C3.4 | proposer_update_freq | {1, 5, 10} | `understanding/03_u03_proposer_update_freq.sh` |
| C3.5 | Entropy band (mu, sigma) | (0.7,0.25), (0.9,0.35), (1.1,0.45) | `understanding/04_u04_entropy_band.sh` |
| C3.6 | KL sensitivity | (kl_coef, kl_target) grid | `understanding/05_u05_kl_sensitivity.sh` |
| C3.7 | LoRA rank | {8, 16, 32} | `understanding/06_u06_lora_capacity.sh` |

**How to report**: Line plots or small tables showing metric vs parameter
value.  Identify the sweet spot and discuss sensitivity.

### 4C. Unified Cycle Ratio

| ID | U:G Ratio | Config |
|----|-----------|--------|
| C3.8a | 1:1 | `--understanding_steps_per_cycle 1 --generation_steps_per_cycle 1` |
| C3.8b | 3:2 (default) | `--understanding_steps_per_cycle 3 --generation_steps_per_cycle 2` |
| C3.8c | 5:1 | `--understanding_steps_per_cycle 5 --generation_steps_per_cycle 1` |

**Hypothesis**: 3:2 balances well because understanding steps are cheaper
and provide tighter solver supervision for generation rewards.  5:1 starves
the generator; 1:1 may destabilize understanding.

### 4D. Reward Weight Sensitivity (Generation)

| ID | (spec, cycle, div, contra) | Notes |
|----|---------------------------|-------|
| C3.9a | (0.65, 0.20, 0.10, 0.20) | Default |
| C3.9b | (0.80, 0.10, 0.05, 0.20) | Spec-heavy |
| C3.9c | (0.50, 0.30, 0.10, 0.20) | Cycle-heavy |
| C3.9d | (0.55, 0.20, 0.20, 0.20) | Diversity-heavy |

---

## 5. Tier 4 — Analysis Experiments

These produce figures and qualitative evidence for the paper.  No new
training runs — they use checkpoints and logs from Tier 1 and 2 runs.

### 5A. Training Dynamics (Figure: learning curves)

**Plot over training steps** (from `iter_log.jsonl` and W&B):

- Solver reward mean (understanding accuracy proxy)
- Spec score mean (generation faithfulness)
- Cycle consistency score (semantic alignment)
- KL divergence per adapter (regularization health)
- Proposer entropy (question difficulty evolution)
- Adaptive beta values (KL controller behavior)

**Overlay**: Understanding-only (E1.2) vs Generation-only (E1.3) vs
Unified (E1.4) on the same axes.  Shows mutual improvement visually.

### 5B. Self-Evolving Curriculum Visualization (Figure: qualitative)

Save proposer outputs and model responses at checkpoints:
steps {100, 500, 1000, 2500, 5000, 10000}.

**Understanding examples** (side-by-side):
- Image + proposed question + solver answers + reward
- Shows questions becoming harder and more compositional over training

**Generation examples** (side-by-side):
- Source image + proposed spec + best generated image + spec score
- Shows specs becoming more specific and generation becoming more faithful

### 5C. Diversity Score Evolution (Figure)

Plot per-candidate diversity scores over training.

**Expected pattern**:
- Early: all candidates similar (low diversity, near-random generation)
- Mid: diversity increases as generator explores the prompt space
- Late: diversity stabilizes at a healthy level (not collapsed, not random)

### 5D. Reward Correlation Analysis (Figure: heatmap)

Compute Pearson correlation matrix between reward components
(spec, cycle, diversity, contradiction) across all training steps.

**Expected**:
- spec ↔ cycle: positive (good images score high on both)
- diversity ↔ spec: weak/zero (orthogonal signal)
- contradiction ↔ spec: negative (contradictions reduce spec score)

### 5E. Self-Judge Bias Audit (Table)

The central methodological concern: since all reward signals come from the
model itself, how do we know it is not simply gaming its own rewards?

**Protocol**:
1. After training, evaluate on all external benchmarks (Section 1)
2. Compare self-reported training metrics with external metrics
3. Compute Spearman rank correlation between internal and external scores
   across checkpoints (steps 500, 1000, ..., 10000)

**If correlation is high** → self-evaluation tracks real improvement.
**If correlation is low** → reward hacking is occurring.

| Checkpoint | solver_reward (internal) | VQAv2 (external) | spec_score (internal) | GenEval (external) |
|-----------|------------------------|------------------|-----------------------|-------------------|
| Step 500  | ... | ... | ... | ... |
| Step 1000 | ... | ... | ... | ... |
| Step 5000 | ... | ... | ... | ... |
| Step 10000| ... | ... | ... | ... |

### 5F. Mutual Benefit Analysis (Figure)

Plot understanding metrics for the solver adapter when it receives synthetic
updates from generation (A2.7 ablation comparison).

**Two curves**:
1. Unified with synthetic solver bridge (E1.4)
2. Unified without synthetic solver bridge (A2.7)

**Expected**: (1) improves faster and reaches higher accuracy, proving that
the generation phase provides useful training signal for the understanding role.

---

## 6. Experiment Priority and Compute Budget

### Phase 1 — Must-Have for Submission

| Experiment | Runs | GPU-hours (est. per run on A100) |
|-----------|------|--------------------------------|
| E1.1 Baseline eval | 1 | 1h (eval only) |
| E1.2 Understanding-only (3 seeds) | 3 | 24h |
| E1.3 Generation-only (3 seeds) | 3 | 36h |
| E1.4 Unified (3 seeds) | 3 | 48h |
| A2.1 No cycle consistency | 1 | 48h |
| A2.5 Jaccard cycle consistency | 1 | 48h |
| A2.7 No synthetic solver bridge | 1 | 48h |
| A2.9 No KL regularization | 1 | 48h |
| C3.1b DPO update rule | 1 | 48h |
| **Total Phase 1** | **15 runs** | **~565 GPU-hours** |

Plus evaluation runs at checkpoints.

### Phase 2 — Strengthens the Paper

| Experiment | Runs |
|-----------|------|
| A2.2, A2.3, A2.4, A2.6, A2.8 | 5 |
| C3.2–C3.7 (hyperparameter sweeps) | ~18 |
| C3.8 Cycle ratio | 2 |
| C3.9 Reward weights | 3 |
| **Total Phase 2** | **~28 runs** |

### Phase 3 — Appendix / Camera-Ready

- Multi-seed runs for all Phase 2 experiments
- Scaling analysis (vary data size, model size)
- Additional qualitative examples

---

## 7. Existing Shell Scripts

### Understanding Experiments

```
scripts/self_evolving/understanding/
├── 00_u00_main_method.sh              # Main method (3 seeds: 42, 123, 777)
├── 01_u01_solver_samples.sh           # Ablation: num_solver_samples {3, 5, 7}
├── 02_u02_solver_gamma.sh             # Ablation: solver_soft_gamma {0.5, 0.7, 0.9}
├── 03_u03_proposer_update_freq.sh     # Ablation: proposer_update_freq {1, 5, 10}
├── 04_u04_entropy_band.sh             # Ablation: entropy (mu, sigma) pairs
├── 05_u05_kl_sensitivity.sh           # Ablation: (kl_coef, kl_target) grid
├── 06_u06_lora_capacity.sh            # Ablation: lora_r {8, 16, 32}
├── 07_u07_frozen_proposer_proxy.sh    # Ablation: frozen proposer baseline
├── 90_run_all_understanding.sh        # Meta-script: runs 00-07
└── run_understanding_all_standalone.sh
```

### Generation Experiments

```
scripts/self_evolving/generation/
└── 00_g00_main_method.sh              # Main generation baseline
```

### Unified Experiments

```
scripts/self_evolving/unified/
├── 00_x00_main_method.sh              # Main unified method
└── 01_x01_dpo_style.sh                # Generator with DPO update rule
```

### Scripts Still Needed

| Experiment | Script to create |
|-----------|-----------------|
| A2.1–A2.4 Reward ablations | `unified/02_x02_reward_ablations.sh` |
| A2.5 Jaccard cycle consistency | `unified/03_x03_jaccard_cycle.sh` |
| A2.6 Frozen proposer | `unified/04_x04_frozen_proposer.sh` |
| A2.7 No synthetic bridge | `unified/05_x05_no_synthetic_bridge.sh` |
| A2.8 Single adapter | `unified/06_x06_single_adapter.sh` |
| A2.9 No KL | `unified/07_x07_no_kl.sh` |
| C3.8 Cycle ratios | `unified/08_x08_cycle_ratios.sh` |
| C3.9 Reward weights | `unified/09_x09_reward_weights.sh` |
| Evaluation scripts | `scripts/eval/` (VQAv2, GenEval, etc.) |

---

## 8. Logging and Data Collection

All data needed for Tier 4 analysis is already logged by the training pipeline.

### Per-Run Outputs

```
runs/<experiment_name>/
├── iter_log.jsonl               # Per-step metrics (rewards, losses, KL, baselines)
├── logs/
│   ├── proposer_prompts.jsonl   # Generated questions / specs
│   ├── generation_candidates.jsonl  # Generated image metadata
│   ├── rewards.jsonl            # Reward component breakdowns
│   ├── policy_updates.jsonl     # Loss, KL coef, advantage per update
│   └── dpo_pairs.jsonl          # (DPO only) chosen/rejected pairs
├── generated/                   # Saved generated images
├── ablation_summary.json        # Final run summary
└── checkpoints/
    ├── step_00500/
    ├── step_01000/
    └── ...
```

### W&B Integration

All metrics are logged to Weights & Biases when `--wandb_mode online`.
Key dashboard panels:

- `train/solver_reward_mean_soft` — understanding accuracy proxy
- `train/best_spec_score` — generation faithfulness
- `train/best_cycle_score` — cycle consistency
- `kl/solver_beta`, `kl/generator_beta` — adaptive KL health
- `train/best_diversity_score` — diversity monitor

---

## 9. Reproducibility Notes

### Fixed Seeds

All experiments use deterministic mode (`--deterministic True`) with
explicit seeds.  Multi-seed runs use {42, 123, 777} by convention.

### Configuration Snapshot

Every run saves a full configuration snapshot in `ablation_summary.json`
including all hyperparameters, model name, data paths, and git commit hash
(if available).

### Checkpoint Evaluation

Evaluation scripts should load checkpoints at regular intervals
(steps 500, 1000, 2500, 5000, 10000) to plot learning curves, not just
final performance.  This is critical for:

- Tier 4 training dynamics plots
- Self-judge bias audit (correlating internal vs external metrics over time)
- Identifying when mutual benefit between understanding and generation emerges
