# End-to-end Training Pipeline: Unsupervised Self-Evolving BLIP3o-NEXT

> **Document Version**: 1.0  
> **Target Model**: BLIP3o-NEXT (discrete image tokens + AR-as-policy + diffusion renderer)  
> **Training Style**: Fully unsupervised self-evolving loop (no annotated QA, no paired captions)

---

## 0) Scope and Assumptions

### Goal

Train a unified model that improves **both**:

1. **Image understanding** (VQA / reasoning / captioning)
2. **Image generation & editing** (prompt following, text rendering, consistency)

…using a **fully unsupervised self-evolving loop**, i.e., **no annotated QA**, no paired captions, no external reward model, no human labels—only raw images, plus the model's own internal consistency signals (EvoLMM-style).

### Key Architectural Choice

Use **BLIP3o-NEXT** as the backbone:

* **AR model** initialized from **Qwen3** and trained to predict **discrete image tokens** (and text tokens)
* **Diffusion model** (SANA1.5) refines the AR "blueprint" into final pixels via VAE latents

This matters because BLIP3o-NEXT's RL approach explicitly treats the AR output trajectory as a token policy (like an LLM), and decodes via a (frozen) diffusion model for reward evaluation.

### Training Style

Run **multiple interleaved phases**:

* **Phase U** (Understanding self-evolution) — EvoLMM propose→solve loop, but using BLIP3o-NEXT tokenization
* **Phase G** (Generation self-evolution) — propose→generate→verify loop where the reward is computed from internal cycle-consistency and solver verification

---

## 1) Definitions and Notation

### Scalars / Hyperparameters

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| B | Batch size (# anchor images per optimizer step) | 4-16 |
| N | # solver answer samples per question | 5 |
| G | # image samples per prompt (GRPO group size) | 4 |
| H, W | Diffusion output resolution | 512 or 1024 |
| H₃₈₄, W₃₈₄ | SigLIP2 quantization resolution | 384 |
| R | VAE downsampling factor (SANA DC-AE) | 32 |
| C_lat | Latent channels for SANA DC-AE | 32 |

### Shapes / Dtypes Conventions

```python
# Images
uint8_image: np.ndarray  # [H, W, 3], dtype=uint8
tensor_image: torch.Tensor  # [3, H, W], dtype=float32/bfloat16

# Token sequences
text_tokens: torch.Tensor  # [L_text], dtype=int64
image_tokens: torch.Tensor  # [729], dtype=int64

# Hidden states
hidden_states: torch.Tensor  # [L, D], dtype=bfloat16

# Masks
attention_mask: torch.Tensor  # [L], dtype=bool
```

---

## 2) Model Components and Their I/O Contracts

### 2.1 Image Tokenizer (SigLIP2 + Quantizer)

BLIP3o-NEXT encodes image → SigLIP2 embeddings → quantizes into discrete vocabulary.

```python
# Input
x_384: torch.Tensor  # [B, 3, 384, 384], float32/bfloat16

# Output
img_tok: torch.Tensor  # [B, 729], int64 (fixed length)
img_emb: torch.Tensor  # [B, 729, D_siglip], bfloat16 (optional)
```

**Vocabulary**: `V_img = 65536` image token classes

### 2.2 Text Tokenizer

```python
# Input
prompt: str

# Output
txt_tok: torch.Tensor  # [B, L_txt], int64
```

### 2.3 AR Backbone (Qwen3-based Multimodal Transformer)

Single shared base model with role-specific LoRA adapters.

**Config (BLIP3o-NEXT-Pretrain-3B)**:
- `hidden_size = 2048`
- `num_hidden_layers = 28`
- `num_attention_heads = 16`

#### Understanding Forward

```python
# Input
inp_ids: torch.Tensor  # [B, L_in], L_in = L_sys + 729 + L_txt
attn_mask: torch.Tensor  # [B, L_in]

# Output
logits: torch.Tensor  # [B, L_in, V_total]
hidden: torch.Tensor  # [B, L_in, D_model]
```

#### Generation Forward (Image Token Sampling)

```python
# Input
inp_ids: torch.Tensor  # [B, L_prompt]

# Output (sampled)
gen_img_tok: torch.Tensor  # [B, 729], int64
gen_hidden: torch.Tensor  # [B, 729, D_model]
```

### 2.4 Diffusion Renderer (SANA1.5 + DC-AE)

```python
# Noise latent
z_T: torch.Tensor  # [B, 32, H_lat, W_lat], bfloat16

# Condition tokens (projected from AR hidden)
cond: torch.Tensor  # [B, L_cond, 2304], bfloat16

# Output
z_0: torch.Tensor  # [B, 32, H_lat, W_lat], bfloat16
img_out: torch.Tensor  # [B, 3, H, W], float32
```

---

## 3) Data: Training on Raw Images Only

### 3.1 Raw Data Source

A pool of **raw images only** (no captions, no labels).

```python
# Data format
{
    "image_id": str,
    "jpg_bytes": bytes,
    "metadata": dict  # optional, ignored for training
}
```

### 3.2 Preprocessing Outputs (Cached)

```python
# Tokenizer view (384²)
x_384: torch.Tensor  # [3, 384, 384]
img_tok: torch.Tensor  # [729]

# Diffusion view (H×W)
x_hw: torch.Tensor  # [3, H, W]
vae_lat: torch.Tensor  # [32, H/32, W/32] (optional cache)
```

---

## 4) Roles and Adapters (EvoLMM-style)

### Three Role Policies

1. **Proposer π_prop**: Generates questions (U) or prompts + verification specs (G)
2. **Solver π_sol**: Answers questions, verifies generated images
3. **Generator π_gen**: Generates 729 discrete image tokens

### Implementation

```python
# Base transformer (frozen initially)
θ_base: torch.nn.Module

# LoRA adapters
ϕ_prop: LoRAConfig  # Proposer adapter
ϕ_sol: LoRAConfig   # Solver adapter
ϕ_gen: LoRAConfig   # Generator adapter
```

### LoRA Tensor Shapes

For linear W ∈ ℝ^{d_out × d_in}:
- A ∈ ℝ^{r × d_in}
- B ∈ ℝ^{d_out × r}
- ΔW = (α/r) · (B @ A)

Typical: r = 16, α = 32

---

## 5) Training Pipeline Overview

### Episode Types

**U-Episode (Understanding self-play):**
```
raw image x
  → proposer(question q | x)
  → solver(answers y1..yN | x, q)
  → compute self-consistency reward
  → RL update proposer/solver
```

**G-Episode (Generation self-play):**
```
raw image x (anchor)
  → proposer(modified prompt p + verifications V | x)
  → generator(image tokens z1..zG | p)  [AR sampling]
  → diffusion decode each zg → image Ig (no grad)
  → solver(caption + answers to V | Ig)
  → compute internal generation reward rg
  → RL update generator (and optionally proposer)
```

### Alternating Schedule

```python
# For every K_U understanding steps
for _ in range(K_U):
    run_u_episode()

# For every K_G generation steps
for _ in range(K_G):
    run_g_episode()
```

Recommended: K_U = 3, K_G = 1

---

## 6) Phase U: Understanding Self-Evolution

### 6.1 Inputs

```python
# Per batch element
jpg_bytes: bytes
x_uint8: np.ndarray  # [H0, W0, 3]
x_384: torch.Tensor  # [3, 384, 384]
img_tok: torch.Tensor  # [729]

# Batch
x_384: torch.Tensor  # [B, 3, 384, 384]
img_tok: torch.Tensor  # [B, 729]
```

### 6.2 Proposer Forward

```python
# Input
inp_prop_ids: torch.Tensor  # [B, L_prop_in]
attn_mask: torch.Tensor  # [B, L_prop_in]

# Output
q_tok: torch.Tensor  # [B, T_q], int64
q_text: List[str]
logp_prop: torch.Tensor  # [B, T_q], float32
kl_prop: torch.Tensor  # [B, T_q], float32
```

### 6.3 Solver Forward (N samples)

```python
# For each sample j ∈ {1..N}
a_tok_pad: torch.Tensor  # [B, N, T_a_max]
logp_sol_pad: torch.Tensor  # [B, N, T_a_max]
kl_sol_pad: torch.Tensor  # [B, N, T_a_max]
```

### 6.4 EvoLMM Rewards

```python
# Compute empirical distribution
K_i: int  # unique answer count
p_i: torch.Tensor  # [K_i], probabilities

# Entropy
H_i = -sum(p_i * log(p_i))  # float32

# Rewards
r_sol: torch.Tensor  # [B, N], float32 (per answer sample)
r_prop: torch.Tensor  # [B], float32 (entropy-band reward)
```

### 6.5 RL Losses

```python
# Advantage
adv = reward - baseline

# REINFORCE loss
L_pg = -mean(adv * logπ(y_t | h_t))

# KL penalty
L_kl = β * mean(KL(π || π_ref))

# Total
L = L_pg + L_kl
```

### 6.6 Optimizer Step

- Backprop ∇ϕ_sol every step
- Backprop ∇ϕ_prop every 5 steps (EvoLMM schedule)

---

## 7) Phase G: Generation Self-Evolution

### 7.1 Why BLIP3o-NEXT Makes This Feasible

Discrete image tokens make RL structurally identical to LLM RL:
- Sample 729-token trajectories
- Decode with frozen diffusion
- Score images
- Apply GRPO/REINFORCE on token logprobs

### 7.2 G-Episode Inputs

```python
# Anchor image
x: np.ndarray  # raw, unlabeled
img_tok: torch.Tensor  # [729]

# Optional for editing
vae_lat: torch.Tensor  # [32, H_lat, W_lat]
```

### 7.3 Proposer: Prompt + Verification Plan

```python
# Output fields
p_text: str  # Modified prompt describing variation
V: List[str]  # Verification questions
# e.g., ["Is it nighttime?", "Is the bird red?"]

# Tensor outputs
p_tok: torch.Tensor  # [B, T_p]
logp_prop_gen: torch.Tensor  # [B, T_p]
kl_prop_gen: torch.Tensor  # [B, T_p]
```

### 7.4 Generator: G Image-Token Trajectories

```python
# Generator outputs (sampling)
z_tok: torch.Tensor  # [B, G, 729]
logp_gen: torch.Tensor  # [B, G, 729]
kl_gen: torch.Tensor  # [B, G, 729]
```

### 7.5 Diffusion Decoding (No Gradients)

```python
# Per sample (i, g)
h_img: torch.Tensor  # [729, D_model]
cond: torch.Tensor  # [729, 2304]

# Diffusion
z_T: torch.Tensor  # [32, H_lat, W_lat]
z_0: torch.Tensor  # [32, H_lat, W_lat]
I_uint8: np.ndarray  # [H, W, 3]
```

### 7.6 Solver as Verifier

```python
# For each generated image I[i,g]
I_tok: torch.Tensor  # [729] (re-quantized)
c_tok: torch.Tensor  # caption tokens
v_ans: List[str]  # verification answers
```

### 7.7 Internal Reward Design

#### (A) Text-Image Cycle Consistency

```python
# Compare prompt with caption
r_cycle = similarity(p_text, c_text)
# Or: log P(prompt | image, caption) under solver
```

#### (B) Verification Success

```python
r_verif = mean([score(v_ans[k]) for k in V])
```

#### (C) Multi-Sample Agreement

```python
# Agreement on core attributes across group
r_agree = attribute_consistency(I[i, 1..G])
```

#### (D) Diversity Regularizer

```python
# Prevent mode collapse
r_diversity = embedding_variance(I[i, 1..G])
```

#### Total Reward

```python
r[i,g] = wA*r_cycle + wB*r_verif + wC*r_agree + wD*r_diversity
```

### 7.8 GRPO Update

```python
# Advantage (group-relative)
A[i,g] = r[i,g] - mean(r[i,:])

# Policy gradient loss
L_gen_pg = -mean(A[i,g] * mean_t(logπ_gen(z_tok[i,g,t])))

# KL penalty
L_gen_kl = β_gen * mean(KL(π_gen || π_ref))

# Total
L_gen = L_gen_pg + L_gen_kl
```

---

## 8) Image Editing Self-Evolution (Optional)

### Edit-Based Self-Play Loop

1. Sample raw image x
2. Proposer generates edit instruction p (e.g., "change the bird to red")
3. Generator produces edited image I'
4. Solver verifies:
   - "Is the bird red?" (edit success)
   - "Is the background unchanged?" (identity preservation)
5. Reward combines edit success + preservation score

---

## 9) Full Training Loop: Step-by-Step

### 9.1 Common: Sample and Preprocess

```python
# Input
batch_image_bytes: List[bytes]  # length B

# Decode
x_uint8: List[np.ndarray]  # [H0, W0, 3]

# Create 384 view
x_384: torch.Tensor  # [B, 3, 384, 384]
img_tok: torch.Tensor  # [B, 729]

# Create diffusion view
x_hw: torch.Tensor  # [B, 3, H, W]
vae_lat: torch.Tensor  # [B, 32, H_lat, W_lat]
```

### 9.2 U-Step: Understanding Self-Play

```python
# (1) Proposer
inp_prop_ids: [B, L_prop_in]
q_tok: [B, T_q]
logp_prop: [B, T_q]
kl_prop: [B, T_q]

# (2) Solver samples
a_tok_pad: [B, N, T_a_max]
logp_sol_pad: [B, N, T_a_max]
kl_sol_pad: [B, N, T_a_max]

# (3) Rewards
r_sol: [B, N]
r_prop: [B]
H_entropy: [B]

# (4) Losses
L_sol: scalar
L_prop: scalar

# (5) Optimizer
backprop ϕ_sol (every step)
backprop ϕ_prop (every 5 steps)
```

### 9.3 G-Step: Generation Self-Play

```python
# (1) Proposer: prompt + verification
p_tok: [B, T_p]
V: List[List[str]]

# (2) Generator: sample trajectories
z_tok: [B, G, 729]
logp_gen: [B, G, 729]
kl_gen: [B, G, 729]

# (3) Diffusion decode (no grad)
cond: [729, 2304]
z_T: [32, H_lat, W_lat]
I_uint8: [H, W, 3]

# (4) Solver verification
I_tok: [729]
c_tok: variable
v_ans: List[str]

# (5) Reward
r: [B, G]
A: [B, G]

# (6) GRPO update
L_gen_pg: scalar
L_gen_kl: scalar
optimizer step ϕ_gen
```

---

## 10) Logging, Checkpoints, and Evaluation

### 10.1 Logged Artifacts

**U-step:**
- image_id, q_text, answers, entropy H
- r_sol stats, KL stats, β stats

**G-step:**
- prompt p, verification V
- rewards per sample r[i,g]
- top-1/bottom-1 images per prompt
- solver captions + verification answers

### 10.2 Checkpoints

```python
{
    "base_model": str,  # frozen weights path
    "lora_proposer": dict,
    "lora_solver": dict,
    "lora_generator": dict,
    "optimizer_state": dict,
    "kl_controller": {"β": float, "baselines": dict},
}
```

### 10.3 Evaluation (Offline)

- **Understanding**: ChartQA, MathVista (EvoLMM suite)
- **Generation**: GenEval, prompt-following tests

---

## Appendix A: BLIP3-o (Continuous Features)

BLIP3-o uses continuous CLIP features (not discrete tokens):
- 64 vectors per image
- Sequential training (understanding → generation)
- Harder for token-level RL

**Not recommended for this pipeline.**

---

## Appendix B: Minimal MVP Pipeline

1. **Port EvoLMM Phase U** onto BLIP3o-NEXT tokenization
2. **Add Phase G** with editing-based self-play:
   - anchor image x
   - proposer: "change attribute A"
   - generator: I'
   - solver: verify "A changed" + "rest unchanged"
3. Apply GRPO on **generator image-token head first** (freeze else)
