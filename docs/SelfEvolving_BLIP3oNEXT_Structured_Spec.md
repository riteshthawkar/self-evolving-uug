# Structured Spec: Fully Unsupervised Self‑Evolving Training for BLIP3o‑NEXT (Understanding + Generation)

**Version:** 1.0 (implementation-oriented)  
**Date:** 2026‑01‑30  
**Primary target model:** BLIP3o‑NEXT (Autoregressive discrete image tokens + Diffusion renderer) fileciteturn0file1  
**Self‑evolving framework basis:** EvoLMM propose–solve with continuous internal rewards fileciteturn0file0  
**Project feasibility context:** internal feasibility report fileciteturn0file3  

---

## 0. What this spec is (and what it is not)

This document is a **single end‑to‑end engineering spec** that you can implement directly. It defines:

- **Inputs / outputs** for every module
- **Dataflow** through the system
- **Episode formats** (JSON schemas) for both training and logging
- **Prompt templates** for proposer/solver/generator
- **Reward definitions** (including EvoLMM continuous rewards and new generation rewards)
- **Training objectives** (REINFORCE + KL, GRPO-style group advantages)
- **Exact intermediate tensor shapes + dtypes** (parameterized with defaults)
- **Reference configuration** (YAML-like) and **pseudocode** (PyTorch-style)

It does **not** prescribe your exact dataset choice, nor does it assume proprietary captioners or external reward models. Training uses **raw images only** plus **the model itself** to create tasks and rewards, consistent with EvoLMM’s constraint of no annotations and no external reward models. fileciteturn0file0

---

## 1. High-level system architecture

### 1.1 Modules (runtime graph)

**Core modules**:
1) **Image Loader / Preprocessor**  
2) **Image Tokenizer** (SigLIP2 encoder + quantizer, per BLIP3o‑NEXT) fileciteturn0file1  
3) **Text Tokenizer**
4) **Shared AR backbone** (single transformer with role adapters)
5) **Roles (policies)**:
   - **Proposer** π_prop: generates questions (U) and prompts+verifiers (G)
   - **Solver/Judge** π_sol: answers questions and verifies generated images
   - **Generator** π_gen: generates **discrete image tokens** (729 tokens) fileciteturn0file1
6) **Diffusion Renderer** (frozen initially): decodes AR conditions → pixels fileciteturn0file1  

**Episode engines**:
- **U‑Episode engine** (Understanding self‑evolution): propose → solve (N samples) → internal reward → RL update fileciteturn0file0  
- **G‑Episode engine** (Generation self‑evolution): propose → generate (G rollouts) → render → verify → internal reward → GRPO/RL update fileciteturn0file1  

### 1.2 Design rationale (why BLIP3o‑NEXT)
BLIP3o‑NEXT produces **discrete visual tokens** and explicitly supports RL on the AR policy using GRPO-like group rollouts, while keeping diffusion frozen during RL. fileciteturn0file1  
This matches the token-policy view required for stable, scalable self-evolving RL.

---

## 2. Notation, constants, and tensor conventions

### 2.1 Global constants / defaults (parameterized)
You must treat these as config values; defaults below match BLIP3o‑NEXT paper claims where possible.

| Symbol | Meaning | Default |
|---|---|---|
| `B` | batch size (anchor images per step) | 1–8 (start with 1) |
| `N` | solver samples per question (U‑episodes) | 5 (EvoLMM default) fileciteturn0file0 |
| `G` | generator rollouts per prompt (G‑episodes) | 2–4 (start with 2) |
| `Htok` | image token length | 729 fileciteturn0file1 |
| `Vimg` | image token vocab size | 65536 (from BLIP3o‑NEXT release configs; make configurable) |
| `Vtxt` | text vocab size | depends on tokenizer |
| `D` | AR hidden size | model-specific (e.g., 2k–4k) |
| `Dcond` | diffusion conditioning channel dim | model-specific (often 2304 if using SANA-style cond) |
| `H` | generation resolution | 512 or 1024 |
| `R` | VAE downsample factor | 32 (SANA DC‑AE typical) |
| `Hlat` | latent height | `H/R` |
| `Wlat` | latent width | `W/R` |

### 2.2 Shapes and dtypes (global conventions)
- Images:
  - `img_uint8`: `[H0, W0, 3]`, dtype `uint8`
  - `img_f32`: `[3, H, W]`, dtype `float32`
- Token ids:
  - `txt_ids`: `[L]`, dtype `int32/int64`
  - `img_ids`: `[Htok]`, dtype `int32/int64`
- Transformer hidden states:
  - `h`: `[L, D]`, dtype `bfloat16` (training)
- Token logprobs:
  - `logp`: `[L]`, dtype `float32`
- KL per token:
  - `kl`: `[L]`, dtype `float32`

---

## 3. Data inputs and preprocessing (raw images only)

### 3.1 Dataset interface
Each sample `s` is:
```json
{
  "image_id": "string",
  "image_bytes": "bytes"  // jpeg/png
}
```

### 3.2 Preprocessing pipeline (per image)
**Input:** `image_bytes`  
**Output cached fields (recommended):**
1) `img_tok_view`:
   - `img_384`: `[3, 384, 384]`, `float32`
   - `img_tok`: `[729]`, `int64`  (SigLIP2→quantize) fileciteturn0file1
2) `img_render_view` (optional cache for editing or reconstruction tasks):
   - `img_hw`: `[3, H, W]`, `float32`
   - `vae_lat`: `[C_lat, Hlat, Wlat]`, `bfloat16` (if using editing consistency; optional)

### 3.3 Image tokenization (BLIP3o‑NEXT)
BLIP3o‑NEXT: encode with SigLIP2; quantize into tokens; yields **729 tokens per 384×384 image**. fileciteturn0file1

**Contract:**
- `encode_quantize(img_384)` →  
  - `img_tok`: `[729]` int64  
  - optionally `img_embed`: `[729, Dsig]` bf16

---

## 4. Model roles and adapters

### 4.1 Shared backbone + role adapters
Following EvoLMM, implement roles using **separate adapters** (LoRA) on a single frozen base. fileciteturn0file0  

- Base parameters: `θ_base` (frozen initially)
- Proposer LoRA: `ϕ_prop`
- Solver LoRA: `ϕ_sol`
- Generator LoRA: `ϕ_gen`

**Key design choice:** keep a **frozen reference policy** for each role (`π_ref_prop`, `π_ref_sol`, `π_ref_gen`) for KL regularization (EvoLMM). fileciteturn0file0

### 4.2 Role behaviors
- **Proposer**:
  - U‑episodes: `q ~ π_prop(q | x)` (question conditioned on image)
  - G‑episodes: `(p, V) ~ π_prop(p, V | x)` (prompt and verification plan from image)
- **Solver**:
  - U‑episodes: `y ~ π_sol(y | x, q)` (answer)
  - G‑episodes: `caption ~ π_sol(c | I)` and `ans_k ~ π_sol(ans_k | I, v_k)` (verification)
- **Generator**:
  - G‑episodes: `img_tok_gen ~ π_gen(z | p)` (729 image tokens) fileciteturn0file1

---

## 5. Prompt templates (exact)

All prompts below are plain strings passed to the AR backbone, with images inserted as **image tokens** using your model’s multimodal formatting (e.g., `<image>` placeholder or special image-start token sequence).

### 5.1 Proposer system prompt (U‑episodes)
**SYSTEM_PROP_U**
```
You are the Proposer. Given an image, generate ONE visually grounded question that can be answered from the image alone.
Constraints:
- The question must be specific and unambiguous.
- The question must be answerable without external knowledge.
- Output format:
<question> ... </question>
```

**USER_PROP_U**
```
<image>
Generate one question.
```

**Expected output text**
```
<question> ... ? </question>
```

### 5.2 Solver system prompt (U‑episodes)
**SYSTEM_SOL_U**
```
You are the Solver. Answer the user's question using ONLY the image.
You must think step-by-step internally, but ONLY output the final answer.
Output format:
<answer> ... </answer>
```

**USER_SOL_U**
```
<image>
Question: {q_text}
```

**Expected output**
```
<answer> ... </answer>
```

### 5.3 Proposer system prompt (G‑episodes: prompt + verification plan)
The proposer must generate (1) a prompt for generation (often an edit/variation), and (2) a verification checklist `V` that will later be asked about the generated image.

**SYSTEM_PROP_G**
```
You are the Proposer for image generation.
Given an input image, create:
(1) a text prompt describing a NEW target image (a grounded variation/edit of the input), and
(2) a list of verification questions that can be answered by looking at the generated image.

Constraints:
- The prompt must be grounded in the input image content but should introduce a meaningful change.
- Verification questions must be specific, visual, and checkable.
- Avoid subjective questions ("Is it beautiful?").
- Output format EXACTLY:
<prompt> ... </prompt>
<verify>
- ...
- ...
</verify>
```

**USER_PROP_G**
```
<image>
Create a grounded prompt and verification checklist.
```

### 5.4 Generator system prompt (G‑episodes)
The generator uses the prompt from proposer and must output **image tokens** (special generation mode in your AR model).

**SYSTEM_GEN**
```
You are an image generation model.
Given a prompt, generate an image.
```

**USER_GEN**
```
Prompt: {p_text}
<image_generation>
```

**Expected output**
- discrete image token ids `z_tok`: shape `[729]` (no text)
- optionally: your implementation may also output a special `<image_end>` token.

### 5.5 Solver system prompt (G‑episodes: caption + verification)
**SYSTEM_SOL_G_CAPTION**
```
You are the Solver. Describe the image accurately and concisely.
Output:
<caption> ... </caption>
```

**USER_SOL_G_CAPTION**
```
<image>
Write a caption.
```

**SYSTEM_SOL_G_VERIFY**
```
You are the Solver. Answer the verification question using ONLY the image.
Output:
<answer> ... </answer>
```

**USER_SOL_G_VERIFY**
```
<image>
Verification question: {v_k}
```

---

## 6. Episode types and JSON schemas

All episode records must be serializable and stored for debugging, ablations, and replay analysis.

### 6.1 Common fields (all episodes)
```json
{
  "episode_id": "uuid",
  "timestamp": "iso-8601",
  "image_id": "string",
  "seed": 0,
  "model_versions": {
    "base": "string",
    "prop_adapter": "string",
    "sol_adapter": "string",
    "gen_adapter": "string",
    "ref_prop": "string",
    "ref_sol": "string",
    "ref_gen": "string"
  },
  "config_snapshot": { "any": "json" }
}
```

### 6.2 U‑Episode schema (Understanding self-play)
```json
{
  "type": "U",
  "image_tok": {
    "shape": [729],
    "dtype": "int64"
  },
  "proposer": {
    "question_text": "string",
    "question_tokens": { "shape": ["Tq"], "dtype": "int64" },
    "logp": { "shape": ["Tq"], "dtype": "float32" },
    "kl": { "shape": ["Tq"], "dtype": "float32" }
  },
  "solver": {
    "N": 5,
    "answers_text": ["string", "... N"],
    "answers_tokens": { "shape": [5, "Ta_max"], "dtype": "int64" },
    "logp": { "shape": [5, "Ta_max"], "dtype": "float32" },
    "kl": { "shape": [5, "Ta_max"], "dtype": "float32" }
  },
  "metrics": {
    "entropy_H": "float32",
    "reward_prop": "float32",
    "reward_sol": { "shape": [5], "dtype": "float32" }
  }
}
```

### 6.3 G‑Episode schema (Generation self-play)
```json
{
  "type": "G",
  "anchor_image_tok": { "shape": [729], "dtype": "int64" },
  "proposer": {
    "prompt_text": "string",
    "verify_list": ["string", "... K"],
    "prompt_tokens": { "shape": ["Tp"], "dtype": "int64" },
    "logp": { "shape": ["Tp"], "dtype": "float32" },
    "kl": { "shape": ["Tp"], "dtype": "float32" }
  },
  "generator": {
    "G": 2,
    "image_token_rollouts": { "shape": [2, 729], "dtype": "int64" },
    "logp": { "shape": [2, 729], "dtype": "float32" },
    "kl": { "shape": [2, 729], "dtype": "float32" }
  },
  "renderer": {
    "resolution": [H, W],
    "images_uint8": { "shape": [2, H, W, 3], "dtype": "uint8" },
    "diffusion_steps": "int"
  },
  "solver_verify": {
    "captions_text": { "shape": [2], "dtype": "string" },
    "verify_answers": [
      {
        "question": "string",
        "answers_text": { "shape": [2], "dtype": "string" }
      }
    ]
  },
  "metrics": {
    "reward_components": {
      "cycle": { "shape": [2], "dtype": "float32" },
      "verify": { "shape": [2], "dtype": "float32" },
      "agree": { "shape": [2], "dtype": "float32" },
      "diversity": "float32"
    },
    "reward_total": { "shape": [2], "dtype": "float32" },
    "advantage_group": { "shape": [2], "dtype": "float32" }
  }
}
```

---

## 7. Rewards (exact definitions)

### 7.1 EvoLMM continuous rewards (U‑episodes)
EvoLMM reward design: solver reward is continuous self-consistency; proposer reward is entropy-band Gaussian. fileciteturn0file0

#### 7.1.1 Empirical answer distribution
From N answers `y1..yN`, compute canonicalized strings `a1..aN` and histogram counts.

- `p(a|x,q) = count(a)/N`
- majority answer `â = argmax_a p(a|x,q)`
- entropy (nats):
  - `H(x,q) = - Σ_a p(a|x,q) log p(a|x,q)`  fileciteturn0file0

#### 7.1.2 Solver reward (per sample i)
EvoLMM defines a continuous reward scaling with agreement and length penalty. fileciteturn0file0  

Let `p_i = p(y_i | x,q)` be the empirical agreement probability of the specific answer string produced in sample i.

Let:
- `γ ∈ (0,1]` softness exponent (default 0.7 in EvoLMM training details) fileciteturn0file0  
- `w_i` = # words before `<answer>` tag (or before answer span)
- `τ` = target brevity threshold (default 6 words)
- `λ_len` = length penalty weight (default 0.10)

Then:
- `r_sol_i = (p_i)^γ * (1 - λ_len * max(0, (w_i - τ)/τ))`

Outputs:
- `r_sol`: `[N]` float32

#### 7.1.3 Proposer reward (per question)
EvoLMM uses a band-pass reward that peaks at moderate entropy. fileciteturn0file0  

Hyperparams:
- `μ_H` (default 0.90)
- `σ_H` (default 0.35)

Reward:
- `r_prop = exp( - (H - μ_H)^2 / (2 σ_H^2) )`

### 7.2 Generation rewards (G‑episodes) — fully internal
Goal: reward prompt-following and verifiable correctness without external reward models. This is the core “new” part your feasibility report discusses. fileciteturn0file3  

We define total reward per rollout `g`:
- `r_total[g] = w_cycle*r_cycle[g] + w_verify*r_verify[g] + w_agree*r_agree[g] + w_div*r_div`

All weights are config-controlled.

#### 7.2.1 Cycle reward: prompt ↔ caption consistency (internal)
Given prompt `p_text` and solver caption `c_text[g]` on generated image `I[g]`.

**Recommended implementation (model-internal likelihood)**:
- Query solver to score the prompt given the image and caption:
  - `s[g] = log P_sol( p_text | I[g], c_text[g] )`
- Normalize within batch/group:
  - `r_cycle[g] = clamp( (s[g] - s_min) / (s_max - s_min + ε), 0, 1 )`

**Fallback implementation (string-based)**:
- Extract key attributes from both strings (color words, counts, object nouns) using a small internal parser.
- `r_cycle[g] = F1(keywords(p), keywords(c))`

#### 7.2.2 Verification reward: checklist success (internal)
Given verification list `V = {v_k}` from proposer.

For each rollout g and each v_k:
- solver answers: `ans[g,k]`

Map each (v_k, ans) to a score in [0,1]:
- For boolean questions, accept `{"yes","no"}` canonical forms
- For numeric questions, parse integers
- For color/object presence, use a restricted answer set

Define:
- `r_verify[g] = mean_k score(ans[g,k])`

#### 7.2.3 Agreement reward: consistency across rollouts (stability)
Compute agreement of captions across the group:
- Let `E_caption[g]` be an internal embedding from solver (e.g., pooled last hidden state).
- Let `sim(g,g') = cosine(E_caption[g], E_caption[g'])`

Reward should prefer *semantic* agreement without forcing identical pixels:
- `r_agree[g] = mean_{g'≠g} sim(g,g')`

(You can also compute agreement across verification answers.)

#### 7.2.4 Diversity reward: anti-collapse across rollouts (group-level)
Diversity is group-level and applied equally to all rollouts:

- Compute embeddings `E_img[g]` (from SigLIP2 encoder or solver pooled embedding).
- Compute average pairwise distance:
  - `div = mean_{g<g'} (1 - cosine(E_img[g], E_img[g']))`
- Define:
  - `r_div = clamp(div / div_target, 0, 1)`

**Important:** you do *not* want r_div to dominate; it is only a collapse-preventer.

---

## 8. Training objectives (exact)

### 8.1 KL-regularized REINFORCE (EvoLMM-style)
EvoLMM trains proposer and solver with REINFORCE + token-level KL to a frozen reference and adaptive β. fileciteturn0file0  

For role A ∈ {prop, sol} with sampled tokens `y_1..y_T`:

- Policy gradient loss:
  - `L_pg = - (r - b) * (1/T) * Σ_t log π_A(y_t | h_t)`
- KL penalty:
  - `KL = (1/T) * Σ_t KL( π_A(·|h_t) || π_ref_A(·|h_t) )`
- Total:
  - `L_A = L_pg + β_A * KL`

Baselines:
- `b_A` = exponential moving average of rewards

Adaptive KL controller (EvoLMM):
- `β_A ← clip( β_A * exp( η * (KL - τ_A) / τ_A ), β_min, β_max )` fileciteturn0file0

### 8.2 Generator optimization (G‑episodes): GRPO‑style group relative advantages
BLIP3o‑NEXT applies GRPO to AR model by sampling G trajectories (729 tokens), decoding via frozen diffusion, scoring with rewards, and optimizing AR policy with group-normalized advantages and KL to reference. fileciteturn0file1  

We will use a simplified GRPO/REINFORCE variant:

For each anchor prompt p, rollouts `o_1..o_G` (each `o_g` is 729 tokens):
- rewards `r[g]`

Group-relative advantage:
- `A[g] = r[g] - mean_g r[g]`  (or standardized by std)

Generator token loss:
- `L_gen_pg = - mean_g ( A[g] * (1/Htok) * Σ_t log π_gen(o_g[t] | h_t) )`

Generator KL:
- `KL_gen = mean_{g,t} KL(π_gen || π_ref_gen)`

Total:
- `L_gen = L_gen_pg + β_gen * KL_gen`

**Diffusion renderer stays frozen** for RL stability initially (matches BLIP3o‑NEXT RL pipeline). fileciteturn0file1

---

## 9. Training schedule (phases + alternating)

This spec supports two safe schedules:

### 9.1 Schedule S1 (recommended MVP): alternating U and G with frozen diffusion
- Steps:
  - do `KU` U‑steps (e.g., KU=10)
  - do `KG` G‑steps (e.g., KG=1)
- Keep diffusion renderer frozen throughout MVP.

### 9.2 Schedule S2 (later): staged + alternating
Motivated by unified training concerns and sequential strategies observed in BLIP3‑o, staged training helps preserve understanding. fileciteturn0file2  

- Stage 0: Load pretrained BLIP3o‑NEXT weights (understanding+generation baseline).
- Stage 1: U‑only self-evolution (warm start proposer/solver).
- Stage 2: G‑only generator RL (freeze proposer/solver or keep slow).
- Stage 3: alternating U+G for co-evolution.

---

## 10. Detailed step-by-step dataflow (with tensors)

### 10.1 U‑step (Understanding episode) — tensors

**Inputs**
- `img_tok`: `[B, 729]` int64

**Proposer**
- input ids `inp_prop`: `[B, L_prop]` int64
- sampled question tokens `q_tok`: `[B, Tq]` int64
- `logp_prop`: `[B, Tq]` float32
- `kl_prop`: `[B, Tq]` float32

**Solver samples (N)**
- `a_tok`: `[B, N, Ta_max]` int64
- `logp_sol`: `[B, N, Ta_max]` float32
- `kl_sol`: `[B, N, Ta_max]` float32

**Reward compute**
- `H_entropy`: `[B]` float32
- `r_prop`: `[B]` float32
- `r_sol`: `[B, N]` float32

**Losses**
- `L_prop`: scalar
- `L_sol`: scalar

**Gradients**
- update `ϕ_sol` every step
- update `ϕ_prop` every 5 steps (EvoLMM schedule) fileciteturn0file0

### 10.2 G‑step (Generation episode) — tensors

**Inputs**
- anchor `img_tok_anchor`: `[B, 729]` int64 (for proposer grounding only)

**Proposer outputs**
- prompt tokens `p_tok`: `[B, Tp]` int64
- verify list: python list of strings

**Generator rollouts**
- `z_tok`: `[B, G, 729]` int64
- `logp_gen`: `[B, G, 729]` float32
- `kl_gen`: `[B, G, 729]` float32

**Render (no grad)**
- `I_uint8`: `[B, G, H, W, 3]` uint8

**Solver verification**
- `caption_text`: list length `B*G`
- `verify_answers`: structured list

**Reward**
- `r_total`: `[B, G]` float32
- `A_group`: `[B, G]` float32

**Loss**
- `L_gen`: scalar

**Gradients**
- update `ϕ_gen` (and optionally `ϕ_prop` with a delayed schedule)

---

## 11. Implementation pseudocode (PyTorch-style)

> This is “runnable skeleton pseudocode”: names match the spec contracts.

```python
class SelfEvolvingTrainer:
    def __init__(self, models, tokenizers, renderer, cfg):
        self.base = models.base            # frozen base transformer weights
        self.prop = models.prop_adapter    # LoRA
        self.sol  = models.sol_adapter     # LoRA
        self.gen  = models.gen_adapter     # LoRA
        self.ref_prop = models.ref_prop    # frozen
        self.ref_sol  = models.ref_sol     # frozen
        self.ref_gen  = models.ref_gen     # frozen

        self.img_tokenizer = tokenizers.img
        self.txt_tokenizer = tokenizers.txt
        self.renderer = renderer           # diffusion renderer (frozen initially)

        self.beta_prop = cfg.kl.beta_prop
        self.beta_sol  = cfg.kl.beta_sol
        self.beta_gen  = cfg.kl.beta_gen
        self.baseline_prop = 0.0
        self.baseline_sol  = 0.0
        self.baseline_gen  = 0.0

    # ----------------------------
    # U-EPISODE (UNDERSTANDING)
    # ----------------------------
    def u_step(self, batch_images):
        # Preprocess -> image tokens
        img_tok = self.img_tokenizer.quantize(batch_images, size=384)  # [B,729] int64

        # Proposer: sample question
        q_tok, logp_prop, kl_prop = sample_text(
            policy=self.prop, ref=self.ref_prop,
            input_image_tokens=img_tok,
            prompt_template=SYSTEM_PROP_U + USER_PROP_U
        )

        q_text = detokenize(q_tok)

        # Solver: sample N answers
        answers = []
        logp_sol_list, kl_sol_list = [], []
        for n in range(cfg.u.N):
            a_tok, logp_sol, kl_sol = sample_text(
                policy=self.sol, ref=self.ref_sol,
                input_image_tokens=img_tok,
                prompt_template=SYSTEM_SOL_U + USER_SOL_U.format(q_text=q_text)
            )
            answers.append(detokenize(a_tok))
            logp_sol_list.append(logp_sol)
            kl_sol_list.append(kl_sol)

        # Compute rewards (EvoLMM)
        H_entropy, r_prop, r_sol = compute_evolmm_rewards(
            answers=answers,
            gamma=cfg.u.gamma,
            lambda_len=cfg.u.lambda_len,
            tau_words=cfg.u.tau_words,
            mu_H=cfg.u.mu_H,
            sigma_H=cfg.u.sigma_H
        )

        # Losses
        L_prop = reinforce_with_kl(
            logp=logp_prop,
            reward=r_prop,
            baseline=self.baseline_prop,
            kl=kl_prop,
            beta=self.beta_prop
        )

        # solver reward is per-sample; average over N
        L_sol = 0.0
        for n in range(cfg.u.N):
            L_sol += reinforce_with_kl(
                logp=logp_sol_list[n],
                reward=r_sol[n],
                baseline=self.baseline_sol,
                kl=kl_sol_list[n],
                beta=self.beta_sol
            )
        L_sol /= cfg.u.N

        # Backprop
        (L_sol + cfg.u.prop_loss_weight*L_prop).backward()
        clip_grad_norm_(self.sol.parameters(), cfg.opt.grad_clip)
        clip_grad_norm_(self.prop.parameters(), cfg.opt.grad_clip)
        self.opt_sol.step(); self.opt_sol.zero_grad(set_to_none=True)
        if self.global_step % cfg.u.prop_update_every == 0:
            self.opt_prop.step(); self.opt_prop.zero_grad(set_to_none=True)

        # Update baselines and betas (adaptive KL controller)
        self.baseline_sol  = ema(self.baseline_sol,  r_sol.mean().item(), cfg.ema)
        self.baseline_prop = ema(self.baseline_prop, r_prop.item(),      cfg.ema)
        self.beta_sol  = update_beta(self.beta_sol,  kl_sol_mean(kl_sol_list), cfg.kl)
        self.beta_prop = update_beta(self.beta_prop, kl_prop.mean().item(),    cfg.kl)

        return metrics_dict(...)

    # ----------------------------
    # G-EPISODE (GENERATION)
    # ----------------------------
    def g_step(self, batch_images):
        img_tok_anchor = self.img_tokenizer.quantize(batch_images, size=384)  # [B,729]

        # Proposer: prompt + verification list
        p_tok, logp_prop, kl_prop, verify_list = sample_prompt_and_verifiers(
            policy=self.prop, ref=self.ref_prop,
            input_image_tokens=img_tok_anchor,
            template=SYSTEM_PROP_G + USER_PROP_G
        )
        p_text = detokenize(p_tok)

        # Generator: sample G rollouts of 729 image tokens (discrete policy)
        z_tok, logp_gen, kl_gen = sample_image_tokens_group(
            policy=self.gen, ref=self.ref_gen,
            prompt_text=p_text,
            num_rollouts=cfg.g.G, image_token_len=729
        )  # z_tok: [B,G,729]

        # Render each rollout (no grad; environment step)
        with torch.no_grad():
            imgs_uint8 = self.renderer.decode(z_tok, prompt_text=p_text, resolution=cfg.render.res)  # [B,G,H,W,3]

        # Solver: caption + verify for each rendered image
        captions = []
        verify_answers = []
        for g in range(cfg.g.G):
            I_g = imgs_uint8[:, g]  # [B,H,W,3]
            I_tok = self.img_tokenizer.quantize(I_g, size=384)  # [B,729]

            c_tok = sample_text(self.sol, self.ref_sol, I_tok, SYSTEM_SOL_G_CAPTION + USER_SOL_G_CAPTION)
            c_text = detokenize(c_tok); captions.append(c_text)

            ans_k = []
            for v in verify_list:
                a_tok = sample_text(self.sol, self.ref_sol, I_tok, SYSTEM_SOL_G_VERIFY + USER_SOL_G_VERIFY.format(v_k=v))
                ans_k.append(detokenize(a_tok))
            verify_answers.append(ans_k)

        # Compute generation rewards (internal)
        r_total, r_parts = compute_generation_rewards(
            prompt=p_text,
            captions=captions,
            verify_list=verify_list,
            verify_answers=verify_answers,
            weights=cfg.g.reward_weights
        )  # r_total: [B,G]

        # Group advantages (GRPO-style)
        A = r_total - r_total.mean(dim=1, keepdim=True)  # [B,G]

        # Policy gradient loss on generator tokens
        L_gen = grpo_like_loss(
            logp=logp_gen, A=A,
            kl=kl_gen, beta=self.beta_gen,
            clip_eps=cfg.g.clip_eps
        )

        # Backprop only generator adapter
        L_gen.backward()
        clip_grad_norm_(self.gen.parameters(), cfg.opt.grad_clip)
        self.opt_gen.step(); self.opt_gen.zero_grad(set_to_none=True)

        # Update beta_gen, baseline_gen
        self.baseline_gen = ema(self.baseline_gen, r_total.mean().item(), cfg.ema)
        self.beta_gen = update_beta(self.beta_gen, kl_gen.mean().item(), cfg.kl)

        return metrics_dict(...)
```

---

## 12. Reference configuration (YAML-like)

```yaml
model:
  name: blip3o-next
  adapters:
    proposer: lora_rank: 16
    solver:   lora_rank: 16
    gen:      lora_rank: 16
  freeze_base: true
  freeze_renderer: true   # diffusion

data:
  source: raw_images_only
  image_size_tokenizer: 384
  cache:
    enable: true
    store_img_tok: true

u:  # understanding
  N: 5
  gamma: 0.7
  lambda_len: 0.10
  tau_words: 6
  mu_H: 0.90
  sigma_H: 0.35
  prop_update_every: 5
  prop_loss_weight: 1.0

g:  # generation
  G: 2
  clip_eps: 0.2
  reward_weights:
    cycle: 0.35
    verify: 0.45
    agree: 0.10
    diversity: 0.10
  diversity_target: 0.25

render:
  res: [512, 512]
  diffusion_steps: 20

kl:
  beta_prop: 0.01
  beta_sol:  0.01
  beta_gen:  0.01
  target_kl_prop: 0.03
  target_kl_sol:  0.03
  target_kl_gen:  0.03
  eta: 0.1
  beta_min: 1e-4
  beta_max: 1.0

opt:
  type: adamw
  lr_prop: 1e-6
  lr_sol:  1e-6
  lr_gen:  1e-6
  wd: 0.01
  grad_clip: 1.0

schedule:
  KU: 10   # U steps per cycle
  KG: 1    # G steps per cycle
  total_steps: 6000
```

Defaults for `u.*` match EvoLMM training details (N=5, γ=0.7, λ_len=0.10, τ=6, μ=0.90, σ=0.35, proposer update every 5). fileciteturn0file0

---

## 13. Logging and checkpoint spec

### 13.1 Logging
Log every step:
- `losses`: `L_sol, L_prop, L_gen, KL_*`
- `rewards`: means and histograms for each reward component
- `collapse monitors`: diversity, caption entropy, per-token KL
- `samples`:
  - U: image + question + N answers
  - G: prompt + verify + top/bottom image rollouts + captions + answers

### 13.2 Checkpoints
Checkpoint object:
```python
{
  "step": int,
  "adapters": {
    "prop": state_dict(ϕ_prop),
    "sol":  state_dict(ϕ_sol),
    "gen":  state_dict(ϕ_gen),
  },
  "optim": {
    "prop": opt_state,
    "sol":  opt_state,
    "gen":  opt_state
  },
  "kl_controller": {
    "beta_prop": float,
    "beta_sol": float,
    "beta_gen": float,
    "baseline_prop": float,
    "baseline_sol": float,
    "baseline_gen": float
  },
  "rng": {...}
}
```

---

## 14. Evaluation spec (offline)

Even if training is unsupervised, evaluation can be standard.

### 14.1 Understanding eval
- Use held-out benchmark images + prompts (ChartQA/MathVista-style) if available in your environment.
- Primary metric: accuracy.
EvoLMM shows improvements on math reasoning benchmarks using raw images only. fileciteturn0file0  

### 14.2 Generation eval
- Prompt-following: GenEval-like object composition and text rendering (offline evaluator).
BLIP3o‑NEXT uses verifiable reward settings for RL improvements and reports GenEval gains. fileciteturn0file1  
(You do not need GenEval as a training reward; just evaluation.)

### 14.3 Editing eval (optional)
BLIP3o‑NEXT includes image editing and discusses consistency improvements using VAE latents. fileciteturn0file1  
If you later add editing episodes, evaluate on an editing benchmark.

---

## 15. Failure modes and guardrails (must implement)

This is where self-evolving systems often fail.

### 15.1 Reward hacking / collusion (generator + solver)
Mitigations:
- Keep solver as a **slow-moving judge** (freeze solver during early G-episodes)
- Use **frozen judge snapshots** for reward scoring
- Use **multi-constraint** reward (cycle + verify + agree + diversity)
- Maintain KL constraints via adaptive controller (EvoLMM) fileciteturn0file0

### 15.2 Mode collapse
Mitigations:
- diversity term
- group rollouts (G>=2)
- log diversity and penalize collapse

### 15.3 Forgetting understanding
Mitigations:
- alternating KU:KG schedule
- keep base frozen (LoRA only), as EvoLMM found LoRA stable in unsupervised RL. fileciteturn0file0  
- optionally periodic U-only refresh phase

---

## 16. Appendix: How this differs from BLIP3‑o (May 2025)

BLIP3‑o (not NEXT) uses diffusion to generate CLIP features and recommends sequential training to preserve understanding. fileciteturn0file2  
If you target BLIP3‑o specifically, the generator policy is not naturally discrete token trajectories, so RL plumbing is more complex. The BLIP3o‑NEXT discrete token design is the most direct match to EvoLMM-style RL loops.

---

## 17. Traceability to your feasibility report

This spec implements the core feasibility ideas:
- Extending EvoLMM’s internal consistency reward into generation via cycle/verification loops fileciteturn0file3  
- Routing learning through the AR token policy (not through denoising steps) by keeping diffusion frozen for RL, aligned with BLIP3o‑NEXT RL pipeline fileciteturn0file1  
- Alternating training to reduce catastrophic forgetting consistent with unified training strategy discussions fileciteturn0file2  

