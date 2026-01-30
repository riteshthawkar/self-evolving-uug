# Detailed Pipeline Documentation: Self-Evolving Unified Multimodal Model

**Document Type:** Technical Implementation Guide  
**Date:** January 30, 2026  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Loading Pipeline](#2-data-loading-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Forward Pass: Understanding](#4-forward-pass-understanding)
5. [Forward Pass: Generation](#5-forward-pass-generation)
6. [Reward Computation](#6-reward-computation)
7. [RL Training Loop](#7-rl-training-loop)
8. [Complete Training Step](#8-complete-training-step)

---

## 1. Overview

The unified self-evolving pipeline combines two frameworks:

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED PIPELINE OVERVIEW                     │
├─────────────────────────────────────────────────────────────────┤
│  RAW IMAGE                                                       │
│      │                                                           │
│      ▼                                                           │
│  ┌─────────────┐                                                │
│  │  Proposer   │ ──► Generates Question/Prompt                  │
│  └─────────────┘                                                │
│      │                                                           │
│      ▼                                                           │
│  ┌─────────────┐                                                │
│  │   Solver/   │ ──► Answers Questions OR Generates Images      │
│  │  Generator  │                                                 │
│  └─────────────┘                                                │
│      │                                                           │
│      ▼                                                           │
│  ┌─────────────┐                                                │
│  │   Reward    │ ──► Self-Consistency Score (No Human Labels)   │
│  │ Computation │                                                 │
│  └─────────────┘                                                │
│      │                                                           │
│      ▼                                                           │
│  ┌─────────────┐                                                │
│  │ RL Update   │ ──► REINFORCE + KL Regularization              │
│  │  (LoRA)     │                                                 │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Loading Pipeline

### 2.1 Image Pool (EvoLMM Approach)

**Key Principle:** Only raw images needed - NO annotations, NO labels, NO metadata.

```python
class ImagePool:
    """
    Loads images from a directory structure.
    No labels/annotations required - fully unsupervised.
    """
    DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = []  # List of all image paths
        
        root = os.path.abspath(cfg.data_dir)
        
        # Scan directory recursively for images
        for subfolder in os.listdir(root):
            subfolder_path = os.path.join(root, subfolder)
            if os.path.isdir(subfolder_path):
                for fname in os.listdir(subfolder_path):
                    if self._is_img(fname):
                        self.paths.append(os.path.join(subfolder_path, fname))
        
        # Deterministic shuffling for reproducibility
        self.indices = list(range(len(self.paths)))
        random.Random(cfg.seed).shuffle(self.indices)
    
    def _is_img(self, fn):
        return fn.lower().endswith(self.DEFAULT_EXTS)
    
    def sample_by_iter(self, iter_no):
        """Deterministic sampling based on iteration number."""
        idx = self.indices[iter_no % len(self.indices)]
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        meta = {"path": path, "subfolder": os.path.dirname(path)}
        return image, meta
```

### 2.2 Data Flow Diagram

```
Directory Structure:
images/
├── train/
│   ├── charts/
│   │   ├── chart_001.png
│   │   └── chart_002.png
│   ├── diagrams/
│   │   └── diagram_001.png
│   └── photos/
│       └── photo_001.jpg

                    ▼
            ┌───────────────┐
            │   ImagePool   │
            │   __init__    │
            └───────────────┘
                    │
                    ▼
            paths = [
              "images/train/charts/chart_001.png",
              "images/train/charts/chart_002.png",
              "images/train/diagrams/diagram_001.png",
              "images/train/photos/photo_001.jpg"
            ]
                    │
                    ▼
            ┌───────────────┐
            │ sample_by_iter│
            │   (step=42)   │
            └───────────────┘
                    │
                    ▼
            Returns: (PIL.Image, {"path": "...", "subfolder": "..."})
```

---

## 3. Model Architecture

### 3.1 Unified Model Components

```
┌────────────────────────────────────────────────────────────────────┐
│                      BLIP3-o + EvoLMM Architecture                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    VISION ENCODER (Frozen)                   │   │
│  │                                                              │   │
│  │   Input Image ──► SigLIP/CLIP ──► 64 Visual Tokens          │   │
│  │   (224x224)        Encoder        (Each: 1024-dim)          │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LLM BACKBONE (LoRA Adapters)              │   │
│  │                                                              │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │               Qwen2.5-VL-7B-Instruct                  │  │   │
│  │   │                                                       │  │   │
│  │   │  Base Weights (Frozen)                               │  │   │
│  │   │       +                                               │  │   │
│  │   │  LoRA Adapter: "proposer" (Trainable)                │  │   │
│  │   │       +                                               │  │   │
│  │   │  LoRA Adapter: "solver" (Trainable)                  │  │   │
│  │   │                                                       │  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│              ┌───────────────┴───────────────┐                      │
│              ▼                               ▼                      │
│  ┌─────────────────────┐         ┌─────────────────────┐           │
│  │  TEXT OUTPUT        │         │  DIFFUSION HEAD     │           │
│  │  (Understanding)    │         │  (Generation)       │           │
│  │                     │         │                     │           │
│  │  Cross-Entropy Loss │         │  Visual Features Q  │           │
│  │  on Answer Tokens   │         │       │             │           │
│  │                     │         │       ▼             │           │
│  │                     │         │  Diffusion          │           │
│  │                     │         │  Transformer        │           │
│  │                     │         │  (Flow Matching)    │           │
│  │                     │         │       │             │           │
│  │                     │         │       ▼             │           │
│  │                     │         │  VAE Decoder        │           │
│  │                     │         │       │             │           │
│  │                     │         │       ▼             │           │
│  │                     │         │  Generated Image    │           │
│  └─────────────────────┘         └─────────────────────┘           │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 LoRA Adapter Configuration

```python
# LoRA targets for both Proposer and Solver
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,
    target_modules=[
        "q_proj",            # Query projection
        "k_proj",            # Key projection
        "v_proj",            # Value projection
        "o_proj",            # Output projection
        "gate_proj",         # MLP gate
        "up_proj",           # MLP up
        "down_proj",         # MLP down
        "mm_projector",      # Vision-language connector
    ],
    task_type="CAUSAL_LM",
)

# Apply LoRA to create two adapters on same backbone
model = get_peft_model(base_model, lora_config)
model.add_adapter("proposer", lora_config)  # Second adapter
# "default" adapter = solver
```

---

## 4. Forward Pass: Understanding

### 4.1 Proposer: Question Generation

```
INPUT: Raw Image
OUTPUT: Visually-grounded question

Step-by-Step:
─────────────
1. Image Encoding
   Image (PIL) ──► Processor ──► pixel_values tensor
   
2. Prompt Construction
   prompt = """You are given an image. Generate a challenging 
   mathematical or reasoning question about this image.
   
   <question>YOUR_QUESTION_HERE</question>"""
   
3. Text Tokenization
   prompt ──► Tokenizer ──► input_ids, attention_mask
   
4. Forward Pass (with "proposer" adapter active)
   model.set_adapter("proposer")
   outputs = model.generate(
       pixel_values=pixel_values,
       input_ids=input_ids,
       max_new_tokens=128,
       temperature=1.0,
       top_p=1.0,
       do_sample=True
   )
   
5. Decode Output
   question = tokenizer.decode(outputs)
   question = extract_between_tags(question, "question")
```

### 4.2 Solver: Answer Generation

```
INPUT: Image + Question
OUTPUT: N sampled answers

Step-by-Step:
─────────────
1. Prompt Construction (for each of N samples)
   prompt = f"""Look at this image and answer the question.
   
   Question: {question}
   
   Think step by step, then provide your final answer.
   <answer>YOUR_ANSWER</answer>"""

2. Generate N Samples (with "solver" adapter active)
   model.set_adapter("solver")  # or "default"
   
   answers = []
   for _ in range(N):  # N = 5 typically
       output = model.generate(
           pixel_values=pixel_values,
           input_ids=input_ids,
           max_new_tokens=128,
           temperature=1.0,  # Enable sampling
           do_sample=True
       )
       answer = extract_between_tags(output, "answer")
       answers.append(answer)
   
   # answers = ["42", "40", "42", "42", "38"]

3. Compute Answer Distribution
   normalized = [normalize(a) for a in answers]
   # normalized = ["42", "40", "42", "42", "38"]
   
   histogram = Counter(normalized)
   # histogram = {"42": 3, "40": 1, "38": 1}
   
   probabilities = {k: v/N for k, v in histogram.items()}
   # probabilities = {"42": 0.6, "40": 0.2, "38": 0.2}
```

---

## 5. Forward Pass: Generation

### 5.1 Proposer: Prompt Generation

```
INPUT: Raw Image
OUTPUT: Text prompt for image generation

Step-by-Step:
─────────────
1. Caption the image (using understanding mode)
   caption = model.caption(image)
   # caption = "A bar chart showing sales data for Q1-Q4"

2. Generate creative prompt variation
   prompt_template = f"""Based on this image showing: {caption}
   
   Generate a creative prompt for generating a similar image.
   <prompt>YOUR_PROMPT</prompt>"""
   
   generated_prompt = model.generate(prompt_template)
   # generated_prompt = "A colorful bar chart with 4 bars..."
```

### 5.2 Generator: Image Generation

```
INPUT: Text prompt
OUTPUT: Generated image(s)

Step-by-Step (BLIP3-o architecture):
────────────────────────────────────
1. Text Encoding
   prompt ──► Tokenizer ──► input_ids
   
2. Generate Visual Feature Tokens Q
   # Autoregressive model generates intermediate features
   visual_tokens = model.generate_visual_features(
       input_ids=input_ids,
       num_tokens=64  # Fixed length for CLIP features
   )
   # visual_tokens shape: [batch, 64, hidden_dim]

3. Diffusion Connector
   # Project LLM hidden states to diffusion space
   diffusion_input = diffusion_connector(visual_tokens)
   # diffusion_input shape: [batch, 64, 2304]

4. Flow Matching Diffusion Process
   # Initialize from noise
   x_t = torch.randn(batch, latent_channels, H, W)
   
   # Iterative denoising (e.g., 20 steps)
   for t in scheduler.timesteps:
       # Predict velocity
       v_pred = diffusion_transformer(
           x_t, 
           timestep=t,
           encoder_hidden_states=diffusion_input
       )
       # Update x_t
       x_t = scheduler.step(v_pred, t, x_t)
   
   # x_0 = final denoised latent

5. VAE Decoding
   generated_image = vae.decode(x_0)
   # generated_image shape: [batch, 3, 1024, 1024]
```

---

## 6. Reward Computation

### 6.1 Understanding Rewards

```python
def compute_understanding_rewards(answers, question):
    """
    Compute rewards for understanding task.
    
    Parameters:
    - answers: List[str] - N sampled answers
    - question: str - The proposed question
    
    Returns:
    - solver_rewards: List[float] - Reward for each answer
    - proposer_reward: float - Reward for the question
    """
    N = len(answers)
    
    # Step 1: Normalize answers
    normalized = [normalize_answer(a) for a in answers]
    
    # Step 2: Build histogram
    histogram = Counter(normalized)
    # Example: {"42": 3, "40": 1, "38": 1}
    
    # Step 3: Compute probabilities
    probs = {k: v/N for k, v in histogram.items()}
    # Example: {"42": 0.6, "40": 0.2, "38": 0.2}
    
    # Step 4: Compute entropy
    entropy = -sum(p * math.log(p) for p in probs.values())
    # Example: H = -0.6*log(0.6) - 0.2*log(0.2) - 0.2*log(0.2) ≈ 0.95 nats
    
    # ═══════════════════════════════════════════════════════
    # SOLVER REWARDS (Continuous Self-Consistency)
    # ═══════════════════════════════════════════════════════
    
    gamma = 0.7  # Softness parameter
    solver_rewards = []
    
    for i, ans in enumerate(normalized):
        p_ans = probs[ans]  # Probability of this answer
        
        # Length penalty (encourage concise answers)
        word_count = count_words_before_answer(answers[i])
        target_words = 6
        penalty = max(0, (word_count - target_words) / target_words)
        
        # Continuous reward formula
        reward = (p_ans ** gamma) * (1 - 0.1 * penalty)
        solver_rewards.append(reward)
    
    # Example rewards:
    # Answer "42" (p=0.6): reward = 0.6^0.7 * 0.9 = 0.64
    # Answer "40" (p=0.2): reward = 0.2^0.7 * 0.9 = 0.28
    # Answer "38" (p=0.2): reward = 0.2^0.7 * 0.9 = 0.28
    
    # ═══════════════════════════════════════════════════════
    # PROPOSER REWARD (Entropy-Based Band-Pass)
    # ═══════════════════════════════════════════════════════
    
    mu = 0.9      # Target entropy (moderate difficulty)
    sigma = 0.35  # Width of band
    
    proposer_reward = math.exp(-((entropy - mu)**2) / (2 * sigma**2))
    
    # If entropy ≈ 0.9 (moderate difficulty): reward ≈ 1.0
    # If entropy ≈ 0.0 (trivial question): reward ≈ 0.1
    # If entropy ≈ 2.0 (ambiguous question): reward ≈ 0.1
    
    return solver_rewards, proposer_reward
```

### 6.2 Generation Rewards

```python
def compute_generation_rewards(images, prompt, model):
    """
    Compute rewards for generation task.
    FULLY SELF-SUPERVISED - no external models!
    
    Parameters:
    - images: List[PIL.Image] - N generated images
    - prompt: str - The generation prompt
    - model: UnifiedModel - BLIP3-o model (used for understanding)
    
    Returns:
    - generator_rewards: List[float] - Reward for each image
    - proposer_reward: float - Reward for the prompt
    """
    N = len(images)
    
    # ═══════════════════════════════════════════════════════
    # METHOD 1: CLIP Embedding Consistency
    # ═══════════════════════════════════════════════════════
    
    # Get CLIP embeddings (using BLIP3-o's vision encoder)
    embeddings = []
    for img in images:
        emb = model.vision_tower.encode(img)  # [64, 1024]
        emb_pooled = emb.mean(dim=0)          # [1024]
        embeddings.append(emb_pooled)
    
    embeddings = torch.stack(embeddings)  # [N, 1024]
    
    # Compute pairwise cosine similarities
    similarities = torch.mm(embeddings, embeddings.T)
    # Off-diagonal mean = consistency score
    mask = ~torch.eye(N, dtype=bool)
    consistency = similarities[mask].mean()
    # consistency ∈ [0, 1], higher = more consistent
    
    # ═══════════════════════════════════════════════════════
    # METHOD 2: Round-Trip Verification (Self-Supervised)
    # ═══════════════════════════════════════════════════════
    
    # Use BLIP3-o's understanding mode to caption generated images
    captions = []
    for img in images:
        caption = model.caption(img)  # Uses understanding mode
        captions.append(caption)
    
    # Compare captions to original prompt
    alignments = []
    for caption in captions:
        # Simple word overlap or embedding similarity
        alignment = compute_text_similarity(caption, prompt)
        alignments.append(alignment)
    
    # ═══════════════════════════════════════════════════════
    # GENERATOR REWARDS (Per-Image)
    # ═══════════════════════════════════════════════════════
    
    generator_rewards = []
    for i in range(N):
        reward = 0.5 * alignments[i] + 0.5 * consistency
        generator_rewards.append(reward)
    
    # ═══════════════════════════════════════════════════════
    # PROPOSER REWARD (Entropy-Based)
    # ═══════════════════════════════════════════════════════
    
    # Compute entropy over caption distribution
    caption_embeddings = [model.text_encode(c) for c in captions]
    caption_variance = torch.var(torch.stack(caption_embeddings))
    
    # Band-pass reward (like understanding task)
    proposer_reward = gaussian_bandpass(caption_variance, mu=0.5, sigma=0.2)
    
    return generator_rewards, proposer_reward
```

### 6.3 Reward Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      REWARD COMPUTATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  UNDERSTANDING TASK:                                             │
│  ──────────────────                                              │
│                                                                  │
│  Answers: ["42", "40", "42", "42", "38"]                        │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────┐                        │
│  │  Histogram: {42: 3, 40: 1, 38: 1}   │                        │
│  │  Probs: {42: 0.6, 40: 0.2, 38: 0.2} │                        │
│  │  Entropy H = 0.95 nats              │                        │
│  └─────────────────────────────────────┘                        │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌────────────────┐      ┌────────────────┐                     │
│  │ Solver Rewards │      │Proposer Reward │                     │
│  │ r = p^γ × pen  │      │ r = exp(-d²)   │                     │
│  │                │      │   d = H - μ    │                     │
│  │ [0.64, 0.28,   │      │                │                     │
│  │  0.64, 0.64,   │      │ r = 0.98       │                     │
│  │  0.28]         │      │                │                     │
│  └────────────────┘      └────────────────┘                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GENERATION TASK:                                                │
│  ────────────────                                                │
│                                                                  │
│  Generated Images: [img1, img2, img3, img4, img5]               │
│                    │                                             │
│          ┌────────┴────────┐                                    │
│          ▼                 ▼                                     │
│  ┌───────────────┐ ┌───────────────┐                            │
│  │CLIP Embedding │ │  Round-Trip   │                            │
│  │  Consistency  │ │  Captioning   │                            │
│  │               │ │               │                            │
│  │ cos_sim=0.85  │ │ align=[0.7,   │                            │
│  │               │ │  0.8, 0.75,   │                            │
│  │               │ │  0.7, 0.8]    │                            │
│  └───────────────┘ └───────────────┘                            │
│          │                 │                                     │
│          └────────┬────────┘                                    │
│                   ▼                                              │
│  ┌─────────────────────────────────────┐                        │
│  │ Generator Rewards = 0.5*align +     │                        │
│  │                     0.5*consistency │                        │
│  │ = [0.775, 0.825, 0.8, 0.775, 0.825]│                        │
│  └─────────────────────────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. RL Training Loop

### 7.1 REINFORCE with KL Regularization

```python
class PolicyUpdater:
    """
    Implements REINFORCE policy gradient with:
    1. Token-level KL regularization to reference model
    2. Adaptive KL coefficient (β)
    3. Moving average baseline for variance reduction
    """
    
    def __init__(self, policy, ref_policy, cfg):
        self.policy = policy        # Trainable LoRA model
        self.ref_policy = ref_policy  # Frozen reference model
        self.kl_coef = cfg.kl_coef   # Initial KL coefficient
        self.kl_target = cfg.kl_target  # Target KL divergence
        self.kl_adapt_rate = cfg.kl_adapt_rate
        
        # Optimizer (only LoRA parameters)
        self.opt = AdamW(
            self.policy.trainable_params(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    
    def step(self, image, prompt, completion, reward, baseline):
        """
        Perform one policy gradient update.
        
        Loss = advantage × CE_loss + β × KL_loss
        
        Where:
        - advantage = reward - baseline
        - CE_loss = -log P(completion | prompt, image)
        - KL_loss = KL(policy || reference)
        """
        
        # Compute advantage
        advantage = reward - baseline
        
        # Forward pass: policy model
        inputs = self.tokenize(image, prompt + completion)
        
        self.policy.train()
        outputs_policy = self.policy(**inputs)
        logits_policy = outputs_policy.logits
        
        # Forward pass: reference model (frozen)
        with torch.no_grad():
            outputs_ref = self.ref_policy(**inputs)
            logits_ref = outputs_ref.logits
        
        # Compute log probabilities
        logp_policy = compute_logprob(logits_policy, inputs.labels)
        logp_ref = compute_logprob(logits_ref, inputs.labels)
        
        # Token-level KL divergence
        p_policy = torch.softmax(logits_policy, dim=-1)
        kl_per_token = (p_policy * (logits_policy - logits_ref)).sum(-1)
        kl_loss = kl_per_token.mean()
        
        # Cross-entropy loss (negative log likelihood)
        ce_loss = -logp_policy.mean()
        
        # Total loss
        loss = advantage * ce_loss + self.kl_coef * kl_loss
        
        # Backward pass
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.trainable_params(), 1.0)
        self.opt.step()
        
        # Adapt KL coefficient
        self._adapt_beta(kl_loss.item())
        
        return {
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
            "advantage": advantage,
            "total_loss": loss.item()
        }
    
    def _adapt_beta(self, current_kl):
        """
        Adaptive KL coefficient:
        - If KL too high: increase β to penalize more
        - If KL too low: decrease β to allow exploration
        """
        if current_kl > self.kl_target:
            self.kl_coef *= (1 + self.kl_adapt_rate)
        else:
            self.kl_coef *= (1 - self.kl_adapt_rate)
        
        # Clamp to reasonable range
        self.kl_coef = max(1e-6, min(1.0, self.kl_coef))
```

### 7.2 Baseline Update (Variance Reduction)

```python
class MovingAverageBaseline:
    """
    Exponential moving average baseline for variance reduction.
    """
    def __init__(self, momentum=0.9):
        self.value = 0.0
        self.momentum = momentum
    
    def update(self, reward):
        self.value = self.momentum * self.value + (1 - self.momentum) * reward
        return self.value
    
    def get(self):
        return self.value
```

---

## 8. Complete Training Step

### 8.1 Full Training Loop Pseudocode

```python
def train(cfg):
    # Initialize components
    pool = ImagePool(cfg)
    
    # Shared backbone with two LoRA adapters
    backbone = load_model("Qwen2.5-VL-7B-Instruct")
    backbone = apply_lora(backbone, adapters=["proposer", "solver"])
    
    proposer = VLMRole(backbone, adapter="proposer")
    solver = VLMRole(backbone, adapter="solver")
    
    # Reference model (frozen, no adapters)
    ref_model = load_model("Qwen2.5-VL-7B-Instruct")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # Policy updaters
    proposer_updater = PolicyUpdater(proposer, ref_model, cfg)
    solver_updater = PolicyUpdater(solver, ref_model, cfg)
    
    # Baselines
    proposer_baseline = MovingAverageBaseline()
    solver_baseline = MovingAverageBaseline()
    
    # Training loop
    for step in range(1, cfg.total_steps + 1):
        
        # ════════════════════════════════════════════════
        # STEP 1: Sample image
        # ════════════════════════════════════════════════
        image, meta = pool.sample_by_iter(step)
        
        # ════════════════════════════════════════════════
        # STEP 2: Proposer generates question
        # ════════════════════════════════════════════════
        proposer_prompt = build_proposer_prompt(meta)
        question = proposer.generate(image, proposer_prompt)
        question = extract_question(question)
        
        # ════════════════════════════════════════════════
        # STEP 3: Solver answers N times
        # ════════════════════════════════════════════════
        solver_prompt = build_solver_prompt(question)
        answers = []
        completions = []
        
        for _ in range(cfg.num_solver_samples):  # N=5
            output = solver.generate(image, solver_prompt)
            answer = extract_answer(output)
            answers.append(answer)
            completions.append(output)
        
        # ════════════════════════════════════════════════
        # STEP 4: Compute rewards
        # ════════════════════════════════════════════════
        solver_rewards, proposer_reward = compute_understanding_rewards(
            answers, question
        )
        
        # ════════════════════════════════════════════════
        # STEP 5: Update Solver (for each sample)
        # ════════════════════════════════════════════════
        for completion, reward in zip(completions, solver_rewards):
            stats = solver_updater.step(
                image=image,
                prompt=solver_prompt,
                completion=completion,
                reward=reward,
                baseline=solver_baseline.get()
            )
            solver_baseline.update(reward)
        
        # ════════════════════════════════════════════════
        # STEP 6: Update Proposer (every K steps)
        # ════════════════════════════════════════════════
        if step % cfg.proposer_update_freq == 0:  # K=5
            stats = proposer_updater.step(
                image=image,
                prompt=proposer_prompt,
                completion=question,
                reward=proposer_reward,
                baseline=proposer_baseline.get()
            )
            proposer_baseline.update(proposer_reward)
        
        # ════════════════════════════════════════════════
        # STEP 7: Logging and checkpointing
        # ════════════════════════════════════════════════
        if step % cfg.log_every == 0:
            log_metrics(step, solver_rewards, proposer_reward)
        
        if step % cfg.save_every == 0:
            save_checkpoint(step, backbone, proposer_updater, solver_updater)
```

### 8.2 Visual Summary of One Training Step

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONE TRAINING STEP (Understanding)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ①  SAMPLE IMAGE                                                │
│      pool.sample_by_iter(step=42) → image, meta                 │
│                                                                  │
│  ②  PROPOSER GENERATES QUESTION                                 │
│      proposer.generate(image, prompt) →                         │
│      "What is the sum of all values in the bar chart?"          │
│                                                                  │
│  ③  SOLVER ANSWERS 5 TIMES                                      │
│      solver.generate(image, question) × 5 →                     │
│      ["42", "40", "42", "42", "38"]                             │
│                                                                  │
│  ④  COMPUTE REWARDS                                             │
│      ┌───────────────────────────────────────┐                  │
│      │ Histogram: {42: 3, 40: 1, 38: 1}      │                  │
│      │ Entropy: H = 0.95 nats                │                  │
│      │                                        │                  │
│      │ Solver Rewards:                        │                  │
│      │   r = p^γ × penalty                    │                  │
│      │   [0.64, 0.28, 0.64, 0.64, 0.28]      │                  │
│      │                                        │                  │
│      │ Proposer Reward:                       │                  │
│      │   r = exp(-(H-μ)²/2σ²) = 0.98         │                  │
│      └───────────────────────────────────────┘                  │
│                                                                  │
│  ⑤  UPDATE SOLVER (5 gradient steps)                            │
│      for each (completion, reward):                             │
│        advantage = reward - baseline                            │
│        loss = advantage × CE + β × KL                           │
│        loss.backward()                                          │
│        optimizer.step()                                         │
│                                                                  │
│  ⑥  UPDATE PROPOSER (every 5 steps)                             │
│      if step % 5 == 0:                                          │
│        advantage = proposer_reward - baseline                   │
│        loss = advantage × CE + β × KL                           │
│        loss.backward()                                          │
│        optimizer.step()                                         │
│                                                                  │
│  ⑦  UPDATE BASELINES                                            │
│      solver_baseline = 0.9 × old + 0.1 × mean(rewards)         │
│      proposer_baseline = 0.9 × old + 0.1 × proposer_reward     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Hyperparameters Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_solver_samples` | 5 | N - number of answer samples |
| `proposer_update_freq` | 5 | Update proposer every K steps |
| `lr` | 1e-6 | Learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling |
| `solver_soft_gamma` | 0.7 | γ in p^γ reward |
| `prop_entropy_mu` | 0.9 | μ for proposer entropy band |
| `prop_entropy_sigma` | 0.35 | σ for proposer entropy band |
| `kl_target` | 0.02 | Target KL divergence |
| `kl_adapt_rate` | 0.10 | KL coefficient adaptation rate |
| `kl_coef` | 0.001 | Initial KL coefficient |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling |
| `lora_dropout` | 0.05 | LoRA dropout |

---

*Document generated for implementation reference.*
