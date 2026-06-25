# WISE Evaluation for BLIP3o

WISE (World Knowledge-Informed Semantic Evaluation) evaluates T2I models on 1000 prompts
across 25 sub-domains of cultural common sense, spatio-temporal reasoning, and natural science.

Metric: **WiScore** (GPT-4o based evaluation of consistency, realism, aesthetics).

## Setup

```bash
# 1. Clone WISE repo
cd BLIP3o/eval/wise
git clone https://github.com/PKU-YuanGroup/WISE.git wise_repo

# 2. Set OpenAI API key (required for GPT-4o evaluation)
export OPENAI_API_KEY="your-api-key"
```

## Evaluation Pipeline

### Step 1: Generate images (1 image per prompt, filenames: 1.png to 1000.png)
```bash
# Base model:
bash generate_wise_base.sh

# Our model:
CHECKPOINT_DIR=/path/to/step_00500 bash generate_wise_our.sh
```

### Step 2: Evaluate with WiScore (requires GPT-4o API)
```bash
bash evaluate_wise.sh /path/to/generated_images
```

## Categories
- cultural_common_sense.json (domains: festivals, mythology, art, architecture, etc.)
- spatio-temporal_reasoning.json (seasons, time of day, geography, etc.)
- natural_science.json (physics, biology, chemistry, astronomy, etc.)
