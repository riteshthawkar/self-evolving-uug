# DPG-Bench Evaluation for BLIP3o

DPG-Bench (Dense Prompt Graph Benchmark) evaluates text-to-image models on 1K dense prompts
using hierarchical VQA-based scoring.

## Setup

```bash
# 1. Clone ELLA repo (contains DPG-Bench eval code)
cd BLIP3o/eval/dpg_bench
git clone https://github.com/TencentQQGYLab/ELLA.git ella_repo

# 2. Install dependencies
pip install t2v-metrics  # for VQAScore (alternative metric)
pip install modelscope   # for mplug VQA model (official DPG-Bench metric)

# 3. The prompts and eval CSV are in ella_repo/dpg_bench/
```

## Evaluation Pipeline

### Step 1: Generate images
```bash
# Base BLIP3o model:
bash generate_dpg_base.sh

# Our self-evolving trained model:
CHECKPOINT_DIR=/path/to/step_00500 bash generate_dpg_our.sh
```

### Step 2: Evaluate with DPG-Bench
```bash
# Uses mplug VQA model for question answering
bash evaluate_dpg.sh /path/to/generated_images 512
```

### Step 3: (Optional) Evaluate with VQAScore
```bash
python evaluate_vqascore.py --image_dir /path/to/generated_images --prompt_file prompts.txt
```
