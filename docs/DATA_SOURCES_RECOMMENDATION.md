# Data Sources for Self-Evolving Unified Model Training

**Date:** 2026-01-30  
**Purpose:** Recommend data sources for training the EvoLMM + BLIP3-o self-evolving framework.

---

## Key Insight: We Only Need Raw Images

Since the self-evolving framework is **fully unsupervised**, we don't need:
- ❌ Question-answer pairs
- ❌ Image captions (except for generation training initialization)
- ❌ Human labels or annotations

We only need:
- ✅ **Diverse, high-quality images**
- ✅ **Variety in visual complexity** (for curriculum learning)
- ✅ **Domain coverage** (charts, documents, natural scenes, etc.)

---

## Current Codebase Data Patterns

### EvoLMM: `ImagePool` (Folder-Based)
```python
# From EvoLMM/src/train.py
class ImagePool:
    # Scans folders for images (.png, .jpg, .jpeg, .webp, .bmp, .tiff)
    # No labels needed - just image paths
    # Supports subfolder organization for domain grouping
```

**Config example:**
```yaml
data_dir: "/path/to/images"
include_subfolders: ["chartqa", "docvqa", "natural"]  # Optional filtering
```

### BLIP3-o: Webdataset (Not Required for Self-Evolution)
BLIP3-o typically uses webdataset with captions for supervised training, BUT:
- For self-evolution, we bypass this and use EvoLMM's ImagePool
- Captions are **generated internally**, not loaded from data

---

## Recommended Data Sources

### Tier 1: High Priority (Start Here)

| Dataset | Size | Domain | Why Use It | Access |
|---------|------|--------|------------|--------|
| **SA-1B** (Segment Anything) | 11M images | Natural scenes, objects | Massive diversity, clean images | [HuggingFace](https://huggingface.co/datasets/facebook/segment-anything-1-billion) |
| **LAION-COCO** | 600K | Natural + artistic | High-quality subset of LAION | [HuggingFace](https://huggingface.co/datasets/laion/laion-coco) |
| **ChartQA Images** | 20K | Charts, plots | Document understanding domain | [HuggingFace](https://huggingface.co/datasets/ahmed-masry/ChartQA) |
| **DocVQA Images** | 50K | Documents, forms | OCR and document understanding | [HuggingFace](https://huggingface.co/datasets/lmms-lab/DocVQA) |

### Tier 2: Domain Expansion

| Dataset | Size | Domain | Why Use It |
|---------|------|--------|------------|
| **TextVQA Images** | 28K | Scene text | Text-in-the-wild understanding |
| **AI2D** | 5K | Scientific diagrams | Diagram reasoning |
| **MathVista Images** | 6K | Math problems | Mathematical visual reasoning |
| **GQA Images** | 113K | Visual reasoning | Compositional questions |
| **COCO 2017** | 118K | Natural scenes | Standard benchmark images |

### Tier 3: Generation Quality (Optional)

| Dataset | Size | Domain | Why Use It |
|---------|------|--------|------------|
| **JourneyDB** | 4M | Midjourney outputs | High-quality generation targets |
| **DiffusionDB** | 14M | Diffusion outputs | Diverse generation styles |
| **Unsplash-Lite** | 25K | Professional photos | Clean, high-res images |

---

## Recommended Dataset Mix for Training

### Phase 1: Understanding Warm-Start (EvoLMM-style)
Focus on **diverse reasoning tasks**:
```
10K ChartQA images
10K DocVQA images  
20K Natural images (COCO/SA-1B subset)
5K AI2D diagrams
5K MathVista images
─────────────────
50K total
```

### Phase 2: Joint Understanding + Generation
Add **high-quality generation targets**:
```
Phase 1 images (50K)
+ 50K LAION-COCO (captioned for generation)
+ 20K JourneyDB (high-quality generation)
─────────────────
120K total
```

### Phase 3: Scale Up (If Needed)
```
500K SA-1B subset (diverse natural scenes)
100K Document images
50K Chart/diagram images
50K Generation-quality images (JourneyDB)
─────────────────
700K total
```

---

## Data Preparation Script

### Step 1: Directory Structure
```
/data/self_evolving/
├── understanding/
│   ├── chartqa/          # ChartQA images
│   ├── docvqa/           # DocVQA images
│   ├── ai2d/             # AI2D diagrams
│   ├── mathvista/        # MathVista images
│   └── natural/          # COCO/SA-1B natural scenes
├── generation/
│   ├── laion_coco/       # LAION-COCO with captions
│   └── journeydb/        # High-quality generation
└── mixed/                # Combined for joint training
```

### Step 2: Download Commands

```bash
# ChartQA images
python -c "
from datasets import load_dataset
ds = load_dataset('ahmed-masry/ChartQA', split='train')
ds.save_to_disk('/data/self_evolving/understanding/chartqa')
"

# DocVQA images (images only)
python -c "
from datasets import load_dataset
ds = load_dataset('lmms-lab/DocVQA', split='train')
ds.save_to_disk('/data/self_evolving/understanding/docvqa')
"

# COCO 2017 images
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d /data/self_evolving/understanding/natural/

# SA-1B subset (use HuggingFace streaming)
python -c "
from datasets import load_dataset
ds = load_dataset('facebook/segment-anything-1-billion', split='train', streaming=True)
# Take first 100K images
import itertools
subset = list(itertools.islice(ds, 100000))
# Save images to disk
"

# LAION-COCO (for generation training)
python -c "
from datasets import load_dataset
ds = load_dataset('laion/laion-coco', split='train[:50000]')
ds.save_to_disk('/data/self_evolving/generation/laion_coco')
"
```

### Step 3: Extract Images to Folders

```python
# convert_to_folder.py
import os
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

def extract_images(dataset_path, output_dir):
    ds = load_from_disk(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for i, item in enumerate(tqdm(ds)):
        img = item['image']
        if isinstance(img, Image.Image):
            img.save(os.path.join(output_dir, f"{i:08d}.jpg"))
        elif 'bytes' in img:
            with open(os.path.join(output_dir, f"{i:08d}.jpg"), 'wb') as f:
                f.write(img['bytes'])

# Run for each dataset
extract_images('/data/self_evolving/understanding/chartqa', 
               '/data/self_evolving/images/chartqa')
```

---

## EvoLMM Config for Custom Data

```python
# In EvoLMM training config
cfg = Config(
    data_dir="/data/self_evolving/images",
    include_subfolders=["chartqa", "docvqa", "natural", "ai2d"],  # Select domains
    # ... other config
)
```

---

## Data Quality Considerations

### For Understanding Tasks
- ✅ **Diverse complexity:** Mix simple (natural scenes) with complex (charts, documents)
- ✅ **Multi-domain:** Cover different visual domains for generalization
- ✅ **Good resolution:** At least 512×512 for detail

### For Generation Tasks
- ✅ **High aesthetic quality:** Clean, well-composed images
- ✅ **Diverse content:** Avoid repetitive styles
- ✅ **Caption availability:** Need prompts for generation training
- ✅ **1024×1024 preferred:** BLIP3-o uses 1024 for generation

### Anti-Patterns to Avoid
- ❌ Duplicate images (causes overfitting)
- ❌ Watermarked images (confuses generation)
- ❌ Very low resolution (<256px)
- ❌ Corrupted/truncated files

---

## Compute vs. Data Tradeoff

| Data Size | Understanding Steps | Generation Steps | Total GPU Hours (8×A100) |
|-----------|---------------------|------------------|--------------------------|
| 50K | 10K steps | 2K steps | ~24 hours |
| 120K | 25K steps | 5K steps | ~72 hours |
| 700K | 100K steps | 20K steps | ~2 weeks |

**Recommendation:** Start with 50K to validate the loop, then scale.

---

## Summary: What to Download Now

1. **ChartQA images** (20K) - Document understanding
2. **DocVQA images** (50K) - OCR/document domain  
3. **COCO 2017 train** (118K) - Natural scenes
4. **LAION-COCO subset** (50K) - Generation with captions

**Total: ~240K images, ~50GB storage**

This gives you a balanced mix for both understanding and generation training.
