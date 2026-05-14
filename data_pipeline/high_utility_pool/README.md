# High-Utility Image Pool Pipeline

This folder builds a compact unlabeled image pool for the BLIP3o self-evolving
framework. The goal is not generic image quality. The goal is high proposer
utility: images should support hard, objective, visually grounded questions
that create useful solver disagreement without becoming ambiguous.

## What It Builds

Default output:

```text
data/high_utility_pool_10k/
  images/
    relational/
    spatial/
    ocr/
    chart_document/
    openimages_dense/
    natural/
  manifest.jsonl
  rejected.jsonl
  audit_report.json
  scores/
    heuristic_scores.jsonl
    vlm_scores.jsonl
```

Each manifest row stores source, domain, image path, quality features, utility
score, duplicate hash, source URL/license when available, and optional VLM judge
scores.

## Sources

The pipeline uses two low-space sources by default:

1. Existing local pool: `data/joint_pool_10k/images`
2. Open Images V7 validation metadata and annotations

Open Images is used through its official metadata/annotation CSVs and selected
thumbnail URLs, so we only download images that pass dense-scene filters. This
avoids downloading full datasets.

## Scoring Signal

The heuristic score combines:

- resolution and aspect ratio
- blur/detail via Laplacian variance
- pixel variance
- edge-region density
- color diversity
- OCR/layout proxy
- Open Images box, label, relationship, occlusion, and group annotations
- predicted solver-disagreement proxy
- domain balancing

Optional VLM judging can refine accepted candidates with GPT/Gemini-style
multimodal evaluation or a local Apple Silicon MLX VLM. The VLM is used only
for filtering and metadata, not as training labels. The default VLM sampling
strategy is stratified across domains and heuristic-score quantiles, so a small
local judging budget audits obvious strong images plus mid/borderline cases.

## Quick Smoke Test

This runs locally without downloading Open Images and uses hardlinks, so it is
cheap on disk.

```bash
python3 data_pipeline/high_utility_pool/build_high_utility_pool.py \
  --local_source data/joint_pool_10k/images \
  --output_dir data/high_utility_pool_smoke \
  --target_count 64 \
  --smoke_test
```

## Build The 10k Pool

```bash
python3 data_pipeline/high_utility_pool/build_high_utility_pool.py \
  --config data_pipeline/high_utility_pool/config_high_utility_10k.json
```

The default config is capped at `max_output_gb=8.0`. Local images are hardlinked
when possible. Downloaded Open Images thumbnails are resized to at most 896 px
on the long side before final materialization. The config scores 10k Open
Images candidates to keep the final 10k pool at the stricter default utility
threshold. URL download and image scoring use `download_workers=8` by default;
reduce it if your network is unstable.

Explicit command-line flags override values from the JSON config.

## Optional OpenAI VLM Judge

Use this when you want a stronger final filter on the top heuristic candidates.
Keep `vlm_max_images` modest because image inputs cost money.

```bash
OPENAI_API_KEY=... python3 data_pipeline/high_utility_pool/build_high_utility_pool.py \
  --config data_pipeline/high_utility_pool/config_high_utility_10k.json \
  --vlm_backend openai \
  --vlm_model gpt-4.1-mini \
  --vlm_max_images 1500 \
  --vlm_candidate_top_k 4000
```

## Optional Gemini VLM Judge

```bash
GEMINI_API_KEY=... python3 data_pipeline/high_utility_pool/build_high_utility_pool.py \
  --config data_pipeline/high_utility_pool/config_high_utility_10k.json \
  --vlm_backend gemini \
  --vlm_model gemini-2.0-flash \
  --vlm_max_images 1500
```

This requires the `google-genai` package.

## Optional Local Qwen3-VL Judge

On Apple Silicon, use the MLX backend to run a local open VLM without API keys.
The tested low-memory model is `mlx-community/Qwen3-VL-4B-Instruct-4bit`; it
uses about 4 GB peak memory for the judge prompt on this machine.

Install the optional runtime in an isolated Python 3.12 environment that can
import MLX. Keep it separate from the BLIP3o training environment, because
`mlx-vlm` may install a newer `transformers` stack than the trainer expects.

```bash
/usr/local/bin/python3.12 -m pip install -r data_pipeline/high_utility_pool/requirements-local-vlm.txt
```

Run a bounded local audit first:

```bash
HF_HUB_DISABLE_XET=1 /usr/local/bin/python3.12 \
  data_pipeline/high_utility_pool/build_high_utility_pool.py \
  --local_source data/high_utility_pool_10k/images \
  --output_dir data/high_utility_pool_10k_qwen_audit \
  --target_count 10000 \
  --no_openimages \
  --vlm_backend mlx_vlm \
  --vlm_model mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --vlm_max_images 256 \
  --vlm_candidate_top_k 10000 \
  --vlm_selection_strategy stratified
```

For a full VLM-rescored pool, increase `--vlm_max_images` in stages. A full
10k local VLM pass is possible, but it is intentionally not the default because
it can take hours on a laptop.

## H200 Large-VLM Judge

For a 128 GB H200 machine, use the staged launcher below. It first runs
heuristic scoring, then audits the selected candidates with either a vLLM
OpenAI-compatible server or a direct Hugging Face Transformers backend. The
default model is `Qwen/Qwen3-VL-30B-A3B-Instruct` because the model card
documents both Transformers and vLLM usage, and the 30B-A3B MoE checkpoint fits
comfortably on a single H200 for this one-image judge workload.

```bash
cd /path/to/self-evolving-uug
pip install -U -r data_pipeline/high_utility_pool/requirements-h200-vlm.txt
```

Download or unpack the 10k image source first, then run a bounded smoke audit.
The launcher defaults to direct Hugging Face Transformers inference. This is
slower per image than a healthy vLLM server, but it is the most reliable
research path because it removes server startup, IPC, encoder-cache profiling,
CUDA graph capture, and scheduler warmup from the critical path.
It batches independent image judgments by default (`VLM_BATCH_SIZE=4`) and
requests FlashAttention-2 when available; if that attention backend is missing,
the loader automatically retries with the model's default attention.

```bash
SOURCE_DIR=/absolute/path/to/high_utility_pool_10k/images \
DRY_RUN=1 \
VLM_MAX_IMAGES=128 \
VLM_BATCH_SIZE=4 \
bash data_pipeline/high_utility_pool/run_h200_qwen72b_pipeline.sh
```

On a 128 GB H200, tune only `VLM_BATCH_SIZE` for throughput. Start with 4, try
8 if GPU memory has headroom, and drop to 2 only if the process OOMs. This does
not change the judge, prompt, image filtering criteria, or final scoring rule.

To test vLLM explicitly, opt in with `VLM_BACKEND=openai_compatible`. In this
mode, the launcher skips vLLM multimodal startup profiling, caps the expected
request to one 768 px image, and writes all caches under `OUTPUT_DIR` except
for a short vLLM IPC directory under `/tmp` to avoid Unix socket path limits.

```bash
SOURCE_DIR=/absolute/path/to/high_utility_pool_10k/images \
VLM_BACKEND=openai_compatible \
VLM_MAX_IMAGES=128 \
DRY_RUN=1 \
MAX_NUM_SEQS=1 \
bash data_pipeline/high_utility_pool/run_h200_qwen72b_pipeline.sh
```

After the smoke audit succeeds, scale in stages: 128 images, 512 images, 2k
images, then 10k. Do not start the full 10k pass until `audit_report.json`
shows low schema failures and the score distribution is sensible.

```bash
SOURCE_DIR=/absolute/path/to/high_utility_pool_10k/images \
VLM_MAX_IMAGES=10000 \
OUTPUT_DIR=data/high_utility_pool_10k_h200_qwen3vl30b_a3b \
bash data_pipeline/high_utility_pool/run_h200_qwen72b_pipeline.sh
```

The builder stores every judgment in `scores/vlm_scores.jsonl` and writes the
final selected pool plus `audit_report.json` under the requested output
directory. Re-running is resumable because completed VLM rows are read from the
score cache.

If a run is interrupted during VLM scoring, restart with the same `OUTPUT_DIR`.
The builder will reload `scores/vlm_scores.jsonl`, skip candidate IDs that were
already judged, and continue scoring the remaining candidates. If the final
JSONL line was cut off by termination, it is skipped with a warning and all
complete rows are still reused. To resume from a different output directory,
pass `--vlm_cache_path /path/to/old/output/scores/vlm_scores.jsonl`.

## Use With BLIP3o Training

After the pool is built:

```bash
DATA_DIR=$PWD/data/high_utility_pool_10k/images \
  bash scripts/self_evolving/final/E1_main_joint.sh
```

For the paper protocol, the readiness checker expects 6000 images at
`data/joint_6k/images`. To use the curated pool for that path, create a copy or
symlink after verifying the audit report.

```bash
mkdir -p data/joint_6k
ln -sfn ../high_utility_pool_10k/images data/joint_6k/images
```

## Notes

- If the run selects fewer than 10k images, lower `min_heuristic_score`, increase
  `openimages_target`, or add more local sources.
- If disk space is tight, lower `max_output_gb`, keep `link_mode=hardlink`, and
  use fewer VLM/download candidates.
- The pipeline is resumable at the source-cache level. Re-running reuses cached
  Open Images CSVs, downloaded thumbnails, and VLM score cache.
