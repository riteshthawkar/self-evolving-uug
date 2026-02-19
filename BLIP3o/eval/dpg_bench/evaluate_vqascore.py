"""Evaluate DPG-Bench images using VQAScore (alternative to mplug-based evaluation).

VQAScore computes P("Yes" | image, "Does this figure show {text}?") using a VQA model.
This provides a simpler, single-number alignment score per prompt.

Usage:
    python evaluate_vqascore.py \
        --image_dir /path/to/generated_images \
        --csv_path ella_repo/dpg_bench/dpg_bench.csv \
        --model clip-flant5-xxl

Requirements:
    pip install t2v-metrics
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import torch


def load_dpg_prompts(csv_path):
    """Load unique prompts from DPG-Bench CSV."""
    prompts = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["item_id"]
            if item_id not in prompts:
                prompts[item_id] = row["text"]
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with generated images")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to dpg_bench.csv")
    parser.add_argument("--model", type=str, default="clip-flant5-xxl",
                        help="VQAScore model (clip-flant5-xxl or clip-flant5-xl for less VRAM)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.csv_path is None:
        args.csv_path = os.path.join(script_dir, "ella_repo", "dpg_bench", "dpg_bench.csv")
    if args.output is None:
        args.output = os.path.join(args.image_dir, "vqascore_results.json")

    # Load prompts
    prompts = load_dpg_prompts(args.csv_path)
    print(f"Loaded {len(prompts)} unique prompts")

    # Load VQAScore
    import t2v_metrics
    scorer = t2v_metrics.VQAScore(model=args.model)

    # Score each image
    results = {}
    scores_all = []
    missing = 0

    for item_id, text in sorted(prompts.items()):
        img_path = os.path.join(args.image_dir, f"{item_id}.png")
        if not os.path.isfile(img_path):
            missing += 1
            continue

        score = scorer(images=[img_path], texts=[text]).item()
        results[item_id] = {"text": text[:100], "vqascore": score}
        scores_all.append(score)

        if len(scores_all) % 50 == 0:
            avg = sum(scores_all) / len(scores_all)
            print(f"  Processed {len(scores_all)}/{len(prompts)} | Running avg VQAScore: {avg:.4f}")

    # Summary
    if scores_all:
        avg_score = sum(scores_all) / len(scores_all)
        print(f"\n{'='*50}")
        print(f"VQAScore Results")
        print(f"  Model:        {args.model}")
        print(f"  Images:       {len(scores_all)}/{len(prompts)} ({missing} missing)")
        print(f"  Avg VQAScore: {avg_score:.4f}")
        print(f"{'='*50}")

        summary = {
            "model": args.model,
            "num_images": len(scores_all),
            "num_missing": missing,
            "avg_vqascore": avg_score,
            "per_item": results,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        print("ERROR: No images found!")


if __name__ == "__main__":
    main()
