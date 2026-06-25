"""Generate images for WISE benchmark using base BLIP3o model.

WISE expects images named 1.png through 1000.png, one per prompt.
Prompts are loaded from the three WISE category JSON files.

Usage:
    python generate_wise_base.py --model /path/to/BLIP3o-Model-8B --index 0 --n_chunks 8
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import trange

from blip3o.constants import *
from blip3o.conversation import conv_templates
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init


def diffusion_pretrained_args(model_name):
    local_decoder = os.path.join(model_name, "diffusion-decoder")
    if os.path.isdir(local_decoder):
        return local_decoder, {}
    return model_name, {"subfolder": "diffusion-decoder"}


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_template(prompt):
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return [conv.get_prompt()]


def load_wise_prompts(wise_data_dir):
    """Load all prompts from WISE JSON files. Returns list of (index, prompt_text)."""
    categories = [
        "cultural_common_sense.json",
        "spatio-temporal_reasoning.json",
        "natural_science.json",
    ]
    all_prompts = []
    for cat_file in categories:
        fpath = os.path.join(wise_data_dir, cat_file)
        if not os.path.isfile(fpath):
            print(f"WARNING: {fpath} not found, skipping")
            continue
        with open(fpath, "r") as f:
            data = json.load(f)
        for item in data:
            idx = item.get("id", item.get("index", len(all_prompts) + 1))
            prompt = item.get("prompt", item.get("text", ""))
            all_prompts.append((int(idx), prompt))
    all_prompts.sort(key=lambda x: x[0])
    return all_prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--wise_data_dir", type=str, default=None,
                        help="Path to WISE data/ directory")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)
    return parser.parse_args()


torch.set_grad_enabled(False)


def main():
    opt = parse_args()
    model_name = opt.model
    diffusion_path, diffusion_kwargs = diffusion_pretrained_args(model_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if opt.wise_data_dir is None:
        opt.wise_data_dir = os.path.join(script_dir, "wise_repo", "data")
    if opt.outdir is None:
        opt.outdir = os.path.join(script_dir, "outputs", "base_model")
    os.makedirs(opt.outdir, exist_ok=True)

    # Load prompts
    prompts = load_wise_prompts(opt.wise_data_dir)
    print(f"Loaded {len(prompts)} WISE prompts")

    # Chunk
    prompts = prompts[opt.index::opt.n_chunks]
    print(f"Chunk {opt.index}/{opt.n_chunks}, {len(prompts)} prompts.")

    # Load model
    disable_torch_init()
    from diffusers import DiffusionPipeline
    tokenizer, multi_model, _ = load_pretrained_model(model_name)

    if opt.steps != 30:
        import functools
        _orig = multi_model.sample_images
        @functools.wraps(_orig)
        def _patched(*args, num_inference_steps=opt.steps, **kwargs):
            return _orig(*args, num_inference_steps=num_inference_steps, **kwargs)
        multi_model.sample_images = _patched

    pipe = DiffusionPipeline.from_pretrained(
        diffusion_path,
        **diffusion_kwargs,
        custom_pipeline="pipeline_llava_gen",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
        multimodal_encoder=multi_model,
        tokenizer=tokenizer,
        safety_checker=None
    )
    pipe.vae.to('cuda:0')
    pipe.unet.to('cuda:0')

    # Generate
    for i, (idx, prompt_text) in enumerate(prompts):
        set_global_seed(opt.seed)
        gen_prompt = [f"Please generate image based on the following caption: {prompt_text}"]
        gen_prompt = add_template(gen_prompt[0])

        print(f"[{i+1}/{len(prompts)}] #{idx}: {prompt_text[:80]}...")

        img = pipe(gen_prompt, guidance_scale=opt.scale).image
        img.save(os.path.join(opt.outdir, f"{idx}.png"))

    print("Done.")


if __name__ == "__main__":
    main()
