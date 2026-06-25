"""Generate images for DPG-Bench evaluation using base BLIP3o model.

DPG-Bench requires 4 images per prompt arranged in a 2x2 grid.
Filenames must match the prompt item_id from dpg_bench.csv.

Usage:
    python generate_dpg_base.py --model /path/to/BLIP3o-Model-8B --index 0 --n_chunks 8
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import trange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor

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


def load_dpg_prompts(csv_path):
    """Load unique prompts from DPG-Bench CSV. Returns dict: item_id -> prompt_text."""
    prompts = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["item_id"]
            if item_id not in prompts:
                prompts[item_id] = row["text"]
    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Base BLIP3o model path")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to dpg_bench.csv (default: ella_repo/dpg_bench/dpg_bench.csv)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for generated images")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of images per prompt")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--scale", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=int, default=0, help="Chunk index")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total chunks")
    return parser.parse_args()


torch.set_grad_enabled(False)


def main():
    opt = parse_args()
    model_name = opt.model
    diffusion_path, diffusion_kwargs = diffusion_pretrained_args(model_name)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if opt.csv_path is None:
        opt.csv_path = os.path.join(script_dir, "ella_repo", "dpg_bench", "dpg_bench.csv")
    if opt.outdir is None:
        opt.outdir = os.path.join(script_dir, "outputs", "base_model")
    os.makedirs(opt.outdir, exist_ok=True)

    # Load prompts
    prompts = load_dpg_prompts(opt.csv_path)
    item_ids = sorted(prompts.keys())
    print(f"Loaded {len(item_ids)} unique prompts from DPG-Bench")

    # Chunk
    item_ids = item_ids[opt.index::opt.n_chunks]
    print(f"Processing chunk {opt.index}/{opt.n_chunks}, {len(item_ids)} prompts assigned.")

    # Load model
    disable_torch_init()
    from diffusers import DiffusionPipeline
    tokenizer, multi_model, _ = load_pretrained_model(model_name)

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
    device_id = 0
    pipe.vae.to(f'cuda:{device_id}')
    pipe.unet.to(f'cuda:{device_id}')

    # Patch steps if needed
    if opt.steps != 30:
        import functools
        _orig = multi_model.sample_images
        @functools.wraps(_orig)
        def _patched(*args, num_inference_steps=opt.steps, **kwargs):
            return _orig(*args, num_inference_steps=num_inference_steps, **kwargs)
        multi_model.sample_images = _patched

    # Generate
    for idx, item_id in enumerate(item_ids):
        set_global_seed(opt.seed)
        prompt_text = prompts[item_id]
        gen_prompt = [f"Please generate image based on the following caption: {prompt_text}"]
        gen_prompt = add_template(gen_prompt[0])

        print(f"[{idx+1}/{len(item_ids)}] {item_id}: {prompt_text[:80]}...")

        all_imgs = []
        for _ in range(opt.n_samples):
            img = pipe(gen_prompt, guidance_scale=opt.scale).image
            all_imgs.append(img)

        # Save as 2x2 grid (DPG-Bench expects this)
        # padding=0 is critical — DPG-Bench crops at exact (0,0,res,res) boundaries
        tensors = [ToTensor()(img) for img in all_imgs]
        grid = make_grid(torch.stack(tensors), nrow=2, padding=0)
        grid_img = Image.fromarray(
            (255.0 * grid.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
        )
        grid_img.save(os.path.join(opt.outdir, f"{item_id}.png"))

    print("Done.")


if __name__ == "__main__":
    main()
