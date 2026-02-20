"""Generate images for DPG-Bench evaluation using self-evolving trained BLIP3o model (LoRA + DiT).

DPG-Bench requires 4 images per prompt arranged in a 2x2 grid.
Filenames must match the prompt item_id from dpg_bench.csv.

Usage:
    python generate_dpg_our.py \
        --model /path/to/BLIP3o-Model-8B \
        --checkpoint_dir /path/to/step_00500 \
        --adapter generator \
        --index 0 --n_chunks 8
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
    """Load unique prompts from DPG-Bench CSV."""
    prompts = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["item_id"]
            if item_id not in prompts:
                prompts[item_id] = row["text"]
    return prompts


def load_lora_adapter(model, checkpoint_dir, adapter="generator"):
    """Load and merge a LoRA adapter from a self-evolving checkpoint."""
    from peft import PeftModel

    adapter_path = os.path.join(checkpoint_dir, adapter)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    # Handle nested adapter directories
    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        for subdir in os.listdir(adapter_path):
            candidate = os.path.join(adapter_path, subdir)
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "adapter_config.json")):
                print(f"  Found nested adapter: {adapter}/{subdir}/")
                adapter_path = candidate
                break
        else:
            raise FileNotFoundError(f"adapter_config.json not found in {adapter_path} or subdirectories")

    print(f"Loading LoRA adapter '{adapter}' from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("LoRA adapter merged.")
    return model


def _strip_peft_prefix(name):
    prefix = "base_model.model."
    return name[len(prefix):] if name.startswith(prefix) else name


def load_dit_weights(model, checkpoint_dir):
    """Load updated DiT weights from a self-evolving checkpoint if available."""
    dit_index_path = os.path.join(checkpoint_dir, "dit_trainable_index.json")
    dit_dir = os.path.join(checkpoint_dir, "dit_trainable")

    if not os.path.isfile(dit_index_path) or not os.path.isdir(dit_dir):
        print("No DiT trainable weights found, using base DiT.")
        return model

    with open(dit_index_path, "r") as f:
        payload = json.load(f)

    param_map = payload.get("params", {})
    model_param_names = set(n for n, _ in model.named_parameters())
    loaded, skipped = 0, 0

    for param_name, file_name in param_map.items():
        shard = torch.load(os.path.join(dit_dir, file_name), map_location="cpu")
        target = param_name if param_name in model_param_names else _strip_peft_prefix(param_name)
        if target in model_param_names:
            model.load_state_dict({target: shard}, strict=False)
            loaded += 1
        else:
            skipped += 1

    print(f"DiT weights: {loaded} loaded, {skipped} skipped.")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Base BLIP3o model path")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Self-evolving checkpoint path")
    parser.add_argument("--adapter", type=str, default="generator", help="LoRA adapter to load")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to dpg_bench.csv")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    parser.add_argument("--n_samples", type=int, default=4, help="Images per prompt")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--scale", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)
    return parser.parse_args()


torch.set_grad_enabled(False)


def main():
    opt = parse_args()
    model_name = opt.model
    diffusion_path = model_name + "/diffusion-decoder"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if opt.csv_path is None:
        opt.csv_path = os.path.join(script_dir, "ella_repo", "dpg_bench", "dpg_bench.csv")
    if opt.outdir is None:
        if opt.checkpoint_dir:
            opt.outdir = os.path.join(opt.checkpoint_dir, "dpg_bench_images")
        else:
            opt.outdir = os.path.join(script_dir, "outputs", "our_model")
    os.makedirs(opt.outdir, exist_ok=True)

    # Load prompts
    prompts = load_dpg_prompts(opt.csv_path)
    item_ids = sorted(prompts.keys())
    print(f"Loaded {len(item_ids)} unique prompts from DPG-Bench")

    item_ids = item_ids[opt.index::opt.n_chunks]
    print(f"Chunk {opt.index}/{opt.n_chunks}, {len(item_ids)} prompts.")

    # Load model + LoRA + DiT
    disable_torch_init()
    from diffusers import DiffusionPipeline
    tokenizer, multi_model, _ = load_pretrained_model(model_name)

    if opt.checkpoint_dir:
        multi_model = load_lora_adapter(multi_model, opt.checkpoint_dir, opt.adapter)
        multi_model = load_dit_weights(multi_model, opt.checkpoint_dir)

    # Patch steps
    if opt.steps != 30:
        import functools
        _orig = multi_model.sample_images
        @functools.wraps(_orig)
        def _patched(*args, num_inference_steps=opt.steps, **kwargs):
            return _orig(*args, num_inference_steps=num_inference_steps, **kwargs)
        multi_model.sample_images = _patched

    pipe = DiffusionPipeline.from_pretrained(
        diffusion_path,
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

        # Save as 2x2 grid
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
