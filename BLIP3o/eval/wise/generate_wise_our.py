"""Generate images for WISE benchmark using self-evolving trained BLIP3o (LoRA + DiT).

Usage:
    python generate_wise_our.py \
        --model /path/to/BLIP3o-Model-8B \
        --checkpoint_dir /path/to/step_00500 \
        --adapter generator \
        --index 0 --n_chunks 8
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from blip3o.constants import *
from blip3o.conversation import conv_templates
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.train.self_evolving.checkpoint_adapters import prepare_peft_adapter_dir_for_loading


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
    """Load all prompts from WISE JSON files."""
    categories = [
        "cultural_common_sense.json",
        "spatio-temporal_reasoning.json",
        "natural_science.json",
    ]
    all_prompts = []
    for cat_file in categories:
        fpath = os.path.join(wise_data_dir, cat_file)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r") as f:
            data = json.load(f)
        for item in data:
            idx = item.get("id", item.get("index", len(all_prompts) + 1))
            prompt = item.get("prompt", item.get("text", ""))
            all_prompts.append((int(idx), prompt))
    all_prompts.sort(key=lambda x: x[0])
    return all_prompts


def load_lora_adapter(model, checkpoint_dir, adapter="generator"):
    from peft import PeftModel
    adapter_path = os.path.join(checkpoint_dir, adapter)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        for subdir in os.listdir(adapter_path):
            candidate = os.path.join(adapter_path, subdir)
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "adapter_config.json")):
                adapter_path = candidate
                break
        else:
            raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    adapter_path = str(prepare_peft_adapter_dir_for_loading(adapter_path, log=print))
    print(f"Loading LoRA '{adapter}' from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    return model


def _strip_peft_prefix(name):
    prefix = "base_model.model."
    return name[len(prefix):] if name.startswith(prefix) else name


def _find_adapter_config_dir(adapter_root):
    if os.path.isfile(os.path.join(adapter_root, "adapter_config.json")):
        return adapter_root
    for subdir in os.listdir(adapter_root):
        candidate = os.path.join(adapter_root, subdir)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            print(f"  Found nested DiT adapter: dit_lora/{subdir}/")
            return candidate
    raise FileNotFoundError(f"adapter_config.json not found under {adapter_root}")


def _load_dit_lora_adapter(model, checkpoint_dir):
    dit_lora_root = os.path.join(checkpoint_dir, "dit_lora")
    if not os.path.isdir(dit_lora_root):
        return False
    from peft import PeftModel

    adapter_path = _find_adapter_config_dir(dit_lora_root)
    core_model_getter = getattr(model, "get_model", None)
    core_model = core_model_getter() if callable(core_model_getter) else getattr(model, "model", None)
    dit_module = getattr(core_model, "dit", None) if core_model is not None else None
    if dit_module is None:
        raise RuntimeError("Checkpoint contains dit_lora/, but the loaded model has no core_model.dit module.")
    if hasattr(dit_module, "gradient_checkpointing_disable"):
        try:
            dit_module.gradient_checkpointing_disable()
        except Exception as exc:
            print(f"WARNING: failed to disable DiT gradient checkpointing via helper: {exc}")
    for submodule in dit_module.modules():
        if hasattr(submodule, "gradient_checkpointing"):
            submodule.gradient_checkpointing = False
    if hasattr(dit_module, "config") and hasattr(dit_module.config, "_gradient_checkpointing"):
        dit_module.config._gradient_checkpointing = False
    adapter_path = str(prepare_peft_adapter_dir_for_loading(adapter_path, log=print))
    print(f"Loading DiT LoRA from {adapter_path}")
    dit_model = PeftModel.from_pretrained(dit_module, adapter_path)
    try:
        core_model.dit = dit_model.merge_and_unload()
        print("DiT LoRA merged.")
    except Exception as exc:
        core_model.dit = dit_model
        print(f"WARNING: DiT LoRA merge failed; keeping PEFT-wrapped DiT. Reason: {exc}")
    return True


def load_dit_weights(model, checkpoint_dir):
    if _load_dit_lora_adapter(model, checkpoint_dir):
        return model

    dit_index_path = os.path.join(checkpoint_dir, "dit_trainable_index.json")
    dit_dir = os.path.join(checkpoint_dir, "dit_trainable")
    if not os.path.isfile(dit_index_path) or not os.path.isdir(dit_dir):
        print("No DiT trainable weights found, using base.")
        return model
    with open(dit_index_path, "r") as f:
        payload = json.load(f)
    model_param_names = set(n for n, _ in model.named_parameters())
    loaded = 0
    for param_name, file_name in payload.get("params", {}).items():
        shard = torch.load(os.path.join(dit_dir, file_name), map_location="cpu")
        target = param_name if param_name in model_param_names else _strip_peft_prefix(param_name)
        if target in model_param_names:
            model.load_state_dict({target: shard}, strict=False)
            loaded += 1
    print(f"DiT: {loaded} params loaded.")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--adapter", type=str, default="generator")
    parser.add_argument("--wise_data_dir", type=str, default=None)
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
        if opt.checkpoint_dir:
            opt.outdir = os.path.join(opt.checkpoint_dir, "wise_images")
        else:
            opt.outdir = os.path.join(script_dir, "outputs", "our_model")
    os.makedirs(opt.outdir, exist_ok=True)

    prompts = load_wise_prompts(opt.wise_data_dir)
    print(f"Loaded {len(prompts)} WISE prompts")
    prompts = prompts[opt.index::opt.n_chunks]
    print(f"Chunk {opt.index}/{opt.n_chunks}, {len(prompts)} prompts.")

    # Load model + adapters
    disable_torch_init()
    from diffusers import DiffusionPipeline
    tokenizer, multi_model, _ = load_pretrained_model(model_name)

    if opt.checkpoint_dir:
        multi_model = load_lora_adapter(multi_model, opt.checkpoint_dir, opt.adapter)
        multi_model = load_dit_weights(multi_model, opt.checkpoint_dir)

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
