"""GenEval generation script for self-evolving trained BLIP3o checkpoints with LoRA adapters.

Usage:
    python generate_our.py \
        --model /path/to/base/BLIP3o-Model-8B \
        --checkpoint_dir /path/to/step_00500 \
        --adapter generator \
        --steps 50 \
        --outdir outputs_our

checkpoint_dir should point to a self-evolving checkpoint directory containing:
    generator/adapter_config.json + adapter_model.*  (LLM conditioning adapter)
    (optionally) dit_lora/ for DiT LoRA, or legacy dit_trainable/ full-DiT shards

Generation pipeline flow:
    text -> LLM (generator LoRA) -> latent queries -> DiT denoising -> latents -> VAE decode -> image
    The generator LoRA adapts the LLM text-to-conditioning path.
    The DiT adapter/weights (if updated via RWR) improve the denoising quality.
"""

import argparse
import json
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from diffusers import DiffusionPipeline, AutoencoderKL
from blip3o.constants import *
from blip3o.conversation import conv_templates, SeparatorStyle
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.train.self_evolving.checkpoint_adapters import prepare_peft_adapter_dir_for_loading
import random


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
    conv.append_message(conv.roles[0], prompt[0])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return [prompt]


def load_lora_adapter(model, checkpoint_dir, adapter="generator"):
    """Load and merge a LoRA adapter from a self-evolving checkpoint.

    For generation eval, the 'generator' adapter is the correct choice — it adapts
    the LLM's text-to-conditioning path (embed_tokens -> latent_queries -> forward).
    The 'solver' adapter is for understanding (question answering).
    """
    from peft import PeftModel

    adapter_path = os.path.join(checkpoint_dir, adapter)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}. "
            f"Expected checkpoint_dir to contain a '{adapter}/' subdirectory."
        )

    # PEFT's save_pretrained with selected_adapters creates a nested subdirectory
    # named after the PEFT adapter name (not the folder name). For example:
    #   solver/default/adapter_config.json   (PEFT name "default" saved into solver/)
    #   proposer/proposer/adapter_config.json
    #   generator/generator/adapter_config.json
    # Search for adapter_config.json at the top level or any immediate subdirectory.
    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        found = False
        for subdir in os.listdir(adapter_path):
            candidate = os.path.join(adapter_path, subdir)
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "adapter_config.json")):
                print(f"  Found nested adapter directory: {adapter}/{subdir}/")
                adapter_path = candidate
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"adapter_config.json not found in {adapter_path} or any subdirectory."
            )

    adapter_path = str(prepare_peft_adapter_dir_for_loading(adapter_path, log=print))
    print(f"Loading LoRA adapter '{adapter}' from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Merging LoRA adapter into base model")
    model = model.merge_and_unload()
    return model


def _strip_peft_prefix(name: str) -> str:
    """Strip the 'base_model.model.' prefix that PeftModel adds to parameter names.

    During training, the model is wrapped in PeftModel which prepends 'base_model.model.'
    to all parameter names. Checkpoints save these prefixed names. But during eval, after
    merge_and_unload(), the model is a plain model without this prefix. This function
    strips the prefix so load_state_dict can match parameters correctly.
    """
    prefix = "base_model.model."
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _find_adapter_config_dir(adapter_root: str) -> str:
    if os.path.isfile(os.path.join(adapter_root, "adapter_config.json")):
        return adapter_root
    for subdir in os.listdir(adapter_root):
        candidate = os.path.join(adapter_root, subdir)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            print(f"  Found nested DiT adapter directory: dit_lora/{subdir}/")
            return candidate
    raise FileNotFoundError(f"adapter_config.json not found under {adapter_root}")


def _load_dit_lora_adapter(model, checkpoint_dir) -> bool:
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

    # The PEFT wrapper checks is_gradient_checkpointing and, if true, prepares
    # inputs through get_input_embeddings(). That is valid for LM modules but
    # not for NextDiTCrossAttn, so disable GC flags for inference-time DiT LoRA
    # loading before wrapping the module.
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
    print(f"Loading DiT LoRA adapter from {adapter_path}")
    dit_model = PeftModel.from_pretrained(dit_module, adapter_path)
    try:
        core_model.dit = dit_model.merge_and_unload()
        print("DiT LoRA adapter merged into DiT.")
    except Exception as exc:
        core_model.dit = dit_model
        print(f"WARNING: DiT LoRA merge failed; keeping PEFT-wrapped DiT. Reason: {exc}")
    return True


def load_dit_weights(model, checkpoint_dir):
    """Load updated DiT weights from a self-evolving checkpoint if available."""
    if _load_dit_lora_adapter(model, checkpoint_dir):
        return model

    dit_index_path = os.path.join(checkpoint_dir, "dit_trainable_index.json")
    dit_dir = os.path.join(checkpoint_dir, "dit_trainable")

    if not os.path.isfile(dit_index_path) or not os.path.isdir(dit_dir):
        print("No DiT trainable weights found in checkpoint, using base DiT weights.")
        return model

    with open(dit_index_path, "r") as f:
        payload = json.load(f)

    param_map = payload.get("params", {})
    print(f"Loading {len(param_map)} updated DiT parameters from {dit_dir}")

    # Build a mapping of model's actual parameter names for verification
    model_param_names = set(name for name, _ in model.named_parameters())
    loaded, skipped = 0, 0

    for param_name, file_name in param_map.items():
        shard_path = os.path.join(dit_dir, file_name)
        shard = torch.load(shard_path, map_location="cpu")

        # Try original name first, then stripped name
        # (checkpoint saves PeftModel-prefixed names like 'base_model.model.model.dit...')
        if param_name in model_param_names:
            target_name = param_name
        else:
            target_name = _strip_peft_prefix(param_name)

        if target_name in model_param_names:
            model.load_state_dict({target_name: shard}, strict=False)
            loaded += 1
        else:
            print(f"  WARNING: DiT param not found in model: {param_name} (also tried: {target_name})")
            skipped += 1

    print(f"DiT weights loaded: {loaded} params loaded, {skipped} skipped.")
    return model


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base BLIP3o model path (HuggingFace name or local path)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to self-evolving training checkpoint directory (e.g. step_00500/)"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="generator",
        help="Which LoRA adapter to load for generation: generator (default, LLM conditioning), solver, or proposer"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="qwen",
        help="Template format"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="GenEval prompt JSONL. Defaults to geneval_prompt.jsonl next to this script.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument("--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks")
    opt = parser.parse_args()
    return opt


def main(opt):
    model_name = opt.model
    diffusion_path, diffusion_kwargs = diffusion_pretrained_args(model_name)

    outdir = opt.outdir
    if outdir == "outputs":
        # Default: put in checkpoint-specific directory
        if opt.checkpoint_dir:
            ckpt_name = os.path.basename(os.path.normpath(opt.checkpoint_dir))
            outdir = f"{opt.checkpoint_dir}/geneval_{opt.prompt_template}"
        else:
            outdir = f"{model_name}/geneval_{opt.prompt_template}"
    os.makedirs(outdir, exist_ok=True)
    prompt_template = opt.prompt_template
    disable_torch_init()

    # Load base model
    tokenizer, multi_model, context_len = load_pretrained_model(model_name)

    # Apply LoRA adapter if checkpoint provided
    if opt.checkpoint_dir:
        multi_model = load_lora_adapter(multi_model, opt.checkpoint_dir, opt.adapter)
        multi_model = load_dit_weights(multi_model, opt.checkpoint_dir)

    # Patch num_inference_steps in sample_images if --steps differs from default (30).
    # generate_image() -> sample_images() hardcodes num_inference_steps=30,
    # and the pipeline doesn't thread this parameter through.
    if opt.steps != 30:
        import functools
        _orig_sample_images = multi_model.sample_images
        @functools.wraps(_orig_sample_images)
        def _patched_sample_images(*args, num_inference_steps=opt.steps, **kwargs):
            return _orig_sample_images(*args, num_inference_steps=num_inference_steps, **kwargs)
        multi_model.sample_images = _patched_sample_images
        print(f"Patched sample_images to use num_inference_steps={opt.steps}")

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

    # Load all prompts. Keep this independent of the caller's working directory.
    prompt_file = opt.prompt_file
    if prompt_file is None:
        prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geneval_prompt.jsonl")
    if not os.path.isfile(prompt_file):
        raise FileNotFoundError(
            f"GenEval prompt file not found: {prompt_file}. "
            "Pass --prompt_file or set PROMPT_FILE in generation_our.sh."
        )
    with open(prompt_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]

    # Split the data into chunks
    metadatas = metadatas[opt.index::opt.n_chunks]
    print(f"Processing chunk {opt.index} out of {opt.n_chunks} total chunks, {len(metadatas)} samples assigned.")

    for index, metadata in enumerate(metadatas):
        set_global_seed(seed=42)
        outpath = os.path.join(outdir, f"{metadata['index']}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']

        prompt = [f"Please generate image based on the following caption: {prompt}"]
        if "qwen" in prompt_template:
            prompt = add_template(prompt)
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        batch_size = opt.batch_size
        n_rows = opt.batch_size
        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):

                gen_img = pipe(prompt, guidance_scale=opt.scale).image

                samples = [gen_img]
                for sample in samples:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not opt.skip_grid:
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

    print("Done.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
