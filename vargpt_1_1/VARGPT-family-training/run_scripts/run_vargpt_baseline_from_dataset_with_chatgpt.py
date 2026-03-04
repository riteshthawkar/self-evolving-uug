#!/usr/bin/env python3
import argparse
import base64
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib import request as urlrequest

import numpy as np
import torch
import torch.nn as nn


def _prepare_imports(train_root: Path) -> None:
    suder_root = train_root.parent
    for p in (train_root, train_root / "src", suder_root):
        p_str = str(p.resolve())
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_dataset_image(dataset_root: Path, seed: int) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_images = [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not all_images:
        raise FileNotFoundError(f"No image files found under dataset root: {dataset_root}")
    rnd = random.Random(seed)
    return rnd.choice(all_images)


def _build_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{suffix};base64,{b64}"


def _ask_chatgpt_for_prompt(image_path: Path, api_key: str, model: str, timeout_sec: int = 120) -> str:
    data_url = _build_data_url(image_path)
    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are helping create a text-to-image prompt from a reference image. "
                            "Write ONE concise prompt (max 35 words) that preserves core subject, scene, "
                            "style, and composition. Output only the prompt text."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    }
    req = urlrequest.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()

    output = body.get("output", [])
    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    raise RuntimeError(f"Could not parse prompt text from OpenAI response: {json.dumps(body)[:1000]}")


def _build_chat_prompt(processor, prompt_text: str) -> str:
    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def _materialize_meta_tensors(module: nn.Module, dtype: torch.dtype) -> int:
    fixed = 0
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is None or not getattr(param, "is_meta", False):
                continue
            t = torch.zeros(param.shape, dtype=dtype, device="cpu")
            sub._parameters[name] = nn.Parameter(t, requires_grad=param.requires_grad)
            fixed += 1
        for name, buf in list(sub._buffers.items()):
            if buf is None or not getattr(buf, "is_meta", False):
                continue
            sub._buffers[name] = torch.zeros(buf.shape, dtype=dtype, device="cpu")
            fixed += 1
    return fixed


def _load_baseline_model(pretrained: str, device: torch.device, dtype: torch.dtype):
    from visionllm.vargpt_qwen_v1_1.modeling_vargpt_qwen2_vl import VARGPTQwen2VLForConditionalGeneration
    from visionllm.vargpt_qwen_v1_1.prepare_vargpt_v1_1 import prepare_vargpt_qwen2vl_v1_1
    from visionllm.vargpt_qwen_v1_1.processing_vargpt_qwen2_vl import VARGPTQwen2VLProcessor
    from patching_utils.patching import patching
    from transformers import AutoTokenizer

    prepare_vargpt_qwen2vl_v1_1(pretrained)

    try:
        model = VARGPTQwen2VLForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        ).eval()
        model = model.to(device).eval()
    except NotImplementedError:
        model = VARGPTQwen2VLForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        ).eval()
        _ = _materialize_meta_tensors(model, dtype=dtype)
        model = model.to(device).eval()

    patching(model)
    processor = VARGPTQwen2VLProcessor.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    return model, processor, tokenizer


def _decode_text(processor, generated_ids: torch.Tensor) -> str:
    return processor.decode(generated_ids[0][:-1], skip_special_tokens=True)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=torch.float32)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick one dataset image, ask ChatGPT for a prompt, run VARGPT baseline generation 4 times."
    )
    parser.add_argument(
        "--train-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to VARGPT-family-training root",
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset image root directory")
    parser.add_argument("--image-path", default="", help="Optional explicit image path; if set, skip random sampling")
    parser.add_argument("--pretrained", default="VARGPT-family/VARGPT-v1.1", help="Baseline VARGPT model path/id")
    parser.add_argument("--openai-model", default="gpt-4.1-mini", help="OpenAI model for prompt generation")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-runs", type=int, default=4, help="How many baseline generations to run")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outdir",
        default="",
        help="Output directory; default: <train_root>/logs/chatgpt_dataset_baseline/<timestamp>",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required in environment.")

    train_root = Path(args.train_root).resolve()
    _prepare_imports(train_root)

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {dataset_root}")

    if args.image_path:
        image_path = Path(args.image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"image-path not found: {image_path}")
    else:
        image_path = _pick_dataset_image(dataset_root, seed=args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir).resolve() if args.outdir else (train_root / "logs" / "chatgpt_dataset_baseline" / ts)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] selected_image={image_path}")
    prompt_text = _ask_chatgpt_for_prompt(
        image_path=image_path,
        api_key=api_key,
        model=args.openai_model,
    )
    print(f"[INFO] chatgpt_prompt={prompt_text}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model, processor, _tokenizer = _load_baseline_model(args.pretrained, device=device, dtype=dtype)
    chat_prompt = _build_chat_prompt(processor, prompt_text)

    meta: Dict[str, Any] = {
        "selected_image": str(image_path),
        "dataset_root": str(dataset_root),
        "pretrained": args.pretrained,
        "openai_model": args.openai_model,
        "chatgpt_prompt": prompt_text,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "outputs": [],
    }

    for i in range(args.num_runs):
        run_seed = args.seed + i
        _set_seed(run_seed)

        inputs = processor(text=chat_prompt, return_tensors="pt")
        inputs = _to_device(inputs, device)
        out_image = outdir / f"gen_{i+1:02d}.png"
        model._IMAGE_GEN_PATH = str(out_image)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
        out_text = _decode_text(processor, output_ids)
        meta["outputs"].append(
            {
                "run_idx": i + 1,
                "seed": run_seed,
                "image_path": str(out_image),
                "decoded_text": out_text,
            }
        )
        print(f"[INFO] run={i+1} saved={out_image}")

    with (outdir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
