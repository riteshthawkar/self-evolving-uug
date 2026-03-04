#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_imports(train_root: Path) -> None:
    suder_root = train_root.parent
    for p in (train_root, train_root / "src", suder_root):
        p_str = str(p.resolve())
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _build_chat(processor, prompt: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


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


def _set_image_path_attr(model: Any, image_path: str) -> None:
    seen = set()
    stack = [model]
    while stack:
        cur = stack.pop()
        obj_id = id(cur)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        try:
            setattr(cur, "_IMAGE_GEN_PATH", image_path)
        except Exception:
            pass
        for attr in ("model", "base_model", "module"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                stack.append(nxt)
        getter = getattr(cur, "get_base_model", None)
        if callable(getter):
            try:
                nxt = getter()
                if nxt is not None and nxt is not cur:
                    stack.append(nxt)
            except Exception:
                pass


def _materialize_meta_tensors(module: nn.Module, dtype: torch.dtype) -> int:
    """
    Replace meta parameters/buffers with real CPU tensors so `.to(device)` can succeed.
    Returns the count of materialized tensors.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GenEval images with VARGPT HF model (+optional LoRA).")
    parser.add_argument("--train_root", required=True, help="Path to VARGPT-family-training root.")
    parser.add_argument("--pretrained", required=True, help="Base VARGPT model path or HF id.")
    parser.add_argument("--peft", default="", help="Optional PEFT adapter path.")
    parser.add_argument("--peft_adapter_name", default="", help="Adapter name to activate (optional).")
    parser.add_argument("--metadata_file", required=True, help="GenEval metadata jsonl file.")
    parser.add_argument("--outdir", required=True, help="Output directory in GenEval expected format.")
    parser.add_argument("--n_samples", type=int, default=4, help="Samples per prompt.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Generation tokens.")
    parser.add_argument("--do_sample", type=int, default=1, choices=[0, 1], help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    args = parser.parse_args()

    train_root = Path(args.train_root).resolve()
    _prepare_imports(train_root)

    from patching_utils.patching import patching
    from visionllm.vargpt_qwen_v1_1.modeling_vargpt_qwen2_vl import VARGPTQwen2VLForConditionalGeneration
    from visionllm.vargpt_qwen_v1_1.prepare_vargpt_v1_1 import prepare_vargpt_qwen2vl_v1_1
    from visionllm.vargpt_qwen_v1_1.processing_vargpt_qwen2_vl import VARGPTQwen2VLProcessor

    if not os.path.isfile(args.metadata_file):
        raise FileNotFoundError(f"metadata_file not found: {args.metadata_file}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.metadata_file, "r", encoding="utf-8") as f:
        metadatas: List[Dict[str, Any]] = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    prepare_vargpt_qwen2vl_v1_1(args.pretrained)

    def _load_base_model():
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": model_dtype,
            # Avoid meta-tensor init path that later fails on `.to(device)`.
            "low_cpu_mem_usage": False,
        }
        m = VARGPTQwen2VLForConditionalGeneration.from_pretrained(
            args.pretrained,
            **load_kwargs,
        ).eval()
        return m

    try:
        model = _load_base_model()
        model = model.to(device).eval()
    except NotImplementedError as exc:
        # Some checkpoints leave a subset of tensors as meta; materialize then retry.
        print(
            f"[WARN] Direct model.to({device}) failed ({exc}). "
            "Materializing meta tensors on CPU and retrying."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = _load_base_model()
        fixed = _materialize_meta_tensors(model, dtype=model_dtype)
        print(f"[WARN] Materialized meta tensors: {fixed}")
        model = model.to(device).eval()

    patching(model)

    if args.peft:
        if PeftModel is None:
            raise ImportError("peft is required to load --peft adapter.")
        model = PeftModel.from_pretrained(model, args.peft, is_trainable=False)
        if args.peft_adapter_name:
            model.set_adapter(args.peft_adapter_name)
        model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    processor = VARGPTQwen2VLProcessor.from_pretrained(args.pretrained)
    _ = tokenizer  # keep for parity/debug, tokenizer is not directly used below.

    for idx, metadata in enumerate(metadatas):
        prompt = str(metadata.get("prompt", "")).strip()
        if not prompt:
            continue

        sample_root = Path(args.outdir) / f"{idx:05d}"
        sample_dir = sample_root / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        with open(sample_root / "metadata.jsonl", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        chat_prompt = _build_chat(processor, prompt)

        for sample_idx in range(args.n_samples):
            sample_seed = int(args.seed) + idx * 100000 + sample_idx
            _set_seed(sample_seed)

            batch = processor(text=chat_prompt, return_tensors="pt")
            batch = _to_device(batch, device)

            out_path = str(sample_dir / f"{sample_idx:05d}.jpg")
            _set_image_path_attr(model, out_path)

            with torch.inference_mode():
                model.generate(
                    **batch,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )

            if not os.path.isfile(out_path):
                raise RuntimeError(
                    f"Image generation failed for prompt_idx={idx}, sample_idx={sample_idx}; "
                    f"expected file missing: {out_path}"
                )


if __name__ == "__main__":
    main()
