#!/usr/bin/env python3
import argparse
import base64
import gc
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import request as urlrequest

import numpy as np
import torch
import torch.nn as nn

TRIPLET_PROMPTS: List[Dict[str, str]] = [
    {
        "category": "counting",
        "prompt": "A simple studio scene with exactly four fruits on a white plate: three red apples and one green pear.",
    },
    {
        "category": "position",
        "prompt": "A blue cube is to the left of a yellow sphere, and a red cone is behind the yellow sphere.",
    },
    {
        "category": "color_attribution",
        "prompt": "On a wooden desk, place a red mug and a blue book; the mug must be red and the book must be blue.",
    },
]

TRIPLET_BEST_SETTINGS: Dict[str, Tuple[bool, float, float]] = {
    "counting": (False, 1.0, 1.0),
    "position": (True, 0.35, 0.85),
    "color_attribution": (False, 1.0, 1.0),
}


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

    model = VARGPTQwen2VLForConditionalGeneration.from_pretrained(
        pretrained,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).eval()
    try:
        model = model.to(device).eval()
    except NotImplementedError as exc:
        fixed = _materialize_meta_tensors(model, dtype=dtype)
        print(
            "[WARN] model.to(device) failed due to meta tensors; "
            f"materialized {fixed} tensors and retrying."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = model.to(device).eval()
    except RuntimeError:
        # Propagate OOM and other runtime failures, but ensure we free temp refs first.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    patching(model)
    processor = VARGPTQwen2VLProcessor.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    return model, processor, tokenizer


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


def _decode_text(processor, generated_ids: torch.Tensor) -> str:
    return processor.decode(generated_ids[0][:-1], skip_special_tokens=True)


def _is_dtype_mismatch_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "mat1 and mat2 must have the same dtype" in msg
        or "mat1 and mat2 must have same dtype" in msg
        or ("mat1" in msg and "mat2" in msg and "dtype" in msg)
    )


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


def _debug_mtime_map(debug_outdir: Path) -> Dict[Path, int]:
    return {pp: pp.stat().st_mtime_ns for pp in debug_outdir.rglob("*.png")}


def _try_copy_changed_debug_image(
    debug_outdir: Path,
    before_mtime: Dict[Path, int],
    out_image: Path,
) -> bool:
    changed_debug = []
    for pp in debug_outdir.rglob("*.png"):
        try:
            cur_mtime = pp.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        prev_mtime = before_mtime.get(pp)
        if prev_mtime is None or cur_mtime > prev_mtime:
            changed_debug.append((cur_mtime, pp))
    if not changed_debug:
        return False
    _, newest = max(changed_debug, key=lambda x: x[0])
    shutil.copy2(newest, out_image)
    return out_image.exists()


def _save_generated_image_tensor(image_tensor: torch.Tensor, out_image: Path) -> bool:
    if image_tensor is None:
        return False
    try:
        arr = image_tensor.detach().cpu().numpy()
    except Exception:
        return False

    # Expect HxWxC; squeeze batch if needed.
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return False

    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr)
        if arr.max(initial=0.0) <= 1.0 and arr.min(initial=0.0) >= 0.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    try:
        import cv2

        cv2.imwrite(str(out_image), arr)
    except Exception:
        try:
            from PIL import Image

            Image.fromarray(arr).save(out_image)
        except Exception:
            return False
    return out_image.exists()


def _run_schedule(mode: str, num_runs: int, do_sample: int, temperature: float, top_p: float) -> List[Tuple[bool, float, float]]:
    if mode == "sweep":
        presets: List[Tuple[bool, float, float]] = [
            (False, 1.0, 1.0),   # deterministic / adherence-heavy
            (True, 0.7, 0.85),   # balanced
            (True, 1.1, 0.92),   # diverse
            (True, 1.6, 1.0),    # stress diversity
        ]
        if num_runs <= len(presets):
            return presets[:num_runs]
        out = presets[:]
        while len(out) < num_runs:
            out.append(presets[(len(out) - len(presets)) % len(presets)])
        return out
    return [(bool(do_sample), float(temperature), float(top_p)) for _ in range(num_runs)]


def _format_gib(nbytes: int) -> str:
    return f"{(float(nbytes) / (1024 ** 3)):.2f} GiB"


def _pick_freest_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    best_idx = 0
    best_free = -1
    for i in range(count):
        try:
            free_b, total_b = torch.cuda.mem_get_info(i)
        except Exception:
            free_b, total_b = (0, 0)
        print(f"[INFO] cuda:{i} free={_format_gib(free_b)} total={_format_gib(total_b)}")
        if free_b > best_free:
            best_free = free_b
            best_idx = i
    if best_free < (8 * 1024 ** 3):
        raise RuntimeError(
            "No CUDA device has enough free memory for VARGPT baseline inference "
            f"(best free={_format_gib(best_free)}). Free up GPU memory or pick another node."
        )
    print(f"[INFO] Selected device cuda:{best_idx} (max free memory).")
    return torch.device(f"cuda:{best_idx}")


def _resolve_device(device_arg: str, auto_select_gpu: bool) -> torch.device:
    # If user passed explicit cuda index (e.g., cuda:3), honor it.
    if device_arg.startswith("cuda:"):
        return torch.device(device_arg if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return _pick_freest_cuda_device() if auto_select_gpu else torch.device("cuda:0")
    return torch.device(device_arg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VARGPT baseline generation using prompt modes: triplet, chatgpt_image, or fixed."
    )
    parser.add_argument(
        "--train-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to VARGPT-family-training root",
    )
    parser.add_argument("--dataset-root", default="", help="Dataset image root directory (required for chatgpt_image mode)")
    parser.add_argument("--image-path", default="", help="Optional explicit image path; if set, skip random sampling")
    parser.add_argument("--pretrained", default="VARGPT-family/VARGPT-v1.1", help="Baseline VARGPT model path/id")
    parser.add_argument("--openai-model", default="gpt-4.1-mini", help="OpenAI model for prompt generation")
    parser.add_argument(
        "--prompt-mode",
        default="triplet",
        choices=["triplet", "chatgpt_image", "fixed"],
        help="triplet: use fixed 3 benchmark prompts; chatgpt_image: prompt from sampled image; fixed: use --prompt-text",
    )
    parser.add_argument("--prompt-text", default="", help="If set, skip ChatGPT and use this exact prompt text")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--auto-select-gpu",
        type=int,
        default=1,
        choices=[0, 1],
        help="When --device cuda, pick the freest GPU automatically.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-runs", type=int, default=4, help="How many baseline generations to run")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--run-mode",
        default="sweep",
        choices=["sweep", "fixed"],
        help="sweep: 4-profile comparison settings; fixed: same decoding params every run",
    )
    parser.add_argument(
        "--triplet-best-profile",
        type=int,
        default=1,
        choices=[0, 1],
        help="In triplet mode, use one tuned decode profile per category and emit one best image per category.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outdir",
        default="",
        help="Output directory; default: <train_root>/logs/chatgpt_dataset_baseline/<timestamp>",
    )
    args = parser.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_root = Path(args.train_root).resolve()
    _prepare_imports(train_root)

    image_path: Path = None
    dataset_root: Path = None
    prompt_entries: List[Dict[str, str]] = []

    if args.prompt_mode == "triplet":
        prompt_entries = TRIPLET_PROMPTS
    elif args.prompt_mode == "fixed":
        if not args.prompt_text.strip():
            raise ValueError("--prompt-text is required when --prompt-mode fixed")
        prompt_entries = [{"category": "fixed", "prompt": args.prompt_text.strip()}]
    else:
        if not args.dataset_root:
            raise ValueError("--dataset-root is required when --prompt-mode chatgpt_image")
        dataset_root = Path(args.dataset_root).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"dataset-root not found: {dataset_root}")
        if args.image_path:
            image_path = Path(args.image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"image-path not found: {image_path}")
        else:
            image_path = _pick_dataset_image(dataset_root, seed=args.seed)
        print(f"[INFO] selected_image={image_path}")
        if args.prompt_text.strip():
            prompt_entries = [{"category": "chatgpt_image", "prompt": args.prompt_text.strip()}]
            print("[INFO] using provided --prompt-text (ChatGPT skipped)")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is required when --prompt-text is not provided.")
            generated_prompt = _ask_chatgpt_for_prompt(
                image_path=image_path,
                api_key=api_key,
                model=args.openai_model,
            )
            prompt_entries = [{"category": "chatgpt_image", "prompt": generated_prompt}]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir).resolve() if args.outdir else (train_root / "logs" / "chatgpt_dataset_baseline" / ts)
    outdir.mkdir(parents=True, exist_ok=True)
    debug_outdir = outdir / "_raw_debug"
    debug_outdir.mkdir(parents=True, exist_ok=True)

    # VARGPT model file reads these env vars at import time.
    os.environ["VARGPT_SAVE_DEBUG_IMAGES"] = "1"
    os.environ["_OUTPUT_IMAGE_PATH"] = str(debug_outdir)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = _resolve_device(args.device, auto_select_gpu=bool(args.auto_select_gpu))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, processor, _tokenizer = _load_baseline_model(args.pretrained, device=device, dtype=dtype)

    meta: Dict[str, Any] = {
        "selected_image": str(image_path) if image_path is not None else None,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "pretrained": args.pretrained,
        "openai_model": args.openai_model,
        "prompt_mode": args.prompt_mode,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "run_mode": args.run_mode,
        "debug_outdir": str(debug_outdir),
        "outputs": [],
    }

    did_float32_fallback = False
    for p_idx, p in enumerate(prompt_entries, start=1):
        category = p["category"]
        prompt_text = p["prompt"]
        print(f"[INFO] prompt[{p_idx}/{len(prompt_entries)}] category={category}")
        print(f"[INFO] prompt_text={prompt_text}")
        chat_prompt = _build_chat_prompt(processor, prompt_text)

        if args.prompt_mode == "triplet" and args.triplet_best_profile:
            runs_for_prompt = 1
            do_sample_i, temp_i, top_p_i = TRIPLET_BEST_SETTINGS.get(category, (False, 0.0, 1.0))
            schedule = [(do_sample_i, temp_i, top_p_i)]
        else:
            runs_for_prompt = args.num_runs
            schedule = _run_schedule(
                mode=args.run_mode,
                num_runs=runs_for_prompt,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        prompt_outdir = outdir if len(prompt_entries) == 1 else (outdir / f"{p_idx:02d}_{category}")
        prompt_outdir.mkdir(parents=True, exist_ok=True)

        for i in range(runs_for_prompt):
            run_seed = args.seed + (p_idx * 1000) + i
            _set_seed(run_seed)

            inputs = processor(text=chat_prompt, return_tensors="pt")
            inputs = _to_device(inputs, device)
            if runs_for_prompt == 1:
                out_image = prompt_outdir / "best.png"
            else:
                out_image = prompt_outdir / f"gen_{i+1:02d}.png"

            do_sample_i, temp_i, top_p_i = schedule[i]
            output_ids = None
            out_text = ""
            image_saved = False

            # Primary decode + conservative fallbacks if image token is not triggered.
            attempts: List[Tuple[bool, float, float]] = [(bool(do_sample_i), float(temp_i), float(top_p_i))]
            if attempts[0] != (False, 1.0, 1.0):
                attempts.append((False, 1.0, 1.0))
            if attempts[0] != (True, 0.9, 0.95):
                attempts.append((True, 0.9, 0.95))

            for attempt_idx, (attempt_do_sample, attempt_temp, attempt_top_p) in enumerate(attempts):
                _set_seed(run_seed + attempt_idx)
                before_debug_mtime = _debug_mtime_map(debug_outdir)
                try:
                    with torch.inference_mode():
                        _set_image_path_attr(model, str(out_image))
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=attempt_do_sample,
                            temperature=attempt_temp,
                            top_p=attempt_top_p,
                        )
                except RuntimeError as exc:
                    if _is_dtype_mismatch_error(exc) and not did_float32_fallback:
                        print(
                            "[WARN] Generation hit dtype mismatch. "
                            "Promoting model to float32 and retrying once."
                        )
                        model = model.float().to(device).eval()
                        did_float32_fallback = True
                        with torch.inference_mode():
                            _set_image_path_attr(model, str(out_image))
                            output_ids = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=attempt_do_sample,
                                temperature=attempt_temp,
                                top_p=attempt_top_p,
                            )
                    else:
                        raise

                if out_image.exists() or _try_copy_changed_debug_image(debug_outdir, before_debug_mtime, out_image):
                    image_saved = True
                    break
                print(
                    f"[WARN] No image artifact after attempt {attempt_idx + 1} "
                    f"(do_sample={int(attempt_do_sample)}, temp={attempt_temp}, top_p={attempt_top_p})."
                )

            # Final fallback: call forward with inference_image_gen=True directly.
            if not image_saved:
                before_debug_mtime = _debug_mtime_map(debug_outdir)
                try:
                    with torch.inference_mode():
                        _set_image_path_attr(model, str(out_image))
                        direct_out = model(**inputs, return_dict=True, inference_image_gen=True)
                    if not out_image.exists():
                        saved_from_tensor = _save_generated_image_tensor(
                            getattr(direct_out, "generated_image", None), out_image
                        )
                        image_saved = bool(saved_from_tensor)
                    if not image_saved:
                        image_saved = _try_copy_changed_debug_image(debug_outdir, before_debug_mtime, out_image)
                except Exception as exc:
                    print(f"[WARN] Direct inference_image_gen fallback failed: {exc}")

            if output_ids is not None:
                out_text = _decode_text(processor, output_ids)

            if not image_saved:
                err_msg = (
                    f"No generated image file found for category={category}, run={i+1}. "
                    f"Tried decode fallbacks and direct inference_image_gen. Debug dir: {debug_outdir}"
                )
                print(f"[ERROR] {err_msg}")
                meta["outputs"].append(
                    {
                        "prompt_index": p_idx,
                        "category": category,
                        "prompt_text": prompt_text,
                        "run_idx": i + 1,
                        "seed": run_seed,
                        "image_path": None,
                        "decoded_text": out_text,
                        "do_sample": bool(do_sample_i),
                        "temperature": float(temp_i),
                        "top_p": float(top_p_i),
                        "status": "failed_no_image_artifact",
                        "error": err_msg,
                    }
                )
                continue

            meta["outputs"].append(
                {
                    "prompt_index": p_idx,
                    "category": category,
                    "prompt_text": prompt_text,
                    "run_idx": i + 1,
                    "seed": run_seed,
                    "image_path": str(out_image),
                    "decoded_text": out_text,
                    "do_sample": bool(do_sample_i),
                    "temperature": float(temp_i),
                    "top_p": float(top_p_i),
                    "status": "ok",
                }
            )
            print(f"[INFO] category={category} run={i+1} saved={out_image}")

    with (outdir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
