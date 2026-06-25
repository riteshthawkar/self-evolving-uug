#!/usr/bin/env python3
"""Static protocol checks for the paper/rebuttal reproduction setup."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "scripts/self_evolving/paper/paper_experiments.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flatten_table_experiment_ids(manifest: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for table in manifest.get("blip3o_table_reproduction", []):
        for exp in table.get("experiments", []):
            exp_id = exp.get("id")
            if exp_id:
                ids.append(str(exp_id))
    return ids


def _check_equal(errors: list[str], label: str, value: Any, expected: Any) -> None:
    if value != expected:
        errors.append(f"{label}: expected {expected!r}, got {value!r}")


def validate() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    manifest = json.loads(_read(MANIFEST))
    protocol = manifest.get("protocol", {})

    _check_equal(errors, "protocol.seed", protocol.get("seed"), 42)
    _check_equal(errors, "protocol.total_steps", protocol.get("total_steps"), 10000)
    _check_equal(errors, "protocol.precision", protocol.get("precision"), "bfloat16")
    _check_equal(errors, "protocol.learning_rate", protocol.get("learning_rate"), 1e-6)
    _check_equal(errors, "protocol.weight_decay", protocol.get("weight_decay"), 0.01)
    _check_equal(errors, "protocol.grad_clip", protocol.get("grad_clip"), 1.0)
    _check_equal(errors, "protocol.grad_accum_steps", protocol.get("grad_accum_steps"), 1)
    _check_equal(errors, "protocol.default_data_dir", protocol.get("default_data_dir"), "data/joint_pool_10k/images")
    _check_equal(errors, "protocol.minimum_data_images", protocol.get("minimum_data_images"), 10000)
    _check_equal(
        errors,
        "protocol.lora.targets",
        protocol.get("lora", {}).get("targets"),
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    _check_equal(
        errors,
        "protocol.lora.solver_merger_targets",
        protocol.get("lora", {}).get("solver_merger_targets"),
        "visual.merger.mlp.0,visual.merger.mlp.2",
    )
    _check_equal(
        errors,
        "protocol.lora.blip3o_dit_targets",
        protocol.get("lora", {}).get("blip3o_dit_targets"),
        "attn2.to_q,attn2.to_k,attn2.to_v,attn2.to_out.0,caption_projection.linear_1,caption_projection.linear_2",
    )
    _check_equal(errors, "protocol.sampling.solver_prompt_framings", protocol["sampling"].get("solver_prompt_framings"), 7)
    _check_equal(errors, "protocol.sampling.proposer_candidates", protocol["sampling"].get("proposer_candidates"), 3)
    _check_equal(errors, "protocol.sampling.generation_candidates", protocol["sampling"].get("generation_candidates"), 3)
    _check_equal(errors, "protocol.sampling.ste_window", protocol["sampling"].get("ste_window"), 128)

    table_ids = _flatten_table_experiment_ids(manifest)
    if len(table_ids) != len(set(table_ids)):
        errors.append("blip3o_table_reproduction has duplicate experiment ids")
    if not table_ids:
        errors.append("blip3o_table_reproduction is empty")

    runner = _read(ROOT / "scripts/self_evolving/paper/run_experiment.sh")
    e1 = _read(ROOT / "scripts/self_evolving/final/E1_main_joint.sh")
    e7 = _read(ROOT / "scripts/self_evolving/final/E7_two_stage.sh")
    parser = _read(ROOT / "BLIP3o/blip3o/train/train_self_evolving.py")
    config = _read(ROOT / "BLIP3o/blip3o/train/self_evolving/config.py")
    trainer = _read(ROOT / "BLIP3o/blip3o/train/self_evolving/unified_trainer.py")

    for exp_id in table_ids:
        if not re.search(rf"^\s*{re.escape(exp_id)}\)", runner, re.MULTILINE):
            errors.append(f"run_experiment.sh has no case for {exp_id}")

    e7_exp = next((x for x in manifest.get("training_experiments", []) if x.get("id") == "blip3o_two_stage"), {})
    if e7_exp.get("data_dir") != "data/joint_pool_10k/images":
        errors.append("blip3o_two_stage manifest must use data/joint_pool_10k/images")
    if e7_exp.get("env", {}).get("TWO_STAGE_IMAGE_SAMPLES") != "10000":
        errors.append("blip3o_two_stage manifest must set TWO_STAGE_IMAGE_SAMPLES=10000")
    if e7_exp.get("env", {}).get("STAGE1_STEPS") != "10000":
        errors.append("blip3o_two_stage manifest must set STAGE1_STEPS=10000")
    if e7_exp.get("env", {}).get("STAGE2_STEPS") != "10000":
        errors.append("blip3o_two_stage manifest must set STAGE2_STEPS=10000")
    if e7_exp.get("env", {}).get("TOTAL_STEPS") != "20000":
        errors.append("blip3o_two_stage manifest must set TOTAL_STEPS=20000")
    if "TWO_STAGE_IMAGE_SAMPLES=\"${TWO_STAGE_IMAGE_SAMPLES:-10000}\"" not in e7:
        errors.append("E7_two_stage.sh must default TWO_STAGE_IMAGE_SAMPLES to 10000")
    if "STAGE1_STEPS=\"${STAGE1_STEPS:-10000}\"" not in e7:
        errors.append("E7_two_stage.sh must default STAGE1_STEPS to 10000")
    if "STAGE2_STEPS=\"${STAGE2_STEPS:-10000}\"" not in e7:
        errors.append("E7_two_stage.sh must default STAGE2_STEPS to 10000")
    if "TOTAL_STEPS=\"${TOTAL_STEPS:-$((STAGE1_STEPS + STAGE2_STEPS))}\"" not in e7:
        errors.append("E7_two_stage.sh must derive TOTAL_STEPS from Stage 1 + Stage 2")
    if "MAX_IMAGES=\"$TWO_STAGE_IMAGE_SAMPLES\"" not in e7:
        errors.append("E7_two_stage.sh must pass the same MAX_IMAGES to both stages")
    for required in (
        "TWO_STAGE_DATA_DIR=",
        "TWO_STAGE_UNDERSTANDING_STEPS=\"${TWO_STAGE_UNDERSTANDING_STEPS:-10000}\"",
        "TWO_STAGE_GENERATION_STEPS=\"${TWO_STAGE_GENERATION_STEPS:-10000}\"",
        "STAGE1_STEPS=\"$TWO_STAGE_UNDERSTANDING_STEPS\"",
        "STAGE2_STEPS=\"$TWO_STAGE_GENERATION_STEPS\"",
    ):
        if required not in runner:
            errors.append(f"run_experiment.sh missing two-stage control: {required}")

    e1_defaults = {
        "DATA_DIR": "$REPO_ROOT/data/joint_pool_10k/images",
        "TOTAL_STEPS": "10000",
        "LR": "1e-6",
        "WEIGHT_DECAY": "0.01",
        "GRAD_CLIP": "1.0",
        "GRAD_ACCUM_STEPS": "1",
        "USE_LORA": "1",
        "LORA_R": "16",
        "LORA_ALPHA": "32",
        "LORA_DROPOUT": "0.05",
        "LOAD_IN_4BIT": "0",
        "BNB_4BIT_QUANT_TYPE": "nf4",
        "BNB_4BIT_USE_DOUBLE_QUANT": "1",
        "BNB_4BIT_COMPUTE_DTYPE": "bfloat16",
        "NUM_SOLVER_SAMPLES": "7",
        "NUM_GENERATIONS": "3",
        "PROPOSER_NUM_CANDIDATES": "3",
        "GENERATION_NUM_INFERENCE_STEPS": "50",
        "GENERATION_GUIDANCE_SCALE": "2.0",
        "DIT_LORA": "1",
        "SOLVER_MERGER_LORA": "1",
        "PROPOSER_GEN_REWARD_ENABLED": "0",
        "GEN_STEP_SOLVER_UPDATE_ENABLED": "0",
        "REWARD_SPEC_WEIGHT": "0.65",
        "REWARD_CYCLE_WEIGHT": "0.20",
        "REWARD_DIVERSITY_WEIGHT": "0.10",
        "REWARD_CONTRADICTION_WEIGHT": "0.20",
        "MIN_SPEC_QUALITY_FOR_UPDATE": "0.35",
        "MIN_SPEC_QA_PAIRS": "2",
        "KL_COEF": "0.01",
        "KL_TARGET": "0.02",
        "KL_ADAPT_RATE": "0.10",
        "KL_MIN": "0.001",
        "KL_MAX": "1e2",
        "SOLVER_TOKEN_ENTROPY_WINDOW_SIZE": "128",
        "SOLVER_TOKEN_ENTROPY_AGGREGATION": "max",
        "SOLVER_PPS_ENABLED": "1",
    }
    for key, value in e1_defaults.items():
        pattern = f'{key}="${{{key}:-{value}}}"'
        if pattern not in e1:
            errors.append(f"E1 default mismatch or missing: {pattern}")

    e1_flags = [
        "--deterministic",
        "--dtype bfloat16",
        "--lora_r \"$LORA_R\"",
        "--lora_alpha \"$LORA_ALPHA\"",
        "--lora_dropout \"$LORA_DROPOUT\"",
        "--load_in_4bit",
        "--bnb_4bit_quant_type \"$BNB_4BIT_QUANT_TYPE\"",
        "--bnb_4bit_compute_dtype \"$BNB_4BIT_COMPUTE_DTYPE\"",
        "--num_solver_samples \"$NUM_SOLVER_SAMPLES\"",
        "--num_generations \"$NUM_GENERATIONS\"",
        "--proposer_num_candidates \"$PROPOSER_NUM_CANDIDATES\"",
        "--generation_num_inference_steps \"$GENERATION_NUM_INFERENCE_STEPS\"",
        "--generation_guidance_scale \"$GENERATION_GUIDANCE_SCALE\"",
        "--reward_spec_weight \"$REWARD_SPEC_WEIGHT\"",
        "--reward_cycle_weight \"$REWARD_CYCLE_WEIGHT\"",
        "--reward_diversity_weight \"$REWARD_DIVERSITY_WEIGHT\"",
        "--reward_contradiction_weight \"$REWARD_CONTRADICTION_WEIGHT\"",
        "--kl_coef \"$KL_COEF\"",
        "--kl_target \"$KL_TARGET\"",
        "--kl_adapt_rate \"$KL_ADAPT_RATE\"",
        "--kl_min \"$KL_MIN\"",
        "--kl_max \"$KL_MAX\"",
        "--solver_token_entropy_aggregation \"$SOLVER_TOKEN_ENTROPY_AGGREGATION\"",
    ]
    for flag in e1_flags:
        if flag not in e1:
            errors.append(f"E1 command missing paper flag: {flag}")

    cli_flags = [
        "--load_in_4bit",
        "--bnb_4bit_quant_type",
        "--bnb_4bit_compute_dtype",
        "--solver_token_entropy_enabled",
        "--disable_solver_token_entropy",
        "--solver_token_entropy_tokens",
        "--solver_token_entropy_window_size",
        "--solver_token_entropy_sigmoid_alpha",
        "--solver_token_entropy_sigmoid_beta",
        "--solver_token_entropy_aggregation",
        "--proposer_ste_primary_weight",
        "--proposer_sample_entropy_weight",
        "--proposer_ste_reward_weight",
        "--solver_pps_enabled",
        "--disable_solver_pps",
    ]
    for flag in cli_flags:
        if flag not in parser:
            errors.append(f"train_self_evolving.py missing CLI flag: {flag}")

    parser_default_snippets = [
        'p.add_argument("--grad_accum_steps", type=int, default=1)',
        'p.add_argument("--proposer_update_freq", type=int, default=1)',
        'p.add_argument("--num_generations", type=int, default=3)',
        'p.add_argument("--kl_min", type=float, default=0.001)',
        'p.add_argument("--save_every", type=int, default=50)',
        'p.add_argument("--generation_num_inference_steps", type=int, default=50)',
        'p.add_argument("--generation_height", type=int, default=896)',
        'p.add_argument("--generation_width", type=int, default=896)',
        'p.add_argument("--synthetic_solver_update_freq", type=int, default=0)',
    ]
    for snippet in parser_default_snippets:
        if snippet not in parser:
            errors.append(f"train_self_evolving.py default mismatch: {snippet}")

    config_fields = [
        "solver_token_entropy_aggregation: str = \"max\"",
        "proposer_ste_primary_weight: float = 0.70",
        "proposer_sample_entropy_weight: float = 0.30",
        "solver_pps_enabled: bool = True",
        "load_in_4bit: bool = False",
        "bnb_4bit_quant_type: str = \"nf4\"",
        "solver_merger_lora_enabled: bool = True",
    ]
    for field in config_fields:
        if config.count(field) < 2:
            errors.append(f"config.py should define field in both configs: {field}")

    if "if _ste_aggregation == \"mean\":" not in trainer:
        errors.append("unified_trainer.py does not implement STE mean aggregation")
    if "proposer_ste_raw_value" not in trainer:
        warnings.append("unified_trainer.py does not log proposer_ste_raw_value")

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=["text", "json"], default="text")
    args = ap.parse_args()

    errors, warnings = validate()
    if args.format == "json":
        print(json.dumps({"ok": not errors, "errors": errors, "warnings": warnings}, indent=2))
    else:
        if errors:
            print("Protocol validation: FAIL")
            for item in errors:
                print(f"- ERROR: {item}")
        else:
            print("Protocol validation: PASS")
        for item in warnings:
            print(f"- WARN: {item}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
