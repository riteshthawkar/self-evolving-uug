"""
Entry point for the self-evolving training framework on VARGPT.

Mirrors the existing suder/workflow.py pattern, adding:
  - Multi-adapter LoRA setup (proposer + solver + generator)
  - SelfEvolvingConfig loading
  - SelfEvolvingTrainer creation
  - Image-folder mode: pass --se_image_folder /path/to/images to skip dataset JSON
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .adapter_manager import setup_multi_adapter
from .config import SelfEvolvingConfig
from .trainer import SelfEvolvingTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = logging.getLogger(__name__)


def run_self_evolving(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """Run the self-evolving proposer-solver-generator training loop.

    Steps:
      1. Prepare VARGPT v1.1 (special tokens, VAR model loading)
      2. Load tokenizer, template, dataset (or skip if image_folder is set), model
      3. Load SelfEvolvingConfig
      4. Setup multi-adapter LoRA (proposer + solver + generator)
      5. Create SelfEvolvingTrainer
      6. Train
    """
    # ── 1. Prepare VARGPT v1.1 ──────────────────────────────────────────
    if finetuning_args.vargpt_version == "qwen2vl-v1.0":
        from visionllm.vargpt.prepare_vargpt_v1 import prepare_vargpt_qwen2vl
        prepare_vargpt_qwen2vl()
    elif finetuning_args.vargpt_version == "llava-v1.0":
        from visionllm.vargpt_llava.prepare_vargpt_llava import prepare_vargpt_llava
        prepare_vargpt_llava()
    elif finetuning_args.vargpt_version == "qwen2vl-v1.1":
        from visionllm.vargpt_qwen_v1_1.prepare_vargpt_v1_1 import (
            prepare_vargpt_qwen2vl_v1_1,
        )
        prepare_vargpt_qwen2vl_v1_1(base_model_id=model_args.model_name_or_path)
    else:
        logger.warning(
            f"Unknown vargpt_version: {finetuning_args.vargpt_version}. "
            f"Proceeding without VARGPT-specific preparation."
        )

    # ── 2. Load tokenizer, template, dataset, model ─────────────────────
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load SelfEvolvingConfig early to check image_folder
    se_config = SelfEvolvingConfig.from_finetuning_args(finetuning_args)

    # If image_folder is set, dataset JSON is optional.
    # We still try to load it (for fallback), but don't fail if unavailable.
    dataset_module = {}
    if se_config.image_folder:
        logger.info(
            f"[SelfEvolving] Image folder mode: {se_config.image_folder}. "
            f"Dataset JSON is optional."
        )
        try:
            dataset_module = get_dataset(
                template, model_args, data_args, training_args,
                stage="sft", **tokenizer_module,
            )
        except Exception as e:
            logger.info(
                f"[SelfEvolving] No dataset loaded (OK in image-folder mode): {e}"
            )
            dataset_module = {"train_dataset": None, "eval_dataset": None}
    else:
        # Standard mode: dataset is required
        dataset_module = get_dataset(
            template, model_args, data_args, training_args,
            stage="sft", **tokenizer_module,
        )

    model = load_model(
        tokenizer, model_args, finetuning_args, training_args.do_train,
    )

    # ── 3. Log config ─────────────────────────────────────────────────────
    logger.info(
        f"[SelfEvolving] Config: U={se_config.understanding_steps_per_cycle}, "
        f"G={se_config.generation_steps_per_cycle}, "
        f"total={se_config.total_steps}, "
        f"imageless={se_config.imageless_proposer_mode}, "
        f"image_folder={se_config.image_folder}"
    )

    # ── 4. Setup multi-adapter LoRA ─────────────────────────────────────
    if finetuning_args.finetuning_type == "lora":
        model = setup_multi_adapter(model, finetuning_args, se_config)
        logger.info("[SelfEvolving] Multi-adapter LoRA setup complete")
    else:
        logger.warning(
            "[SelfEvolving] Non-LoRA finetuning type detected. "
            "Multi-adapter management will be limited."
        )

    # ── 5. Create SelfEvolvingTrainer ───────────────────────────────────
    trainer = SelfEvolvingTrainer(
        model=model,
        args=training_args,
        se_config=se_config,
        finetuning_args=finetuning_args,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # ── 6. Train ────────────────────────────────────────────────────────
    if training_args.do_train:
        trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint,
        )
        trainer.save_model()
        logger.info("[SelfEvolving] Training complete. Model saved.")

    # Create model card
    create_modelcard_and_push(
        trainer, model_args, data_args, training_args, finetuning_args,
    )
