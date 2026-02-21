"""
SelfEvolvingTrainer: Main training loop for the self-evolving framework on VARGPT.

Inherits from HF Trainer but overrides train() with the custom multi-phase
proposer-solver-generator loop:
  - U-step: proposer proposes questions, solver answers, both get GRPO rewards
  - G-step: proposer proposes specs, generator creates images, GRPO on discrete tokens

Ported from BLIP3o's unified_trainer.py with VARGPT-specific adaptations.
"""

import gc
import json
import logging
import math
import os
import pathlib
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from transformers import Trainer

from .adapter_manager import (
    ROLE_GENERATOR,
    ROLE_PROPOSER,
    ROLE_SOLVER,
    use_role,
    collect_role_params,
    get_role_optimizer,
)
from .config import SelfEvolvingConfig
from .generation_helpers import (
    GenerationSpec,
    _ensure_pil_image,
    _parse_generation_spec,
)
from .gen_policy_updater import VARImageGenPolicyUpdater
from .policy_updater import RolePolicyUpdater
from .prompts import (
    _sample_imageless_topic,
    build_generation_spec_prompt,
    build_imageless_spec_prompt,
    build_proposer_multi_prompt,
    build_proposer_prompt,
    build_solver_prompt,
    build_generator_prompt,
)
from .replay_buffer import ReplayBuffer
from .rewards import score_generated_image
from .utils import (
    _build_chat_text,
    _build_text_only_chat,
    _decode_tokens,
    _json_dump,
    _prepare_mm_inputs,
    _prepare_text_only_inputs,
    _set_global_seed,
    _unwrap_model,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    shannon_entropy_nats,
    use_adapter,
    _parse_answer,
    _parse_all_questions,
    _parse_first_question,
)


logger = logging.getLogger(__name__)


class SelfEvolvingTrainer(Trainer):
    """Self-evolving proposer-solver-generator trainer for VARGPT v1.1.

    Overrides HF Trainer's train() with a custom multi-phase loop.
    Reuses HF Trainer for: model save/load, logging, callbacks, DDP setup.
    """

    def __init__(
        self,
        model,
        args,
        se_config: SelfEvolvingConfig,
        finetuning_args=None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.se_config = se_config
        self.finetuning_args = finetuning_args

        # Get processor/tokenizer
        self.processor = kwargs.get("tokenizer", None) or self.tokenizer

        # Device
        self.device = (
            torch.device(f"cuda:{args.local_rank}")
            if args.local_rank >= 0 and torch.cuda.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Role-specific updaters ──────────────────────────────────────
        self.proposer_updater = RolePolicyUpdater(
            model=model,
            processor=self.processor,
            config=se_config,
            adapter_name=ROLE_PROPOSER,
        )
        self.solver_updater = RolePolicyUpdater(
            model=model,
            processor=self.processor,
            config=se_config,
            adapter_name=ROLE_SOLVER,
        )
        self.generator_updater = VARImageGenPolicyUpdater(
            model=model,
            tokenizer=self.processor,
            config=se_config,
        )

        # ── Replay buffer ───────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(
            max_size=se_config.replay_buffer_size,
            min_reward=se_config.replay_min_reward,
            max_staleness=se_config.replay_max_staleness,
        )

        # ── Baselines & tracking ────────────────────────────────────────
        self.proposer_baseline = 0.0
        self.solver_baseline = 0.0
        self.generator_baseline = 0.0
        self.proposer_gen_baseline = 0.0
        self.reward_ema = 0.0
        self.global_step = 0

        # ── DDP detection ────────────────────────────────────────────────
        self._is_ddp = dist.is_available() and dist.is_initialized()

        # ── Image folder mode ────────────────────────────────────────────
        # When se_image_folder is set, scan the folder for images at init
        # so _sample_image() can pick random images without a JSON dataset.
        self._image_folder_paths: List[str] = []
        if se_config.image_folder:
            self._image_folder_paths = self._scan_image_folder(
                se_config.image_folder
            )
            logger.info(
                f"[SelfEvolvingTrainer] Image folder mode: "
                f"found {len(self._image_folder_paths)} images in "
                f"{se_config.image_folder}"
            )

        logger.info(
            f"[SelfEvolvingTrainer] Initialized with "
            f"U={se_config.understanding_steps_per_cycle}, "
            f"G={se_config.generation_steps_per_cycle}, "
            f"total_steps={se_config.total_steps}"
        )

    # ── Main training loop ──────────────────────────────────────────────

    def train(self, resume_from_checkpoint=None, **kwargs):
        """Override HF Trainer's train() with the self-evolving multi-phase loop."""
        cfg = self.se_config

        # Set seed
        _set_global_seed(cfg.seed)

        # Resume
        start_step = cfg.start_step
        if resume_from_checkpoint:
            start_step = self._load_se_checkpoint(resume_from_checkpoint)

        # Create output directory
        output_dir = pathlib.Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        _json_dump(output_dir / "se_config.json", {
            k: str(v) if not isinstance(v, (int, float, bool, str, type(None)))
            else v
            for k, v in cfg.__dict__.items()
        })

        logger.info(f"[SelfEvolvingTrainer] Starting training from step {start_step}")

        cycle_len = cfg.understanding_steps_per_cycle + cfg.generation_steps_per_cycle
        if cycle_len <= 0:
            raise ValueError("Cycle length must be > 0")

        self.model.train()

        for step in range(start_step, cfg.total_steps):
            self.global_step = step
            step_start = time.time()

            # Determine phase
            phase_in_cycle = step % cycle_len
            is_u_step = phase_in_cycle < cfg.understanding_steps_per_cycle

            try:
                if is_u_step:
                    step_stats = self._understanding_step(step)
                    step_stats["phase"] = "understanding"
                else:
                    step_stats = self._generation_step(step)
                    step_stats["phase"] = "generation"
            except Exception as e:
                logger.error(f"[SelfEvolvingTrainer] Step {step} failed: {e}")
                import traceback
                traceback.print_exc()
                step_stats = {"phase": "error", "error": str(e)}

            step_stats["step"] = step
            step_stats["step_time"] = time.time() - step_start

            # ── Logging ─────────────────────────────────────────────────
            if step % cfg.log_every == 0:
                self._log_step(step, step_stats)

            # ── Checkpointing ───────────────────────────────────────────
            if step > 0 and step % cfg.save_every == 0:
                self._save_se_checkpoint(step, output_dir)

            # ── Memory management ───────────────────────────────────────
            if cfg.clear_cache_every > 0 and step % cfg.clear_cache_every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final save
        self._save_se_checkpoint(cfg.total_steps, output_dir)
        logger.info("[SelfEvolvingTrainer] Training complete.")

    # ── Understanding Step ──────────────────────────────────────────────

    def _understanding_step(self, step: int) -> Dict:
        """U-step: proposer proposes questions, solver answers, both update."""
        cfg = self.se_config
        stats = {}

        # ── 1. Sample image ─────────────────────────────────────────────
        image, source = self._sample_image(step)
        if image is None:
            return {"u_skipped": True, "reason": "no_image"}
        stats["image_source"] = source

        # ── 2. Proposer generates question(s) ───────────────────────────
        with use_role(self.model, ROLE_PROPOSER):
            question_text, proposer_completion = self._generate_proposer_question(
                image, step
            )
        if not question_text:
            return {"u_skipped": True, "reason": "no_question"}
        stats["question"] = question_text[:100]

        # ── 3. Solver answers with K samples ────────────────────────────
        with use_role(self.model, ROLE_SOLVER):
            answers, solver_completions = self._generate_solver_answers(
                image, question_text, num_samples=cfg.num_solver_samples,
            )
        stats["num_answers"] = len(answers)

        # ── 4. Compute rewards ──────────────────────────────────────────
        # Normalize answers for voting
        norm_answers = [normalize_answer(a) for a in answers]
        if not norm_answers:
            return {"u_skipped": True, "reason": "no_answers"}

        # Majority vote and entropy
        majority_answer, majority_count = majority_vote(norm_answers)
        majority_frac = majority_count / len(norm_answers)

        # Entropy
        counts = {}
        for a in norm_answers:
            counts[a] = counts.get(a, 0) + 1
        probs = [c / len(norm_answers) for c in counts.values()]
        entropy = shannon_entropy_nats(probs)

        # Proposer reward: based on question difficulty (entropy)
        proposer_reward = gaussian_reward(
            entropy, cfg.prop_entropy_mu, cfg.prop_entropy_sigma,
        )
        # Penalty for zero-entropy (trivially easy question)
        if entropy < 0.01:
            proposer_reward = min(proposer_reward, cfg.zero_entropy_reward_cap)

        # Solver reward: based on accuracy (high majority = good)
        solver_reward = cfg.solver_soft_gamma * majority_frac + (
            1.0 - cfg.solver_soft_gamma
        ) * (1.0 - min(1.0, entropy / max(cfg.sc_entropy_max, 0.01)))

        stats.update({
            "entropy": entropy,
            "majority_frac": majority_frac,
            "majority_answer": majority_answer[:50],
            "proposer_reward": proposer_reward,
            "solver_reward": solver_reward,
        })

        # ── 5. Update proposer ──────────────────────────────────────────
        proposer_prompt = build_proposer_prompt(
            target_difficulty=self._current_difficulty(step)
        )
        prop_stats = self.proposer_updater.step(
            image=image,
            prompt=proposer_prompt,
            completion=proposer_completion,
            reward=proposer_reward,
            baseline=self.proposer_baseline,
            device=self.device,
            ddp_no_sync=self._is_ddp,
        )
        self.proposer_baseline = (
            cfg.baseline_momentum * self.proposer_baseline
            + (1 - cfg.baseline_momentum) * proposer_reward
        )
        stats.update({f"prop_{k}": v for k, v in prop_stats.items()})

        # ── 6. Update solver ────────────────────────────────────────────
        # Use the majority answer as the completion for solver update
        solver_prompt = build_solver_prompt(question_text)
        solver_completion = f"\n<answer>{majority_answer}</answer>"
        sol_stats = self.solver_updater.step(
            image=image,
            prompt=solver_prompt,
            completion=solver_completion,
            reward=solver_reward,
            baseline=self.solver_baseline,
            device=self.device,
            ddp_no_sync=self._is_ddp,
        )
        self.solver_baseline = (
            cfg.baseline_momentum * self.solver_baseline
            + (1 - cfg.baseline_momentum) * solver_reward
        )
        stats.update({f"sol_{k}": v for k, v in sol_stats.items()})

        return stats

    # ── Generation Step ─────────────────────────────────────────────────

    def _generation_step(self, step: int) -> Dict:
        """G-step: proposer proposes specs, generator creates images, GRPO update."""
        cfg = self.se_config
        stats = {}

        # ── 1. Proposer generates spec ──────────────────────────────────
        if cfg.imageless_proposer_mode:
            topic = _sample_imageless_topic(step, cfg.seed)
            spec, spec_completion = self._generate_imageless_spec(topic, step)
            stats["topic"] = topic[:80]
        else:
            # Sample a source image for spec generation
            image, source = self._sample_image(step)
            if image is None:
                return {"g_skipped": True, "reason": "no_source_image"}
            spec, spec_completion = self._generate_spec(image, step)
            stats["image_source"] = source

        if spec is None or not spec.prompt:
            return {"g_skipped": True, "reason": "no_spec"}

        stats["gen_prompt"] = spec.prompt[:100]
        stats["num_qa_pairs"] = len(spec.qa_pairs)

        # ── 2. Generate K candidate images ──────────────────────────────
        K = cfg.num_generations
        candidates = []  # List of (PIL Image, pixel_gen_tensor)
        with use_role(self.model, ROLE_GENERATOR):
            for k in range(K):
                try:
                    gen_image, gen_tensor = self._generate_image(spec.prompt)
                    if gen_image is not None:
                        candidates.append((gen_image, gen_tensor))
                except Exception as e:
                    logger.warning(f"[SelfEvolvingTrainer] Generation {k} failed: {e}")

        if not candidates:
            return {"g_skipped": True, "reason": "no_candidates"}
        stats["num_candidates"] = len(candidates)

        # ── 3. Score candidates ─────────────────────────────────────────
        questions = [qa.question for qa in spec.qa_pairs]
        expected_answers = [qa.expected for qa in spec.qa_pairs]

        rewards = []
        reward_details_list = []
        for gen_image, _ in candidates:
            reward, details = score_generated_image(
                model=self.model,
                processor=self.processor,
                image=gen_image,
                prompt=spec.prompt,
                questions=questions,
                expected_answers=expected_answers,
                device=self.device,
                config=cfg,
            )
            rewards.append(reward)
            reward_details_list.append(details)

        stats["gen_rewards"] = rewards
        stats["gen_reward_mean"] = sum(rewards) / len(rewards)
        stats["gen_reward_max"] = max(rewards)

        # ── 4. GRPO update on generator ─────────────────────────────────
        if len(candidates) >= 2:
            # Prepare inputs for GRPO
            gen_prompt = build_generator_prompt(spec.prompt)
            chat_text = _build_text_only_chat(self.processor, gen_prompt)
            text_inputs = _prepare_text_only_inputs(
                self.processor, self.device, chat_text,
            )
            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]
            labels = input_ids.clone()

            pixel_gen_values_list = [[c[1]] for c in candidates]

            gen_stats = self.generator_updater.step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_gen_values_list=pixel_gen_values_list,
                rewards=rewards,
                device=self.device,
                ddp_no_sync=self._is_ddp,
            )
            stats.update({f"gen_{k}": v for k, v in gen_stats.items()})

        # ── 5. Update proposer with generation quality ──────────────────
        if cfg.proposer_gen_reward_enabled:
            gen_quality = max(rewards)
            prop_gen_reward = (
                cfg.proposer_gen_entropy_weight * 0.5  # placeholder: no entropy here
                + (1 - cfg.proposer_gen_entropy_weight) * gen_quality
            )

            if cfg.imageless_proposer_mode:
                prop_prompt = build_imageless_spec_prompt(
                    topic if cfg.imageless_proposer_mode else "",
                    target_difficulty=self._current_difficulty(step),
                )
                prop_stats = self.proposer_updater.step(
                    image=None,
                    prompt=prop_prompt,
                    completion=spec_completion,
                    reward=prop_gen_reward,
                    baseline=self.proposer_gen_baseline,
                    device=self.device,
                    ddp_no_sync=self._is_ddp,
                )
            else:
                prop_prompt = build_generation_spec_prompt(
                    target_difficulty=self._current_difficulty(step),
                )
                prop_stats = self.proposer_updater.step(
                    image=image,
                    prompt=prop_prompt,
                    completion=spec_completion,
                    reward=prop_gen_reward,
                    baseline=self.proposer_gen_baseline,
                    device=self.device,
                    ddp_no_sync=self._is_ddp,
                )

            self.proposer_gen_baseline = (
                cfg.proposer_gen_baseline_momentum * self.proposer_gen_baseline
                + (1 - cfg.proposer_gen_baseline_momentum) * prop_gen_reward
            )
            stats.update({f"prop_gen_{k}": v for k, v in prop_stats.items()})

        # ── 6. Best image → replay buffer ───────────────────────────────
        best_idx = rewards.index(max(rewards))
        best_image = candidates[best_idx][0]

        added = self.replay_buffer.add(
            image=best_image,
            prompt=spec.prompt,
            questions=questions,
            reference_answers=expected_answers,
            reward=max(rewards),
            step=step,
        )
        stats["replay_buffer_added"] = added
        stats["replay_buffer_size"] = len(self.replay_buffer)

        # Update reward EMA
        self.reward_ema = (
            cfg.reward_ema_momentum * self.reward_ema
            + (1 - cfg.reward_ema_momentum) * max(rewards)
        )
        stats["reward_ema"] = self.reward_ema

        return stats

    # ── Helper Methods ──────────────────────────────────────────────────

    @staticmethod
    def _scan_image_folder(folder_path: str) -> List[str]:
        """Recursively scan a folder for image files. Returns sorted list of paths."""
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
        folder = pathlib.Path(folder_path)
        if not folder.is_dir():
            logger.warning(f"[SelfEvolvingTrainer] image_folder not found: {folder_path}")
            return []
        paths = []
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(p))
        paths.sort()
        return paths

    def _sample_image(self, step: int) -> Tuple[Optional[Image.Image], str]:
        """Sample an image: image folder, replay buffer, or dataset."""
        cfg = self.se_config

        # Gen-mix ratio: linearly ramp from start to max
        warmup = max(1, cfg.gen_mix_ratio_warmup_steps)
        ratio = cfg.gen_mix_ratio_start + (
            cfg.gen_mix_ratio_max - cfg.gen_mix_ratio_start
        ) * min(1.0, step / warmup)

        # Try replay buffer first (for generated image mixing)
        if random.random() < ratio and self.replay_buffer:
            entry = self.replay_buffer.sample()
            if entry is not None:
                return entry.image, "replay_buffer"

        # ── Image folder mode (preferred when set) ────────────────────
        if self._image_folder_paths:
            path = random.choice(self._image_folder_paths)
            try:
                pil_img = Image.open(path).convert("RGB")
                return pil_img, "image_folder"
            except Exception as e:
                logger.warning(
                    f"[SelfEvolvingTrainer] Failed to load {path}: {e}"
                )
                return None, "none"

        # ── Fallback: LLaMA-Factory dataset ───────────────────────────
        try:
            if self.train_dataset is not None and len(self.train_dataset) > 0:
                idx = random.randint(0, len(self.train_dataset) - 1)
                sample = self.train_dataset[idx]
                image_obj = None
                for key in ("images", "image", "pixel_values"):
                    if key in sample and sample[key] is not None:
                        image_obj = sample[key]
                        if isinstance(image_obj, (list, tuple)) and image_obj:
                            image_obj = image_obj[0]
                        break
                if image_obj is not None:
                    if isinstance(image_obj, Image.Image):
                        return image_obj, "dataset"
                    if isinstance(image_obj, str):
                        try:
                            pil_img = Image.open(image_obj).convert("RGB")
                            return pil_img, "dataset"
                        except Exception as e_open:
                            logger.warning(
                                f"[SelfEvolvingTrainer] Failed to open "
                                f"'{image_obj}': {e_open}"
                            )
                            return None, "none"
                    if isinstance(image_obj, bytes):
                        from io import BytesIO
                        pil_img = Image.open(BytesIO(image_obj)).convert("RGB")
                        return pil_img, "dataset"
                    return _ensure_pil_image(image_obj), "dataset"
        except Exception as e:
            logger.warning(f"[SelfEvolvingTrainer] Dataset sampling failed: {e}")

        return None, "none"

    def _generate_proposer_question(
        self, image: Image.Image, step: int
    ) -> Tuple[str, str]:
        """Generate a question from the proposer."""
        cfg = self.se_config
        difficulty = self._current_difficulty(step)

        if cfg.proposer_num_candidates > 1:
            prompt = build_proposer_multi_prompt(
                target_difficulty=difficulty,
                num_questions=cfg.proposer_num_candidates,
            )
        else:
            prompt = build_proposer_prompt(target_difficulty=difficulty)

        chat_text = _build_chat_text(self.processor, image, prompt)
        mm_inputs = _prepare_mm_inputs(
            self.processor, self.device, image, chat_text, model=self.model
        )

        base_model = _unwrap_model(self.model)
        with torch.no_grad():
            gen_ids = base_model.generate(
                **mm_inputs,
                max_new_tokens=cfg.max_new_tokens_proposer,
                do_sample=True,
                temperature=cfg.temp,
                top_p=cfg.top_p,
            )

        input_len = mm_inputs["input_ids"].shape[1]
        new_ids = gen_ids[0, input_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        completion = _decode_tokens(tokenizer, new_ids)

        # Parse question(s)
        questions = _parse_all_questions(completion)
        if not questions:
            question = _parse_first_question(completion)
        else:
            question = questions[0]  # Use first/hardest

        return question, completion

    def _generate_solver_answers(
        self,
        image: Image.Image,
        question: str,
        num_samples: int = 5,
    ) -> Tuple[List[str], List[str]]:
        """Generate solver answers for a given question."""
        cfg = self.se_config
        prompt = build_solver_prompt(question)
        chat_text = _build_chat_text(self.processor, image, prompt)
        mm_inputs = _prepare_mm_inputs(
            self.processor, self.device, image, chat_text, model=self.model
        )

        base_model = _unwrap_model(self.model)
        answers = []
        completions = []

        with torch.no_grad():
            for _ in range(num_samples):
                try:
                    gen_ids = base_model.generate(
                        **mm_inputs,
                        max_new_tokens=cfg.max_new_tokens_solver,
                        do_sample=True,
                        temperature=cfg.temp,
                        top_p=cfg.top_p,
                    )
                    input_len = mm_inputs["input_ids"].shape[1]
                    new_ids = gen_ids[0, input_len:]
                    tokenizer = getattr(self.processor, "tokenizer", self.processor)
                    comp = _decode_tokens(tokenizer, new_ids)
                    completions.append(comp)
                    answers.append(_parse_answer(comp))
                except Exception:
                    pass

        return answers, completions

    def _generate_imageless_spec(
        self, topic: str, step: int
    ) -> Tuple[Optional[GenerationSpec], str]:
        """Generate a spec from topic text (no image)."""
        cfg = self.se_config
        difficulty = self._current_difficulty(step)
        prompt = build_imageless_spec_prompt(topic, target_difficulty=difficulty)

        chat_text = _build_text_only_chat(self.processor, prompt)
        text_inputs = _prepare_text_only_inputs(
            self.processor, self.device, chat_text,
        )

        base_model = _unwrap_model(self.model)
        with use_role(self.model, ROLE_PROPOSER):
            with torch.no_grad():
                gen_ids = base_model.generate(
                    **text_inputs,
                    max_new_tokens=cfg.max_new_tokens_proposer,
                    do_sample=True,
                    temperature=cfg.temp,
                    top_p=cfg.top_p,
                )

        input_len = text_inputs["input_ids"].shape[1]
        new_ids = gen_ids[0, input_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        completion = _decode_tokens(tokenizer, new_ids)

        spec = _parse_generation_spec(completion)
        if spec.fallback_used:
            return None, completion

        return spec, completion

    def _generate_spec(
        self, image: Image.Image, step: int
    ) -> Tuple[Optional[GenerationSpec], str]:
        """Generate a spec from a source image."""
        cfg = self.se_config
        difficulty = self._current_difficulty(step)
        prompt = build_generation_spec_prompt(target_difficulty=difficulty)

        chat_text = _build_chat_text(self.processor, image, prompt)
        mm_inputs = _prepare_mm_inputs(
            self.processor, self.device, image, chat_text, model=self.model
        )

        base_model = _unwrap_model(self.model)
        with use_role(self.model, ROLE_PROPOSER):
            with torch.no_grad():
                gen_ids = base_model.generate(
                    **mm_inputs,
                    max_new_tokens=cfg.max_new_tokens_proposer,
                    do_sample=True,
                    temperature=cfg.temp,
                    top_p=cfg.top_p,
                )

        input_len = mm_inputs["input_ids"].shape[1]
        new_ids = gen_ids[0, input_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        completion = _decode_tokens(tokenizer, new_ids)

        spec = _parse_generation_spec(completion)
        if spec.fallback_used:
            return None, completion

        return spec, completion

    def _generate_image(
        self, prompt: str
    ) -> Tuple[Optional[Image.Image], Optional[torch.Tensor]]:
        """Generate a single image from a text prompt using VARGPT.

        Uses the model's autoregressive_infer_cfg() method for image generation.

        Returns
        -------
        image : PIL Image or None
        tensor : torch.Tensor (the raw pixel tensor for GRPO training)
        """
        cfg = self.se_config
        base_model = _unwrap_model(self.model)

        try:
            # Build generation prompt with special tokens
            gen_prompt = build_generator_prompt(prompt)

            # Tokenize
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            inputs = tokenizer(gen_prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                # Use VARGPT's autoregressive_infer_cfg for image generation
                if hasattr(base_model, "autoregressive_infer_cfg"):
                    # This returns generated images as tensors
                    gen_result = base_model.autoregressive_infer_cfg(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        cfg_scale=cfg.var_cfg_scale,
                        tau=cfg.var_tau,
                        top_k=cfg.var_top_k,
                        top_p=cfg.var_top_p,
                    )

                    if gen_result is not None:
                        if isinstance(gen_result, torch.Tensor):
                            img_tensor = gen_result
                        elif isinstance(gen_result, (list, tuple)):
                            img_tensor = gen_result[0] if gen_result else None
                        else:
                            img_tensor = gen_result

                        if img_tensor is not None:
                            pil_image = _ensure_pil_image(img_tensor)
                            return pil_image, img_tensor
                else:
                    # Fallback: use model.generate() with image generation mode
                    gen_ids = base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=cfg.var_tau,
                        top_p=cfg.var_top_p,
                        inference_image_gen=True,
                    )
                    logger.info("[SelfEvolvingTrainer] Used fallback generate() for image gen")

        except Exception as e:
            logger.warning(f"[SelfEvolvingTrainer] Image generation failed: {e}")
            import traceback
            traceback.print_exc()

        return None, None

    def _current_difficulty(self, step: int) -> str:
        """Determine current difficulty level based on curriculum."""
        cfg = self.se_config
        if not cfg.difficulty_sampler_enabled:
            return "medium"

        # Simple linear curriculum: easy → medium → hard
        frac = step / max(cfg.total_steps, 1)
        if frac < 0.3:
            return "easy"
        elif frac < 0.7:
            return "medium"
        else:
            return "hard"

    # ── Checkpoint Management ───────────────────────────────────────────

    def _save_se_checkpoint(self, step: int, output_dir: pathlib.Path):
        """Save self-evolving specific state."""
        ckpt_dir = output_dir / f"se_checkpoint_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "step": step,
            "proposer_baseline": self.proposer_baseline,
            "solver_baseline": self.solver_baseline,
            "generator_baseline": self.generator_baseline,
            "proposer_gen_baseline": self.proposer_gen_baseline,
            "reward_ema": self.reward_ema,
            "proposer_updater": self.proposer_updater.state_dict(),
            "solver_updater": self.solver_updater.state_dict(),
            "generator_updater": self.generator_updater.state_dict(),
            "replay_buffer_stats": self.replay_buffer.stats(),
        }

        torch.save(state, ckpt_dir / "se_state.pt")

        # Save model adapters
        try:
            self.save_model(str(ckpt_dir / "model"))
        except Exception as e:
            logger.warning(f"[SelfEvolvingTrainer] Model save failed: {e}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(output_dir, keep=self.se_config.max_checkpoints)

        logger.info(f"[SelfEvolvingTrainer] Saved checkpoint at step {step}")

    def _load_se_checkpoint(self, checkpoint_path: str) -> int:
        """Load self-evolving checkpoint. Returns the step to resume from."""
        ckpt_path = pathlib.Path(checkpoint_path)

        se_state_path = ckpt_path / "se_state.pt"
        if not se_state_path.exists():
            # Try looking for se_state.pt in parent
            se_state_path = ckpt_path.parent / "se_state.pt"

        if se_state_path.exists():
            state = torch.load(se_state_path, map_location="cpu")
            self.proposer_baseline = state.get("proposer_baseline", 0.0)
            self.solver_baseline = state.get("solver_baseline", 0.0)
            self.generator_baseline = state.get("generator_baseline", 0.0)
            self.proposer_gen_baseline = state.get("proposer_gen_baseline", 0.0)
            self.reward_ema = state.get("reward_ema", 0.0)

            if "proposer_updater" in state:
                self.proposer_updater.load_state_dict(state["proposer_updater"])
            if "solver_updater" in state:
                self.solver_updater.load_state_dict(state["solver_updater"])
            if "generator_updater" in state:
                self.generator_updater.load_state_dict(state["generator_updater"])

            step = state.get("step", 0)
            logger.info(f"[SelfEvolvingTrainer] Resumed from step {step}")
            return step

        logger.warning(
            f"[SelfEvolvingTrainer] No se_state.pt found at {checkpoint_path}"
        )
        return 0

    def _cleanup_old_checkpoints(self, output_dir: pathlib.Path, keep: int = 5):
        """Remove old checkpoints, keeping only the most recent `keep`."""
        import glob
        ckpt_dirs = sorted(
            output_dir.glob("se_checkpoint_*"),
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
        )
        while len(ckpt_dirs) > keep:
            old_dir = ckpt_dirs.pop(0)
            try:
                import shutil
                shutil.rmtree(old_dir)
            except Exception:
                pass

    # ── Logging ─────────────────────────────────────────────────────────

    def _log_step(self, step: int, stats: Dict):
        """Log step statistics."""
        phase = stats.get("phase", "unknown")
        step_time = stats.get("step_time", 0.0)

        msg_parts = [f"step={step}", f"phase={phase}", f"time={step_time:.1f}s"]

        if phase == "understanding":
            msg_parts.append(f"entropy={stats.get('entropy', 0.0):.3f}")
            msg_parts.append(f"prop_r={stats.get('proposer_reward', 0.0):.3f}")
            msg_parts.append(f"sol_r={stats.get('solver_reward', 0.0):.3f}")
        elif phase == "generation":
            msg_parts.append(f"reward_mean={stats.get('gen_reward_mean', 0.0):.3f}")
            msg_parts.append(f"reward_max={stats.get('gen_reward_max', 0.0):.3f}")
            msg_parts.append(f"replay_sz={stats.get('replay_buffer_size', 0)}")

        logger.info(f"[SE] {' | '.join(msg_parts)}")

        # Log to W&B if available
        try:
            import wandb
            if wandb.run is not None:
                log_dict = {
                    f"se/{k}": v for k, v in stats.items()
                    if isinstance(v, (int, float)) and not math.isnan(v)
                    and not math.isinf(v)
                }
                wandb.log(log_dict, step=step)
        except Exception:
            pass
