"""
Unified (alternating understanding + generation) self-evolving trainer.

Ported from self_evolving/experiments/generation.py (UnifiedSelfEvolvingTrainer).
Extends GenerationSelfEvolvingTrainer with an interleaved understanding phase.
"""

import gc
import time
import traceback
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from PIL import Image

from .config import UnifiedSelfEvolvingConfig
from .generation_helpers import GenerationSpec
from .generation_trainer import GenerationSelfEvolvingTrainer
from .prompts import build_proposer_prompt, build_solver_prompt
from .utils import (
    HAS_WANDB,
    _json_dump,
    _parse_answer,
    _parse_first_question,
    gaussian_reward,
    majority_vote,
    normalize_answer,
    pre_answer_word_count,
    shannon_entropy_nats,
)


class UnifiedSelfEvolvingTrainer(GenerationSelfEvolvingTrainer):
    """
    Unified self-evolving trainer: alternates understanding and generation steps
    within each cycle.

    Extends GenerationSelfEvolvingTrainer with an interleaved understanding phase.
    """

    def __init__(self, config: UnifiedSelfEvolvingConfig):
        if config.enable_solver_updates and config.solver_update_freq <= 0:
            config.solver_update_freq = max(1, config.synthetic_solver_update_freq)
        super().__init__(config)
        self.ucfg = config

    def _understanding_step(self, step: int, image: Image.Image, meta: Dict) -> Dict[str, object]:
        proposer_prompt = build_proposer_prompt()
        proposer_out = self._generate(
            image=image,
            prompt=proposer_prompt,
            adapter_name="proposer" if self.cfg.use_lora else None,
            max_new_tokens=self.cfg.max_new_tokens_proposer,
            temperature=self.cfg.temp,
            top_p=self.cfg.top_p,
        )
        parsed_question = _parse_first_question(proposer_out).replace("\n", " ").strip()
        question = parsed_question or "What is the most salient object in the image?"

        solver_prompt = build_solver_prompt(question)
        solver_outputs: List[str] = []
        solver_answers_raw: List[str] = []
        solver_answers_norm: List[str] = []
        pre_words: List[int] = []

        for _ in range(self.cfg.num_solver_samples):
            solver_out = self._generate(
                image=image,
                prompt=solver_prompt,
                adapter_name="default" if self.cfg.use_lora else None,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=self.cfg.temp,
                top_p=self.cfg.top_p,
            )
            answer_raw = _parse_answer(solver_out)
            solver_outputs.append(solver_out)
            solver_answers_raw.append(answer_raw)
            solver_answers_norm.append(normalize_answer(answer_raw))
            pre_words.append(pre_answer_word_count(solver_out))

        maj_answer, maj_count = majority_vote(solver_answers_norm)
        maj_frac = maj_count / float(self.cfg.num_solver_samples)
        hist: Dict[str, int] = {}
        for ans in solver_answers_norm:
            hist[ans] = hist.get(ans, 0) + 1
        probs = [count / float(self.cfg.num_solver_samples) for count in hist.values()]
        entropy_nats = shannon_entropy_nats(probs)

        solver_rewards_raw = [1.0 if ans == maj_answer else 0.0 for ans in solver_answers_norm]
        target_w = max(1, self.cfg.len_penalty_target_words)
        penalties = [min(1.0, max(0.0, (w - target_w) / float(target_w))) for w in pre_words]
        prob_map = {ans: count / float(self.cfg.num_solver_samples) for ans, count in hist.items()}
        solver_probs = [prob_map[ans] for ans in solver_answers_norm]
        solver_rewards_soft = [
            (prob ** self.cfg.solver_soft_gamma) * (1.0 - self.cfg.len_penalty_weight * pen)
            for prob, pen in zip(solver_probs, penalties)
        ]
        proposer_reward = gaussian_reward(entropy_nats, self.cfg.prop_entropy_mu, self.cfg.prop_entropy_sigma)

        solver_stats_list = []
        solver_update_applied = (
            self.solver_updater is not None
            and self.cfg.solver_update_freq > 0
            and (step % self.cfg.solver_update_freq == 0)
        )
        if solver_update_applied:
            for sample_idx, (completion, reward) in enumerate(zip(solver_outputs, solver_rewards_soft)):
                local_can_solver_update = bool(str(completion).strip())
                can_solver_update = self._dist_all_bool(local_can_solver_update)
                if not can_solver_update:
                    self._append_jsonl(
                        self.policy_updates_log_path,
                        {
                            "step": step,
                            "role": "solver",
                            "source": "understanding",
                            "sample_idx": int(sample_idx),
                            "skipped": True,
                            "reason": "distributed_peer_empty_solver_completion",
                        },
                    )
                    continue
                baseline_before = self.solver_baseline
                stats = self.solver_updater.step(
                    image=image,
                    prompt=solver_prompt,
                    completion=completion,
                    reward=reward,
                    baseline=baseline_before,
                    device=self.device,
                )
                solver_stats_list.append(stats)
                if stats.get("did_step", True):
                    self._policy_update_counts["solver"] += 1
                self._update_baseline("solver", reward)
                self._sync_state_scalars()

                # Aggressive cleanup after each solver update step to avoid OOM
                # on memory-constrained systems (especially with multiple samples).
                del stats
                torch.cuda.empty_cache()
                gc.collect()

        proposer_stats = None
        if step % self.cfg.proposer_update_freq == 0:
            baseline_before = self.proposer_baseline
            proposer_completion = str(proposer_out or "").strip()
            local_can_proposer_update = bool(proposer_completion)
            can_proposer_update = self._dist_all_bool(local_can_proposer_update)
            if can_proposer_update:
                proposer_stats = self.proposer_updater.step(
                    image=image,
                    prompt=proposer_prompt,
                    completion=proposer_completion,
                    reward=proposer_reward,
                    baseline=baseline_before,
                    device=self.device,
                )
                if proposer_stats.get("did_step", True):
                    self._policy_update_counts["proposer"] += 1
                self._update_baseline("proposer", proposer_reward)
            else:
                self._append_jsonl(
                    self.policy_updates_log_path,
                    {
                        "step": step,
                        "role": "proposer",
                        "source": "understanding",
                        "skipped": True,
                        "reason": "distributed_peer_empty_proposer_completion",
                    },
                )
            self._sync_state_scalars()

        record = {
            "step": step,
            "phase": "understanding",
            "image_path": meta.get("path"),
            "question": question,
            "proposer_out": proposer_out,
            "solver_answers_raw": solver_answers_raw,
            "solver_answers_norm": solver_answers_norm,
            "solver_rewards_raw": solver_rewards_raw,
            "solver_rewards_soft": solver_rewards_soft,
            "majority_answer": maj_answer,
            "majority_count": maj_count,
            "majority_fraction": maj_frac,
            "entropy_nats": entropy_nats,
            "proposer_reward": proposer_reward,
            "solver_baseline": self.solver_baseline,
            "proposer_baseline": self.proposer_baseline,
            "solver_update_applied": solver_update_applied,
            "solver_stats": solver_stats_list,
            "proposer_stats": proposer_stats,
        }
        self._append_jsonl(self.iter_log_path, record)

        self._append_jsonl(
            self.rewards_log_path,
            {
                "step": step,
                "phase": "understanding",
                "image_path": meta.get("path"),
                "majority_answer": maj_answer,
                "majority_fraction": maj_frac,
                "entropy_nats": entropy_nats,
                "solver_reward_soft_mean": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
                "proposer_reward": proposer_reward,
            },
        )

        if self.is_main_process and step % self.cfg.log_every == 0:
            print(
                f"[Step {step:05d}][U] maj={maj_count}/{self.cfg.num_solver_samples} "
                f"maj_frac={maj_frac:.2f} H={entropy_nats:.3f} P_R={proposer_reward:.3f}"
            )
            print(f"  Q: {question}")

        self._update_metric("u_majority_fraction", self._dist_mean(maj_frac))
        self._update_metric("u_entropy_nats", self._dist_mean(entropy_nats))
        self._update_metric("u_proposer_reward", self._dist_mean(proposer_reward))

        return record

    def train(self):
        cfg = self.ucfg
        if cfg.total_steps <= self.start_step:
            raise ValueError(
                f"total_steps ({cfg.total_steps}) must be greater than start_step ({self.start_step})."
            )
        cycle = max(1, cfg.understanding_steps_per_cycle + cfg.generation_steps_per_cycle)

        if self.is_main_process:
            print(f"[Unified] Starting run at: {self.run_dir}")
            print(f"[Unified] Model: {cfg.model_name}")
            print(f"[Unified] Generator update rule: {cfg.generator_update_rule}")
            print(f"[Unified] Images: {len(self.pool)}")
            print(f"[Unified] Step range: {self.start_step + 1}..{cfg.total_steps}")
            print(
                f"[Unified] Schedule: Ux{cfg.understanding_steps_per_cycle} + Gx{cfg.generation_steps_per_cycle} (cycle={cycle})"
            )

        last_completed_step = self.start_step
        last_attempted_step = self.start_step
        try:
            for step in range(self.start_step + 1, cfg.total_steps + 1):
                last_attempted_step = step
                image, meta = self._sample_image_for_step(step)

                phase_idx = (step - 1) % cycle
                if phase_idx < cfg.understanding_steps_per_cycle:
                    self._understanding_step(step=step, image=image, meta=meta)
                else:
                    out = self._generation_step(step=step, image=image, meta=meta)
                    source_caption = str(out.get("source_caption", ""))
                    spec: GenerationSpec = out["spec"]
                    scored: List[Dict[str, object]] = out["scored"]
                    spec_quality = float(out.get("spec_quality", 0.0))
                    best_idx = int(out["best_idx"])
                    if cfg.synthetic_solver_update_freq > 0 and step % cfg.synthetic_solver_update_freq == 0:
                        self._solver_synthetic_update_from_best(step, scored[best_idx])

                    rewards = [float(c["total_reward"]) for c in scored]
                    reward_mean = sum(rewards) / max(1, len(rewards))
                    reward_max = max(rewards) if rewards else 0.0
                    reward_min = min(rewards) if rewards else 0.0
                    reward_mean_g = self._dist_mean(reward_mean)
                    reward_max_g = self._dist_mean(reward_max)
                    reward_min_g = self._dist_mean(reward_min)
                    spec_quality_g = self._dist_mean(spec_quality)

                    best = scored[best_idx]
                    best_spec = float(best["spec_score"])
                    best_cycle = float(best["cycle_score"])
                    best_div = float(best["diversity_score"])
                    best_contra = float(best["contradiction_score"])
                    best_spec_g = self._dist_mean(best_spec)
                    best_cycle_g = self._dist_mean(best_cycle)
                    best_div_g = self._dist_mean(best_div)
                    best_contra_g = self._dist_mean(best_contra)

                    self._append_jsonl(
                        self.iter_log_path,
                        {
                            "step": step,
                            "phase": "generation",
                            "image_path": meta.get("path"),
                            "prompt": spec.prompt,
                            "best_idx": best_idx,
                            "best_reward": float(scored[best_idx]["total_reward"]),
                            "spec_quality": spec_quality,
                            "generator_update_rule": self.cfg.generator_update_rule,
                            "generator_update_mode": out.get("generator_update_mode"),
                            "generator_skipped_reason": out.get("generator_skipped_reason"),
                            "generator_baseline": self.generator_baseline,
                            "proposer_baseline": self.proposer_baseline,
                            "solver_baseline": self.solver_baseline,
                        },
                    )

                    self._wandb_log_step(
                        step=step,
                        image_path=meta.get("path"),
                        source_caption=source_caption,
                        spec=spec,
                        scored=scored,
                        best_idx=best_idx,
                        spec_quality=spec_quality_g,
                        reward_mean_global=reward_mean_g,
                        reward_max_global=reward_max_g,
                        reward_min_global=reward_min_g,
                        best_spec_global=best_spec_g,
                        best_cycle_global=best_cycle_g,
                        best_diversity_global=best_div_g,
                        best_contradiction_global=best_contra_g,
                        generator_skipped_reason=out.get("generator_skipped_reason"),
                        generator_update_mode=out.get("generator_update_mode"),
                        proposer_stats=out.get("proposer_stats"),
                        generator_stats=out.get("generator_stats"),
                    )

                if cfg.save_every > 0 and step % cfg.save_every == 0:
                    self._dist_barrier()
                    self._save_checkpoint(step)
                    self._dist_barrier()

                if (
                    torch.cuda.is_available()
                    and cfg.clear_cache_every > 0
                    and step % cfg.clear_cache_every == 0
                ):
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    gc.collect()

                last_completed_step = step

            if cfg.save_every <= 0 or (cfg.total_steps % cfg.save_every) != 0:
                self._dist_barrier()
                self._save_checkpoint(cfg.total_steps)
                self._dist_barrier()
            self._write_ablation_summary(cfg.total_steps, status="completed")
            if self.is_main_process:
                print(f"[Unified] Training complete. Final checkpoint at step {cfg.total_steps:05d}.")

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            interrupted_step = int(last_attempted_step)
            tb = traceback.format_exc()
            if self.is_main_process:
                print(f"[Unified] Training interrupted at step {interrupted_step}: {error_text}")
                _json_dump(
                    self.run_dir / "interruption.json",
                    {
                        "status": "interrupted",
                        "interrupted_at_step": interrupted_step,
                        "last_completed_step": int(last_completed_step),
                        "error": error_text,
                        "traceback": tb,
                    },
                )

            emergency_step = max(1, interrupted_step)
            try:
                self._dist_barrier()
                self._save_checkpoint(emergency_step)
                self._dist_barrier()
            except Exception:
                pass

            self._write_ablation_summary(
                max(last_completed_step, emergency_step),
                status="interrupted",
                interrupted_at_step=interrupted_step,
                error=error_text,
            )
            raise
        finally:
            if self.wandb_run is not None and HAS_WANDB:
                try:
                    import wandb

                    wandb.finish()
                except Exception:
                    pass
            if self.distributed and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
