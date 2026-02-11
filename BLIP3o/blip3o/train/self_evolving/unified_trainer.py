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

    def _phase_local_step_index(self, step: int, phase: str) -> int:
        cycle = max(1, self.cfg.understanding_steps_per_cycle + self.cfg.generation_steps_per_cycle)
        cycle_idx = (step - 1) // cycle
        phase_idx = (step - 1) % cycle
        if phase == "understanding":
            if phase_idx >= self.cfg.understanding_steps_per_cycle:
                return 0
            return cycle_idx * self.cfg.understanding_steps_per_cycle + phase_idx + 1
        if phase == "generation":
            if phase_idx < self.cfg.understanding_steps_per_cycle:
                return 0
            gen_pos = phase_idx - self.cfg.understanding_steps_per_cycle
            return cycle_idx * self.cfg.generation_steps_per_cycle + gen_pos + 1
        raise ValueError(f"Unknown phase: {phase!r}")

    def _is_proposer_update_due(self, step: int, phase: str) -> bool:
        freq = int(getattr(self.cfg, "proposer_update_freq", 0) or 0)
        if freq <= 0:
            return False
        local_idx = self._phase_local_step_index(step, phase)
        if local_idx <= 0:
            return False
        return (local_idx % freq) == 0

    def _solver_top_p_schedule(self) -> List[float]:
        """Vary top_p across solver samples to inject diversity.

        Ranges from solver_top_p_min (default 0.5) to solver_top_p_max (default 1.0).
        Lower top_p forces the model to commit to fewer tokens, producing more
        varied short answers that break unanimous voting on trivially easy questions.
        """
        n = max(1, int(self.cfg.num_solver_samples))
        top_p_min = float(getattr(self.cfg, "solver_top_p_min", 0.5))
        top_p_max = float(getattr(self.cfg, "solver_top_p_max", 1.0))
        if n <= 1:
            return [top_p_max]
        if abs(top_p_max - top_p_min) < 1e-8:
            return [top_p_min] * n
        return [top_p_min + (top_p_max - top_p_min) * (float(i) / float(n - 1)) for i in range(n)]

    def _solver_temperature_schedule(self) -> List[float]:
        n = max(1, int(self.cfg.num_solver_samples))
        base = float(self.cfg.temp)
        if n <= 1 or not bool(getattr(self.cfg, "solver_use_temperature_mix", True)):
            return [base] * n
        tmin = float(getattr(self.cfg, "solver_temp_min", base))
        tmax = float(getattr(self.cfg, "solver_temp_max", base))
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        if abs(tmax - tmin) < 1e-8:
            return [tmin] * n
        return [tmin + (tmax - tmin) * (float(i) / float(n - 1)) for i in range(n)]

    def _update_proposer_entropy_target(self, entropy_nats: float) -> float:
        if not bool(getattr(self.cfg, "adaptive_prop_entropy_target", True)):
            return float(self.cfg.prop_entropy_mu)
        anchor = self._dist_mean(float(entropy_nats))
        prev = float(getattr(self, "proposer_entropy_mu_ema", self.cfg.prop_entropy_mu))
        momentum = float(getattr(self.cfg, "prop_entropy_ema_momentum", 0.95))
        momentum = max(0.0, min(0.9999, momentum))
        ema = momentum * prev + (1.0 - momentum) * anchor
        mu_min = float(getattr(self.cfg, "prop_entropy_mu_min", 0.0))
        mu_max = float(getattr(self.cfg, "prop_entropy_mu_max", 10.0))
        if mu_min > mu_max:
            mu_min, mu_max = mu_max, mu_min
        ema = max(mu_min, min(mu_max, ema))
        self.proposer_entropy_mu_ema = float(ema)
        return float(ema)

    def _understanding_step(self, step: int, image: Image.Image, meta: Dict) -> Dict[str, object]:
        step_t0 = time.perf_counter()
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

        solver_temperatures = self._solver_temperature_schedule()
        solver_top_ps = self._solver_top_p_schedule()
        for sample_idx in range(self.cfg.num_solver_samples):
            solver_temp = (
                float(solver_temperatures[sample_idx])
                if sample_idx < len(solver_temperatures)
                else float(self.cfg.temp)
            )
            solver_top_p = (
                float(solver_top_ps[sample_idx])
                if sample_idx < len(solver_top_ps)
                else float(self.cfg.top_p)
            )
            solver_out = self._generate(
                image=image,
                prompt=solver_prompt,
                adapter_name="default" if self.cfg.use_lora else None,
                max_new_tokens=self.cfg.max_new_tokens_solver,
                temperature=solver_temp,
                top_p=solver_top_p,
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

        sorted_probs = sorted(probs, reverse=True)
        p1 = float(sorted_probs[0]) if sorted_probs else 0.0
        p2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = max(0.0, p1 - p2)
        entropy_min = float(getattr(self.cfg, "sc_entropy_min", 0.15))
        entropy_max = float(getattr(self.cfg, "sc_entropy_max", 1.2))
        if entropy_min > entropy_max:
            entropy_min, entropy_max = entropy_max, entropy_min
        margin_max = float(getattr(self.cfg, "sc_margin_max", 0.9))
        neg_weight = float(getattr(self.cfg, "sc_negative_weight", 0.25))
        solver_informative_local = bool(
            (entropy_min <= entropy_nats <= entropy_max) or (margin <= margin_max)
        )
        solver_informative_all = self._dist_all_bool(solver_informative_local)

        solver_rewards_raw = [
            margin if ans == maj_answer else (-neg_weight * margin)
            for ans in solver_answers_norm
        ]
        target_w = max(1, self.cfg.len_penalty_target_words)
        penalties = [min(1.0, max(0.0, (w - target_w) / float(target_w))) for w in pre_words]
        prob_map = {ans: count / float(self.cfg.num_solver_samples) for ans, count in hist.items()}
        solver_probs = [prob_map[ans] for ans in solver_answers_norm]
        solver_rewards_soft = [
            (prob ** self.cfg.solver_soft_gamma) * (1.0 - self.cfg.len_penalty_weight * pen)
            * reward_raw
            for prob, pen, reward_raw in zip(solver_probs, penalties, solver_rewards_raw)
        ]
        proposer_entropy_mu_used = self._update_proposer_entropy_target(entropy_nats)
        proposer_reward = gaussian_reward(
            entropy_nats,
            proposer_entropy_mu_used,
            self.cfg.prop_entropy_sigma,
        )
        # Penalize zero-entropy (unanimous) outcomes — the question was too easy.
        # When all solvers agree perfectly, the proposer gets at most 10% of the
        # Gaussian reward.  Any disagreement (entropy > 0) removes the penalty.
        zero_entropy_cap = float(getattr(self.cfg, "zero_entropy_reward_cap", 0.10))
        if entropy_nats < 1e-6 and zero_entropy_cap < 1.0:
            proposer_reward = min(proposer_reward, zero_entropy_cap)

        solver_stats_list = []
        solver_update_due = (
            self.solver_updater is not None
            and self.cfg.solver_update_freq > 0
            and (step % self.cfg.solver_update_freq == 0)
        )
        solver_update_applied = bool(solver_update_due)
        solver_update_skip_reason: Optional[str] = None
        if (
            solver_update_applied
            and bool(getattr(self.cfg, "skip_solver_update_when_uninformative", True))
            and not solver_informative_all
        ):
            solver_update_applied = False
            solver_update_skip_reason = "uninformative_self_consistency"
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
        elif solver_update_due and solver_update_skip_reason is not None:
            self._append_jsonl(
                self.policy_updates_log_path,
                {
                    "step": step,
                    "role": "solver",
                    "source": "understanding",
                    "skipped": True,
                    "reason": solver_update_skip_reason,
                    "solver_margin": margin,
                    "entropy_nats": entropy_nats,
                },
            )

        proposer_stats = None
        proposer_skip_reason: Optional[str] = None
        proposer_update_due = self._is_proposer_update_due(step, phase="understanding")
        if proposer_update_due:
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
                proposer_skip_reason = "distributed_peer_empty_proposer_completion"
                self._append_jsonl(
                    self.policy_updates_log_path,
                    {
                        "step": step,
                        "role": "proposer",
                        "source": "understanding",
                        "skipped": True,
                        "reason": proposer_skip_reason,
                    },
                )
            self._sync_state_scalars()
        else:
            proposer_skip_reason = "update_not_due"

        step_dt = time.perf_counter() - step_t0
        record = {
            "step": step,
            "phase": "understanding",
            "image_path": meta.get("path"),
            "step_time_sec": step_dt,
            "question": question,
            "proposer_out": proposer_out,
            "solver_answers_raw": solver_answers_raw,
            "solver_answers_norm": solver_answers_norm,
            "solver_rewards_raw": solver_rewards_raw,
            "solver_rewards_soft": solver_rewards_soft,
            "majority_answer": maj_answer,
            "majority_count": maj_count,
            "majority_fraction": maj_frac,
            "solver_top1_prob": p1,
            "solver_top2_prob": p2,
            "solver_margin": margin,
            "solver_informative_local": solver_informative_local,
            "solver_informative_all": solver_informative_all,
            "solver_temperature_schedule": solver_temperatures,
            "solver_top_p_schedule": solver_top_ps,
            "entropy_nats": entropy_nats,
            "proposer_entropy_mu_used": proposer_entropy_mu_used,
            "proposer_reward": proposer_reward,
            "solver_baseline": self.solver_baseline,
            "proposer_baseline": self.proposer_baseline,
            "solver_update_due": solver_update_due,
            "solver_update_applied": solver_update_applied,
            "solver_update_skip_reason": solver_update_skip_reason,
            "solver_stats": solver_stats_list,
            "proposer_update_due": proposer_update_due,
            "proposer_skip_reason": proposer_skip_reason,
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
                "solver_top1_prob": p1,
                "solver_top2_prob": p2,
                "solver_margin": margin,
                "solver_informative_local": solver_informative_local,
                "solver_informative_all": solver_informative_all,
                "entropy_nats": entropy_nats,
                "solver_reward_soft_mean": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
                "proposer_entropy_mu_used": proposer_entropy_mu_used,
                "proposer_reward": proposer_reward,
            },
        )

        if self.is_main_process and step % self.cfg.log_every == 0:
            print(
                f"[Step {step:05d}][U] maj={maj_count}/{self.cfg.num_solver_samples} "
                f"maj_frac={maj_frac:.2f} H={entropy_nats:.3f} M={margin:.3f} "
                f"info={int(solver_informative_all)} P_R={proposer_reward:.3f} "
                f"dt={step_dt:.1f}s"
            )
            print(f"  Q: {question}")

        self._update_metric("u_majority_fraction", self._dist_mean(maj_frac))
        self._update_metric("u_entropy_nats", self._dist_mean(entropy_nats))
        self._update_metric("u_solver_margin", self._dist_mean(margin))
        self._update_metric("u_solver_informative", 1.0 if solver_informative_all else 0.0)
        self._update_metric("u_proposer_entropy_mu_used", self._dist_mean(proposer_entropy_mu_used))
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
                step_t0 = time.perf_counter()
                last_attempted_step = step
                image, meta = self._sample_image_for_step(step)
                phase_tag = "U"

                phase_idx = (step - 1) % cycle
                if phase_idx < cfg.understanding_steps_per_cycle:
                    self._understanding_step(step=step, image=image, meta=meta)
                else:
                    phase_tag = "G"
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
                            "proposer_update_due": out.get("proposer_update_due"),
                            "proposer_skip_reason": out.get("proposer_skip_reason"),
                            "proposer_stats": out.get("proposer_stats"),
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

                if self.is_main_process:
                    step_dt = time.perf_counter() - step_t0
                    print(f"[Step {step:05d}] phase={phase_tag} total_dt={step_dt:.1f}s")

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
