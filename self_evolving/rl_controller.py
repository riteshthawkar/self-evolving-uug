"""
RL Controller for Self-Evolving Training.
Implements KL-regularized REINFORCE with adaptive β and EMA baselines.
Based on EvoLMM's training infrastructure.
"""

import copy
import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class RLControllerConfig:
    """Configuration for RL controller."""
    # KL regularization
    kl_coeff: float = 0.01
    kl_target: float = 1.0
    kl_horizon: int = 1000
    kl_adaptive: bool = True
    kl_min: float = 0.001
    kl_max: float = 0.1
    
    # Baseline (EMA)
    baseline_ema: float = 0.95
    use_baseline: bool = True
    
    # Clipping
    pg_clip: float = 0.2  # PPO-style clipping (optional)
    use_clipping: bool = False
    
    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    
    # Proposer schedule
    proposer_update_every: int = 5
    
    # Entropy bonus (optional)
    entropy_coeff: float = 0.01
    use_entropy_bonus: bool = True


class RLController:
    """
    KL-regularized REINFORCE controller for self-evolving training.
    
    Features:
    - Reference policy for KL computation
    - Adaptive β coefficient (as in EvoLMM)
    - EMA baseline for variance reduction
    - Support for multiple roles (proposer, solver, generator)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[RLControllerConfig] = None,
    ):
        """
        Initialize RL controller.
        
        Args:
            model: The policy model (will create reference copy)
            config: RL controller configuration
        """
        self.config = config or RLControllerConfig()
        
        # Create frozen reference policy
        self.reference_model = copy.deepcopy(model)
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        
        # β coefficients per role
        self.beta = {
            'proposer': self.config.kl_coeff,
            'solver': self.config.kl_coeff,
            'generator': self.config.kl_coeff,
        }
        
        # EMA baselines per role
        self.baseline = {
            'proposer': 0.0,
            'solver': 0.0,
            'generator': 0.0,
        }
        
        # Running KL estimates for adaptive β
        self.kl_running = {
            'proposer': 0.0,
            'solver': 0.0,
            'generator': 0.0,
        }
        
        # Step counters
        self.step_count = 0
        self.role_step_count = {
            'proposer': 0,
            'solver': 0,
            'generator': 0,
        }
        
        # Statistics for logging
        self.stats = {
            'kl': {'proposer': [], 'solver': [], 'generator': []},
            'reward': {'proposer': [], 'solver': [], 'generator': []},
            'beta': {'proposer': [], 'solver': [], 'generator': []},
            'loss': {'proposer': [], 'solver': [], 'generator': []},
        }

    def compute_kl_divergence(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role: str = 'solver',
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute per-token KL divergence between policy and reference.
        
        Args:
            logits: Policy logits [B, L, V]
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            role: Role name ('proposer', 'solver', 'generator')
            pixel_values: Optional image inputs
            
        Returns:
            Per-token KL divergence [B, L]
        """
        with torch.no_grad():
            # Get reference logits
            ref_outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )
            ref_logits = ref_outputs.logits
        
        # Compute KL(π || π_ref) per token
        # KL = Σ π(a) * (log π(a) - log π_ref(a))
        log_probs = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Get the log prob differences for actual tokens
        probs = F.softmax(logits, dim=-1)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        
        return kl

    def compute_per_token_logprobs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities.
        
        Args:
            logits: Model logits [B, L, V]
            labels: Token labels [B, L]
            mask: Optional mask [B, L]
            
        Returns:
            Log probabilities [B, L]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        # Shift labels by 1 for causal LM
        labels_shifted = labels[:, 1:].contiguous()
        log_probs_shifted = log_probs[:, :-1, :].contiguous()
        
        # Gather
        token_log_probs = log_probs_shifted.gather(
            dim=-1,
            index=labels_shifted.unsqueeze(-1)
        ).squeeze(-1)
        
        if mask is not None:
            mask_shifted = mask[:, 1:].contiguous()
            token_log_probs = token_log_probs * mask_shifted
        
        return token_log_probs

    def compute_reinforce_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        kl_per_token: torch.Tensor,
        mask: torch.Tensor,
        role: str = 'solver',
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute KL-regularized REINFORCE loss.
        
        Args:
            log_probs: Per-token log probabilities [B, L]
            rewards: Rewards [B] or [B, L]
            kl_per_token: Per-token KL divergence [B, L]
            mask: Token mask [B, L]
            role: Role name
            
        Returns:
            Tuple of (loss, stats_dict)
        """
        batch_size = log_probs.size(0)
        
        # Reshape rewards if needed
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1).expand_as(log_probs)
        
        # Update baseline (EMA)
        if self.config.use_baseline:
            reward_mean = rewards[:, 0].mean().item()  # Use first token's reward
            self.baseline[role] = (
                self.config.baseline_ema * self.baseline[role] +
                (1 - self.config.baseline_ema) * reward_mean
            )
            baseline = self.baseline[role]
        else:
            baseline = 0.0
        
        # Compute advantage
        advantage = rewards - baseline
        
        # Normalize rewards (optional)
        if self.config.normalize_rewards:
            advantage_std = advantage.std() + 1e-8
            advantage = advantage / advantage_std
        
        # Clip advantage (optional)
        if self.config.reward_clip > 0:
            advantage = torch.clamp(advantage, -self.config.reward_clip, self.config.reward_clip)
        
        # Policy gradient loss: -E[A * log π]
        pg_loss = -(advantage * log_probs * mask).sum() / mask.sum()
        
        # KL penalty
        kl_mean = (kl_per_token * mask).sum() / mask.sum()
        kl_loss = self.beta[role] * kl_mean
        
        # Entropy bonus (optional)
        if self.config.use_entropy_bonus:
            entropy = -(log_probs * mask).sum() / mask.sum()
            entropy_loss = -self.config.entropy_coeff * entropy
        else:
            entropy_loss = torch.tensor(0.0, device=log_probs.device)
        
        # Total loss
        total_loss = pg_loss + kl_loss + entropy_loss
        
        # Update running KL estimate
        self.kl_running[role] = (
            0.9 * self.kl_running[role] + 0.1 * kl_mean.item()
        )
        
        # Adaptive β update
        if self.config.kl_adaptive:
            self._update_beta(role)
        
        # Stats
        stats = {
            f'{role}/pg_loss': pg_loss.item(),
            f'{role}/kl_loss': kl_loss.item(),
            f'{role}/kl_mean': kl_mean.item(),
            f'{role}/beta': self.beta[role],
            f'{role}/baseline': baseline,
            f'{role}/advantage_mean': advantage.mean().item(),
        }
        
        # Update stats history
        self.stats['kl'][role].append(kl_mean.item())
        self.stats['beta'][role].append(self.beta[role])
        self.stats['loss'][role].append(total_loss.item())
        
        return total_loss, stats

    def _update_beta(self, role: str):
        """
        Adaptive β update following EvoLMM.
        
        If KL > target, increase β to penalize more
        If KL < target, decrease β to allow more exploration
        """
        kl_current = self.kl_running[role]
        kl_target = self.config.kl_target
        
        # Proportional control
        if kl_current > 1.5 * kl_target:
            # KL too high, increase penalty
            self.beta[role] = min(self.beta[role] * 1.5, self.config.kl_max)
        elif kl_current < 0.5 * kl_target:
            # KL too low, decrease penalty
            self.beta[role] = max(self.beta[role] / 1.5, self.config.kl_min)

    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        kl_per_token: torch.Tensor,
        mask: torch.Tensor,
        role: str = 'generator',
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO (Group Relative Policy Optimization) loss.
        
        For generation with multiple samples per prompt:
        - Advantage is relative to group mean
        - Rewards are [B, G] for G samples per prompt
        
        Args:
            log_probs: Per-token log probs [B, G, L]
            rewards: Rewards [B, G]
            kl_per_token: Per-token KL [B, G, L]
            mask: Token mask [B, G, L]
            role: Role name (typically 'generator')
            
        Returns:
            Tuple of (loss, stats_dict)
        """
        B, G, L = log_probs.shape
        
        # Group-relative advantage
        reward_mean = rewards.mean(dim=1, keepdim=True)  # [B, 1]
        advantage = rewards - reward_mean  # [B, G]
        
        # Normalize within batch
        if self.config.normalize_rewards:
            adv_std = advantage.std() + 1e-8
            advantage = advantage / adv_std
        
        # Expand advantage to token level
        advantage = advantage.unsqueeze(-1).expand_as(log_probs)  # [B, G, L]
        
        # Policy gradient loss
        pg_loss = -(advantage * log_probs * mask).sum() / mask.sum()
        
        # KL penalty
        kl_mean = (kl_per_token * mask).sum() / mask.sum()
        kl_loss = self.beta[role] * kl_mean
        
        # Total loss
        total_loss = pg_loss + kl_loss
        
        # Update running estimates
        self.kl_running[role] = 0.9 * self.kl_running[role] + 0.1 * kl_mean.item()
        
        # Adaptive β
        if self.config.kl_adaptive:
            self._update_beta(role)
        
        stats = {
            f'{role}/pg_loss': pg_loss.item(),
            f'{role}/kl_loss': kl_loss.item(),
            f'{role}/kl_mean': kl_mean.item(),
            f'{role}/beta': self.beta[role],
            f'{role}/reward_mean': rewards.mean().item(),
            f'{role}/reward_std': rewards.std().item(),
            f'{role}/advantage_mean': advantage.mean().item(),
        }
        
        return total_loss, stats

    def should_update_proposer(self) -> bool:
        """Check if proposer should be updated this step."""
        return self.step_count % self.config.proposer_update_every == 0

    def step(self):
        """Increment step counter."""
        self.step_count += 1

    def update_reference_model(self, model: torch.nn.Module, ema_decay: float = 0.99):
        """
        Update reference model with EMA from current model.
        
        Args:
            model: Current policy model
            ema_decay: EMA decay coefficient
        """
        with torch.no_grad():
            for p_ref, p_new in zip(
                self.reference_model.parameters(),
                model.parameters()
            ):
                p_ref.data = ema_decay * p_ref.data + (1 - ema_decay) * p_new.data

    def reset_reference_model(self, model: torch.nn.Module):
        """Reset reference model to current model state."""
        self.reference_model = copy.deepcopy(model)
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

    def get_stats(self) -> Dict[str, float]:
        """Get aggregated statistics."""
        result = {}
        for stat_type, role_stats in self.stats.items():
            for role, values in role_stats.items():
                if values:
                    result[f'{stat_type}/{role}_mean'] = sum(values) / len(values)
        return result

    def reset_stats(self):
        """Reset statistics."""
        for stat_type in self.stats:
            for role in self.stats[stat_type]:
                self.stats[stat_type][role] = []


class EvoLMMReward:
    """
    EvoLMM-style reward computation.
    
    For solver: agreement-based continuous reward
    For proposer: entropy-band Gaussian reward
    """
    
    def __init__(
        self,
        entropy_target: float = 1.0,
        entropy_sigma: float = 0.5,
        agreement_threshold: float = 0.5,
    ):
        """
        Initialize reward computer.
        
        Args:
            entropy_target: Target entropy for proposer reward
            entropy_sigma: Sigma for Gaussian entropy band
            agreement_threshold: Threshold for solver agreement
        """
        self.entropy_target = entropy_target
        self.entropy_sigma = entropy_sigma
        self.agreement_threshold = agreement_threshold

    def compute_solver_reward(
        self,
        answers: List[str],
        normalize: bool = True,
    ) -> Tuple[float, float]:
        """
        Compute solver reward based on answer agreement.
        
        Args:
            answers: List of N answer strings
            normalize: Whether to normalize answers
            
        Returns:
            Tuple of (reward, entropy)
        """
        if not answers:
            return 0.0, 0.0
        
        # Normalize answers
        if normalize:
            answers = [self._normalize_answer(a) for a in answers]
        
        # Count unique answers
        from collections import Counter
        counter = Counter(answers)
        total = len(answers)
        
        # Compute probabilities
        probs = [count / total for count in counter.values()]
        
        # Compute entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        
        # Agreement reward: proportion of most common answer
        max_count = max(counter.values())
        agreement = max_count / total
        
        # Continuous reward (higher agreement = higher reward)
        # Following EvoLMM's formulation
        reward = agreement
        
        return reward, entropy

    def compute_proposer_reward(
        self,
        solver_entropy: float,
    ) -> float:
        """
        Compute proposer reward using entropy band.
        
        Gaussian reward centered on target entropy.
        Questions that are too easy (low entropy) or too hard (high entropy)
        receive lower rewards.
        
        Args:
            solver_entropy: Entropy of solver answers
            
        Returns:
            Proposer reward
        """
        # Gaussian reward around target entropy
        diff = solver_entropy - self.entropy_target
        reward = math.exp(-0.5 * (diff / self.entropy_sigma) ** 2)
        
        return reward

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        import re
        answer = answer.lower().strip()
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        # Normalize whitespace
        answer = ' '.join(answer.split())
        # Handle common patterns
        if answer in ['yes', 'true', '1', 'correct']:
            return 'yes'
        if answer in ['no', 'false', '0', 'incorrect']:
            return 'no'
        return answer


def create_rl_controller(
    model: torch.nn.Module,
    kl_coeff: float = 0.01,
    kl_target: float = 1.0,
    proposer_update_every: int = 5,
) -> RLController:
    """
    Convenience function to create RL controller with sensible defaults.
    
    Args:
        model: Policy model
        kl_coeff: Initial KL coefficient
        kl_target: Target KL divergence
        proposer_update_every: Update proposer every N steps
        
    Returns:
        Configured RLController
    """
    config = RLControllerConfig(
        kl_coeff=kl_coeff,
        kl_target=kl_target,
        proposer_update_every=proposer_update_every,
    )
    return RLController(model, config)
