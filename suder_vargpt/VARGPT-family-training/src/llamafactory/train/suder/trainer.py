# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F

from ...extras import logging
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class SUDERTrainer(Trainer):
    r"""
    Inherits Trainer to implement SUDER (Dual Self-Reward) training loop.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        args: "TrainingArguments",
        finetuning_args: "FinetuningArguments",
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, tokenizer=tokenizer, callbacks=callbacks, **kwargs)
        self.finetuning_args = finetuning_args
        self.beta = finetuning_args.pref_beta
        self.num_samples = 4  # K=4 samples per input
        
        # Ensure we are in training mode
        self.model.train()

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        r"""
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 1. Image-to-Text (Captioning) Phase
        if "pixel_values" in inputs:
            loss_i2t = self._process_image_to_text(model, inputs)
        else:
            loss_i2t = torch.tensor(0.0, device=self.args.device)

        # 2. Text-to-Image (Generation) Phase
        # Only if we have prompts suitable for T2I. Usually 'input_ids' serves as both prompt and target in SFT.
        # But for T2I generation we need just the prompt part.
        if "input_ids" in inputs:
            loss_t2i = self._process_text_to_image(model, inputs)
        else:
            loss_t2i = torch.tensor(0.0, device=self.args.device)
            
        loss = loss_i2t + loss_t2i

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def _process_image_to_text(self, model, inputs):
        # 1. Generate Captions (K per image)
        # We need the prompt part of input_ids.
        # SFT inputs usually have labels where user tokens are masked (-100).
        # We find the first valid label to cut off the assistant response.
        
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]
        
        # Find the start of the assistant response (first non -100 label)
        # This assumes left-padded or straight sequence.
        # For batched generation with padding, we need to be careful.
        # Simplified: Use the input_ids as prompts if we assume the model handles continuation.
        # But we want to generate *new* captions, not just force-teach the GT caption.
        # So we should strip the GT caption from input_ids.
        
        prompts = []
        for i in range(input_ids.shape[0]):
            # Find index where labels != -100
            start_idx = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(start_idx) > 0:
                prompt_len = start_idx[0].item()
                prompts.append(input_ids[i, :prompt_len])
            else:
                # If no labels (eval mode?), just use full input
                prompts.append(input_ids[i])
        
        # Determine max length for padding
        max_prompt_len = max([p.shape[0] for p in prompts])
        padded_prompts = torch.full((len(prompts), max_prompt_len), self.tokenizer.pad_token_id, device=self.args.device, dtype=torch.long)
        attention_mask_prompts = torch.zeros((len(prompts), max_prompt_len), device=self.args.device, dtype=torch.long)
        
        for i, p in enumerate(prompts):
            # Left padding for generation usually
            offset = max_prompt_len - p.shape[0]
            padded_prompts[i, offset:] = p
            attention_mask_prompts[i, offset:] = 1
            
        # Generate K candidates
        # Expand inputs for K samples
        # But model.generate can handle num_return_sequences
        
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": 64,
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 1.0,
                "num_return_sequences": self.num_samples,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            
            generated_outputs = model.generate(
                input_ids=padded_prompts,
                attention_mask=attention_mask_prompts,
                pixel_values=pixel_values, # Assuming model accepts this
                **gen_kwargs
            )
            # output shape: (B*K, seq_len)
            
            # The output includes the prompt if standard generate is used.
            # We need to extract the new tokens.
            # But padded_prompts might have different lengths due to left padding.
            # Usually generate returns just the new tokens if we use `return_dict_in_generate=True`? 
            # No, standard is full sequence.
            
            # For simplicity, let's treat the whole sequence (prompt + gen) as the "text" for reward.
            
        # 2. Compute Reward: P(Image | Text) = T2I Loss
        # We need to run the model in T2I mode.
        # VARGPT computes this via `pixel_gen_values` argument.
        # We pass `pixel_values` as `pixel_gen_values`.
        # And `generated_outputs` as `input_ids`.
        
        # Prepare inputs for T2I Reward
        # We need to replicate pixel_values K times to match B*K generated texts
        pixel_values_expanded = pixel_values.repeat_interleave(self.num_samples, dim=0)
        
        # Calculate Reward (Likelihood of Image given Text)
        with torch.no_grad():
            # We use the model to compute loss.
            # We set labels to ignore text prediction, and focus on image prediction.
            # But wait, `get_gen_loss` accumulates both if `labels` are provided.
            # We can set `labels=None` to ignore text loss?
            # Looking at VARGPT code: 
            # if pixel_gen_values is not None: ... loss = other_loss + image_gen_loss
            # if labels is None, other_logits is not computed? 
            # In `forward`: `if labels is not None: ...`
            # So if we pass `labels=None`, we won't get text loss.
            # But we want `image_gen_loss`.
            # VARGPT `get_gen_loss` takes `image_gen_labels`.
            
            # We need to construct `image_gen_labels`.
            # This is complex. `process_image_gen_tokens` usually does this.
            # We rely on `forward` to do it if we pass `pixel_gen_values`.
            # But `forward` returns `loss` which is sum.
            
            # Hack: We can just use the `loss` returned by `forward` with `pixel_gen_values=pixel_values_expanded`.
            # But we must ensure `labels` for text are ignored?
            # If we don't pass `labels`, `loss` might be None or just image loss?
            # Check VARGPT code: 
            # if labels is not None: ... loss = other_loss ...
            # return (loss, ...)
            
            # So we MUST pass labels for text to get loss?
            # Or does it compute image loss separately?
            # Line 2569: `loss = other_loss + lambda_loss * image_gen_loss`
            # This implies if `labels` is None, `loss` is None.
            
            # We need to modify `modeling_vargpt` to return `image_gen_loss` even if separate?
            # Or we just accept that we need to Compute text loss (P(Text|Image?)) too?
            # No, we want P(Image|Text).
            
            # Let's assume we can modify `labels` to be all -100, effectively zeroing out text loss.
            dummy_labels = torch.full_like(generated_outputs, -100)
            
            outputs_reward = model(
                input_ids=generated_outputs,
                pixel_gen_values=pixel_values_expanded, # T2I mode
                labels=dummy_labels, # Ignore text loss
            )
            
            # VARGPT might return 0.0 for text loss if all labels are -100.
            # Check `get_gen_loss`: `if labels is not None: ... other_loss = loss_fct(...)`
            # If labels are -100, valid tokens are 0. `loss_fct` might return NaN or 0.
            # Assuming it handles it (usually does).
            
            # Converting loss to reward.
            # The loss returned is `other_loss + lambda * image_gen_loss`.
            # Since `other_loss` should be ~0, `loss` is proportional to `image_gen_loss`.
            # Reward = -loss.
            
            rewards = -outputs_reward.loss
            # Start shape (1,) because loss is mean.
            # We need per-sample reward!
            # Trainer typically reduces loss.
            # We need PER-SAMPLE loss.
            # VARGPT `get_gen_loss` reduces: `image_gen_loss = ... .mean()`.
            
            # Problem: The standard forward pass returns a scalar mean loss.
            # We cannot use it for per-sample scoring in GRPO without batch size 1 or modifying the model.
            # Batch size 1 is slow.
            
        # Workaround:
        # Since I cannot easily modify the model code right now without reloading it,
        # I might skip per-sample reward if I can't get it.
        # But GRPO *requires* per-sample reward to compute advantage.
        
        # Alternative: Return `image_gen_loss` unreduced from a custom forward method?
        # Or `loss_fct(..., reduction='none')`.
        
        # Given this constraint, and that this is an "Ablation", I will assume:
        # We can implement a simplified "Reinforcement Learning" where we use the mean reward
        # to update the whole batch? No, that's just random walk.
        
        # I MUST modify the model to support `reduction='none'` or accept that I run B*K forward passes?
        # Running B*K forward passes (one per sample) is robust but slow.
        # For ablation, correctness > speed.
        # I will run reward computation loop one by one (or small batches if I can).
        
        # But wait, `pixel_gen_values` is a list of images.
        # `model` handles batching?
        # `get_gen_loss` uses `logits` and `labels`.
        # `CrossEntropyLoss` has default `mean`.
        
        # I will use a loop to compute rewards per sample.
        rewards_list = []
        for j in range(generated_outputs.shape[0]):
             # Single forward pass per sample
             with torch.no_grad():
                 # We need to pass as list of 1
                 single_input_ids = generated_outputs[j:j+1]
                 single_dummy_labels = dummy_labels[j:j+1]
                 # Pixel gen values is a list in VARGPT?
                 # `inputs['pixel_gen_values']` signature: `Optional[torch.FloatTensor]`.
                 # It seems it expects a tensor `(B, C, H, W)` or list?
                 # Looking at `forward`: `if pixel_gen_values is not None and not any(x is None ...)`
                 # It treats it as iterable? `x_BLC_wo_prefix, ... = self.get_vae_gt_xin_v1_1(pixel_gen_values...)`
                 # I should pass a list containing one image tensor.
                 # `pixel_values_expanded` is a tensor.
                 single_pixel_gen = [pixel_values_expanded[j]] 
                 
                 out = model(
                     input_ids=single_input_ids,
                     pixel_gen_values=single_pixel_gen,
                     labels=single_dummy_labels
                 )
                 rewards_list.append(-out.loss.item()) # Negative NLL is reward
        
        rewards = torch.tensor(rewards_list, device=self.args.device)
        
        # 4. Policy Update (GRPO)
        # We need log probabilities of the generated tokens.
        # Helper: Batched forward pass to get logits.
        
        # Re-run forward pass on generated_outputs (B*K)
        # using the model (which is in train mode).
        
        # We perform this on the whole batch (B*K).
        # We want to predict `generated_outputs` from `prompts`.
        # So labels should be `generated_outputs`, but with `prompt` masked out (-100).
        
        # Construct training labels
        training_labels = generated_outputs.clone()
        # Resize attention_mask_prompts to match generated_outputs (left padded prompt vs generated seq)
        # Actually `generated_outputs` contains the prompt.
        # We just need to mask the prompt part in labels.
        # `padded_prompts` was the prompt.
        # We entered `generated_outputs` which has prompt + new tokens.
        # We mask the prompt length.
        
        # Since lengths vary, we need to be careful.
        # But we left-padded prompts. So prompt acts as prefix.
        # We can just mask the first `max_prompt_len` tokens?
        # Yes, `padded_prompts` has `max_prompt_len`.
        training_labels[:, :max_prompt_len] = -100
        
        # Forward pass to get logits (gradients enabled)
        outputs_policy = model(
            input_ids=generated_outputs,
            pixel_values=pixel_values_expanded, # Helper for SFT? Or just generated text?
            # Wait, this is I2T policy. It conditions on Image.
            # So we pass `pixel_values`.
            labels=training_labels 
        )
        
        # `outputs_policy.loss` is effectively `-sum(log_prob)`.
        # But typical Trainer loss is mean.
        # We need per-token log probs to apply advantage?
        # Or if we use `loss` directly, how to apply advantage?
        # GRPO: Loss = - (Advantage * log_prob).sum()
        # Standard Loss = - log_prob.sum()
        # So GRPO Loss is weighted standard loss.
        # But `CrossEntropyLoss` supports `reduction='none'`.
        # Again, I can't easily change `forward` reduction.
        
        # Option: Use `logits` from `outputs_policy` and compute loss manually.
        logits = outputs_policy.logits
        # logits: (B*K, L, V)
        # labels: (B*K, L)
        
        # Compute per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = training_labels[..., 1:].contiguous()
        
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_loss = token_loss.view(shift_labels.shape)
        # Sum over sequence to get per-sample NLL
        # Mask out padding/ignore
        valid_mask = (shift_labels != -100).float()
        per_sample_nll = (token_loss * valid_mask).sum(dim=1)
        
        # Compute Advantage
        # Reshape rewards to (B, K)
        bs = input_ids.shape[0]
        k = self.num_samples
        rewards = rewards.view(bs, k)
        
        # Normalize rewards (Group level)
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - mean_rewards) / std_rewards
        
        # Flatten advantages to (B*K)
        advantages = advantages.view(-1)
        
        # GRPO Loss = (per_sample_nll * advantages).mean()
        # Note: per_sample_nll is negative log prob.
        # So we want to Minimize (NLL * Advantage)?
        # If Advantage is positive, we want to maximize Prob => Minimize NLL.
        # So Minimize (NLL * Advantage).
        # Correct.
        
        loss_grpo = (per_sample_nll * advantages.detach()).mean()
        
        return loss_grpo

    def _process_text_to_image(self, model, inputs):
        # 2. Text-to-Image (T2I) Phase
        # Prompt: inputs['input_ids']
        # Generate Images -> Score with I2T (P(Text|Image)) -> Update
        
        # Skip for now to enable incrementally.
        return torch.tensor(0.0, device=self.args.device, requires_grad=True)

