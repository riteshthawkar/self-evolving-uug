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

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""
    Arguments pertaining to the PPO, DPO and KTO training.
    """

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )


@dataclass
class GaloreArguments:
    r"""
    Arguments pertaining to the GaLore algorithm.
    """

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=0.25,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class BAdamArgument:
    r"""
    Arguments pertaining to the BAdam optimizer.
    """

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: Optional[int] = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Optional[Literal["ascending", "descending", "random", "fixed"]] = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class FinetuningArguments(FreezeArguments, LoraArguments, RLHFArguments, GaloreArguments, BAdamArgument):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto", "suder", "self_evolving"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze vision tower in MLLM training."},
    )
    train_mm_proj_only: bool = field(
        default=False,
        metadata={"help": "Whether or not to train the multimodal projector for MLLM only."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    vargpt_train_stage: int = field(
        default=0,
        metadata={"help": "Which stage to perform in Vargpt training."},
    )
    vargpt_version: str = field(
        default="qwen2vl-v1.0",
        metadata={"help": "Which stage to perform in Vargpt training."},
    )

    # ── Self-Evolving Framework Config ──────────────────────────────────
    se_understanding_steps_per_cycle: int = field(
        default=3,
        metadata={"help": "Number of understanding steps per U/G cycle."},
    )
    se_generation_steps_per_cycle: int = field(
        default=2,
        metadata={"help": "Number of generation steps per U/G cycle."},
    )
    se_total_steps: int = field(
        default=10000,
        metadata={"help": "Total training steps for self-evolving."},
    )
    se_save_every: int = field(
        default=500,
        metadata={"help": "Save a self-evolving checkpoint every N steps."},
    )
    se_replay_buffer_size: int = field(
        default=1,
        metadata={"help": "Max size of the replay buffer for generated images."},
    )
    se_gen_mix_ratio_max: float = field(
        default=0.0,
        metadata={"help": "Max ratio of generated images mixed into U-steps."},
    )
    se_imageless_proposer_mode: bool = field(
        default=False,
        metadata={"help": "Use text-only topics for proposer (no source images)."},
    )
    se_gen_reward_mode: str = field(
        default="clip",
        metadata={"help": "Legacy compatibility flag; generation uses the shared BLIP-style reward scorer."},
    )
    se_num_generations: int = field(
        default=3,
        metadata={"help": "Number of candidate images per G-step."},
    )
    se_reward_spec_weight: float = field(
        default=0.65,
        metadata={"help": "Weight of QA/spec fidelity in generation reward."},
    )
    se_reward_cycle_weight: float = field(
        default=0.20,
        metadata={"help": "Weight of cycle consistency in generation reward."},
    )
    se_reward_diversity_weight: float = field(
        default=0.10,
        metadata={"help": "Weight of inter-candidate diversity in generation reward."},
    )
    se_reward_contradiction_weight: float = field(
        default=0.20,
        metadata={"help": "Penalty weight for yes/no contradictions in generation reward."},
    )
    se_min_spec_quality_for_update: float = field(
        default=0.35,
        metadata={"help": "Minimum proposer spec quality required for G-step updates."},
    )
    se_min_spec_qa_pairs: int = field(
        default=2,
        metadata={"help": "Minimum valid QA pairs required for generation specs."},
    )
    se_max_expected_words: int = field(
        default=8,
        metadata={"help": "Maximum allowed words in a generation spec expected answer."},
    )
    se_max_question_words: int = field(
        default=24,
        metadata={"help": "Maximum allowed words in a generation spec question."},
    )
    se_use_ref_answer_scoring: bool = field(
        default=False,
        metadata={"help": "Use reference-answer log-prob scoring instead of multi-component generation reward."},
    )
    se_lr: float = field(
        default=1e-6,
        metadata={"help": "Learning rate for self-evolving role updaters."},
    )
    se_proposer_gen_reward_enabled: bool = field(
        default=True,
        metadata={"help": "Update proposer with generation quality reward."},
    )
    se_image_folder: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a folder of images (with optional subfolders). "
                "When set, the self-evolving trainer loads images directly "
                "from this folder instead of requiring a JSON dataset. "
                "Supports .jpg, .jpeg, .png, .webp, .bmp, .tiff."
            )
        },
    )
    se_num_solver_samples: int = field(
        default=7,
        metadata={"help": "Number of solver samples per understanding step."},
    )
    se_proposer_num_candidates: int = field(
        default=5,
        metadata={"help": "Number of proposer candidate questions per understanding step."},
    )
    se_proposer_spot_check_samples: int = field(
        default=3,
        metadata={"help": "Solver samples used to spot-check proposer candidates."},
    )
    se_proposer_question_quality_min_score: float = field(
        default=0.60,
        metadata={"help": "Minimum combined rubric/model-judge score for accepting proposer questions."},
    )
    se_proposer_question_structural_min_score: float = field(
        default=0.50,
        metadata={"help": "Minimum structural rubric score before model-judge validation."},
    )
    se_proposer_question_model_judge_enabled: bool = field(
        default=True,
        metadata={"help": "Use the current VLM as a short rubric judge for proposer question quality."},
    )
    se_proposer_question_model_judge_weight: float = field(
        default=0.15,
        metadata={"help": "Weight of the model-judge score in the combined proposer question quality score."},
    )
    se_solver_use_temperature_mix: bool = field(
        default=True,
        metadata={"help": "Use solver temperature/top-p schedules across samples."},
    )
    se_solver_use_forced_choice_from_proposer: bool = field(
        default=True,
        metadata={"help": "Force solver answers to choose from proposer-provided answer options when available."},
    )
    se_solver_temp_min: float = field(
        default=0.5,
        metadata={"help": "Minimum solver temperature in mixed decoding."},
    )
    se_solver_temp_max: float = field(
        default=2.5,
        metadata={"help": "Maximum solver temperature in mixed decoding."},
    )
    se_solver_top_p_min: float = field(
        default=0.3,
        metadata={"help": "Minimum solver top-p in mixed decoding."},
    )
    se_solver_top_p_max: float = field(
        default=1.0,
        metadata={"help": "Maximum solver top-p in mixed decoding."},
    )
    se_solver_skip_update_on_easy: bool = field(
        default=True,
        metadata={"help": "Skip solver updates on unanimous easy cases."},
    )
    se_easy_update_majority_frac_threshold: float = field(
        default=0.85,
        metadata={"help": "Majority fraction threshold for easy-case solver skip."},
    )
    se_difficulty_sampler_enabled: bool = field(
        default=True,
        metadata={"help": "Enable difficulty target sampling for proposer prompts."},
    )
    se_difficulty_sampler_min_samples: int = field(
        default=8,
        metadata={"help": "Minimum observed samples before deficit-based targeting."},
    )
    se_difficulty_target_easy: float = field(
        default=0.0,
        metadata={"help": "Target fraction for easy bucket."},
    )
    se_difficulty_target_medium: float = field(
        default=0.7,
        metadata={"help": "Target fraction for medium bucket."},
    )
    se_difficulty_target_hard: float = field(
        default=0.3,
        metadata={"help": "Target fraction for hard bucket."},
    )
    se_proposer_warm_start_enabled: bool = field(
        default=True,
        metadata={"help": "Enable entropy-free proposer warm-start."},
    )
    se_proposer_warm_start_max_steps: int = field(
        default=30,
        metadata={"help": "Max understanding steps for warm-start."},
    )
    se_proposer_warm_start_exit_window: int = field(
        default=5,
        metadata={"help": "Window size for warm-start exit entropy mean."},
    )
    se_proposer_warm_start_exit_consecutive: int = field(
        default=2,
        metadata={"help": "Consecutive exit-window passes required to finish warm-start."},
    )
    se_proposer_warm_start_entropy_exit_threshold: float = field(
        default=0.10,
        metadata={"help": "Entropy threshold to exit warm-start."},
    )
    se_hardness_debt_enabled: bool = field(
        default=True,
        metadata={"help": "Enable hardness debt controller."},
    )
    se_hardness_debt_inc_easy: float = field(
        default=1.5,
        metadata={"help": "Debt increase on easy observed bucket."},
    )
    se_hardness_debt_dec_non_easy: float = field(
        default=1.0,
        metadata={"help": "Debt decrease on medium/hard observed bucket."},
    )
    se_hardness_debt_hard_recovery_threshold: float = field(
        default=3.0,
        metadata={"help": "Debt threshold to enter hard-recovery targeting."},
    )
    se_all_easy_explore_trigger: int = field(
        default=2,
        metadata={"help": "Consecutive all-easy groups required to trigger forced exploration."},
    )
    se_all_easy_explore_steps: int = field(
        default=16,
        metadata={"help": "Number of forced-exploration understanding steps once triggered."},
    )
    se_all_easy_explore_num_candidates: int = field(
        default=6,
        metadata={"help": "Candidate count during forced exploration."},
    )
    se_proposer_early_failfast_enabled: bool = field(
        default=True,
        metadata={"help": "Enable proposer early failfast health checks."},
    )
    se_proposer_early_failfast_stop: bool = field(
        default=False,
        metadata={"help": "Hard-stop when early failfast unhealthy condition triggers."},
    )
    se_proposer_early_failfast_recover: bool = field(
        default=True,
        metadata={"help": "Arm forced exploration instead of stopping on early failfast trigger."},
    )
    se_fail_on_step_error: bool = field(
        default=True,
        metadata={"help": "Raise on self-evolving step errors instead of continuing."},
    )
    se_max_consecutive_step_errors: int = field(
        default=0,
        metadata={"help": "Maximum consecutive self-evolving step errors before failing."},
    )
    se_max_total_step_errors: int = field(
        default=0,
        metadata={"help": "Maximum total self-evolving step errors before failing."},
    )
    se_generation_failfast_enabled: bool = field(
        default=True,
        metadata={"help": "Enable generation health fail-fast checks."},
    )
    se_generation_failfast_consecutive_skips: int = field(
        default=5,
        metadata={"help": "Consecutive unhealthy generation steps before fail-fast triggers."},
    )
    se_generation_failfast_min_success_rate: float = field(
        default=0.10,
        metadata={"help": "Minimum generation success rate required by the fail-fast window."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: List[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: Optional[List[str]] = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: List[str] = split_arg(self.lora_target)
        self.additional_target: Optional[List[str]] = split_arg(self.additional_target)
        self.galore_target: List[str] = split_arg(self.galore_target)
        self.freeze_vision_tower = self.freeze_vision_tower or self.train_mm_proj_only
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore or BAdam together.")

        if self.use_galore and self.use_badam:
            raise ValueError("Cannot use GaLore with BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.train_mm_proj_only and self.finetuning_type != "full":
            raise ValueError("`train_mm_proj_only` is only valid for full training.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")
