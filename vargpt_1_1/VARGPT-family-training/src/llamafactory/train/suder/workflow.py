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

from typing import TYPE_CHECKING, List, Optional

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import SUDERTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_suder(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    if finetuning_args.vargpt_version == "qwen2vl-v1.0":
        from visionllm.vargpt.prepare_vargpt_v1 import prepare_vargpt_qwen2vl
        prepare_vargpt_qwen2vl()
    elif finetuning_args.vargpt_version == "llava-v1.0":
        from visionllm.vargpt_llava.prepare_vargpt_llava import prepare_vargpt_llava
        prepare_vargpt_llava()
    elif finetuning_args.vargpt_version == "qwen2vl-v1.1":
        from visionllm.vargpt_qwen_v1_1.prepare_vargpt_v1_1 import prepare_vargpt_qwen2vl_v1_1
        prepare_vargpt_qwen2vl_v1_1(base_model_id=model_args.model_name_or_path)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # We use "sft" stage dataset processing initially as SUDER starts from SFT data but generates its own.
    # OR we can treat it like PPO/DPO where we just need prompts.
    # For now, let's load it as 'sft' data to get inputs.
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # Initialize our Trainer
    trainer = SUDERTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
