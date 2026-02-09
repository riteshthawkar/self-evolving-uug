#!/bin/bash

python -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model blip3o \
    --model_args pretrained="your/model/path/" \
    --tasks  mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix blip3o \
    --output_path ./logs/


