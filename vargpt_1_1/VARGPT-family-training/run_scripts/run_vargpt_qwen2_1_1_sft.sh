export HF_ENDPOINT=https://hf-mirror.com

FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0  MASTER_ADDR=127.0.0.1 MASTER_PORT=39547 llamafactory-cli train examples/train_vargpt_qwen2vl_1_1/vargpt_pretraining_7b_v1_1_stage3.yaml

