
### Training
You can use the following command to training vis SFT:
```bash
bash run_scripts/run_vargpt_qwen2_1_1_sft.sh
```

NOTE:  The demo and configuration of the data are completely consistent with [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory). You can freely configure, modify the model architecture, or use any training strategy supported by LLaMA Factory for training.

### Inference
You can use the following command to perform batch image generation inference:
```bash
bash run_scripts/run_eval_vargpt_v1_1.sh
```
The task of batch image editing can be achieved through the following commands:
```bash
bash run_scripts/run_eval_vargpt_v1_1_edit.sh
```
Note:  You need to create and specify the path for image generation in the bash script.

