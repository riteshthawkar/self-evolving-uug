

#!/bin/bash

# 1. Force Python to flush logs to file immediately
export PYTHONUNBUFFERED=1


# Define log file name
LOG_FILE="evolmm_eval_run_our.log"

echo "Starting Evaluation 1..." | tee $LOG_FILE

# 3. First Process (Write to file)
accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
  --model qwen2_5_vl_our \
  --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/home/omkar/ritesh/EvoLMM/checkpoints/solver,max_pixels=12845056,interleave_visuals=False \
  --tasks refcocog_bbox_rec_val,refcocog_bbox_rec_test \
  --batch_size 1 \
  --output_path /home/omkar/ritesh/EvoLMM/logs \
  --use_cache /home/omkar/ritesh/EvoLMM/logs \
  2>&1 | tee -a $LOG_FILE

# echo "Finished Evaluation 1. Starting Evaluation 2..." | tee -a $LOG_FILE

# 4. Second Process (Append to file using -a)
# accelerate launch --num_processes=7 --main_process_port=12347 -m lmms_eval \
#   --model qwen2_5_vl \
#   --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,interleave_visuals=False \
#   --tasks refcocog_bbox_rec_val,refcocog_bbox_rec_test \
#   --batch_size 1 \
#   --output_path /home/omkar/ritesh/EvoLMM/logs \
#   --use_cache /home/omkar/ritesh/EvoLMM/logs \
#   2>&1 | tee -a $LOG_FILE

# echo "All processes completed. Logs saved to $LOG_FILE"














# python - <<'PY'
# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id="omkarthawakar/EvoLMM",
#     local_dir="/home/omkar/ritesh/EvoLMM/checkpoints",
#     local_dir_use_symlinks=False
# )
# print("Downloaded.")
# PY




# qwen2_5_vl_our (pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/home/omkar/ritesh/EvoLMM/checkpoints/solver,max_pixels=12845056,interleave_visuals=False), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
# | Tasks |Version|Filter|n-shot|        Metric         |   |Value |   |Stderr|
# |-------|-------|------|-----:|-----------------------|---|-----:|---|-----:|
# |chartqa|Yaml   |none  |     0|relaxed_augmented_split|↑  |0.9488|±  |0.0062|
# |chartqa|Yaml   |none  |     0|relaxed_human_split    |↑  |0.7816|±  |0.0117|
# |chartqa|Yaml   |none  |     0|relaxed_overall        |↑  |0.8652|±  |0.0068|





# 2026-01-28 00:46:08 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_aggregated:188 - Saving results aggregated
# qwen2_5_vl_our (pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/home/omkar/ritesh/EvoLMM/checkpoints/solver,max_pixels=12845056,interleave_visuals=False), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
# |        Tasks        |Version|Filter|n-shot|    Metric     |   |Value |   |Stderr|
# |---------------------|-------|------|-----:|---------------|---|-----:|---|------|
# |refcoco              |    N/A|      |      |               |   |      |   |      |
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0425|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0164|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0066|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0028|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0072|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_METEOR |↑  |0.0788|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0765|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0414|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0155|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0061|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0025|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0032|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_METEOR |↑  |0.0753|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0734|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0432|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0166|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0069|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0030|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0083|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_METEOR |↑  |0.0816|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0775|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0421|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0162|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0065|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0027|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0062|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_METEOR |↑  |0.0774|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0741|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0412|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0146|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0055|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0022|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0085|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_METEOR |↑  |0.0731|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0749|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0410|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0140|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0052|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0021|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0030|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_METEOR |↑  |0.0709|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0730|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0412|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0151|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0061|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0027|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0127|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_METEOR |↑  |0.0757|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0762|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0412|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0147|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0058|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0024|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0081|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_METEOR |↑  |0.0730|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0749|±  |   N/A|

# [rank0]:[W128 00:46:10.057525479 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
# evaluate.sh: line 15: M/logs: No such file or directory
# (blip3o) omkar@gigabyte-G492-HA0-00:~/ritesh/EvoLMM/Evaluation/lmms-eval$ 





# 2026-01-28 00:46:08 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_aggregated:188 - Saving results aggregated
# qwen2_5_vl_our (pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/home/omkar/ritesh/EvoLMM/checkpoints/solver,max_pixels=12845056,interleave_visuals=False), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
# |        Tasks        |Version|Filter|n-shot|    Metric     |   |Value |   |Stderr|
# |---------------------|-------|------|-----:|---------------|---|-----:|---|------|
# |refcoco              |    N/A|      |      |               |   |      |   |      |
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0425|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0164|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0066|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0028|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0072|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_METEOR |↑  |0.0788|±  |   N/A|
# | - refcoco_bbox_test |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0765|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0414|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0155|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0061|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0025|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0032|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_METEOR |↑  |0.0753|±  |   N/A|
# | - refcoco_bbox_testA|0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0734|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0432|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0166|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0069|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0030|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0083|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_METEOR |↑  |0.0816|±  |   N/A|
# | - refcoco_bbox_testB|0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0775|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0421|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0162|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0065|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0027|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0062|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_METEOR |↑  |0.0774|±  |   N/A|
# | - refcoco_bbox_val  |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0741|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0412|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0146|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0055|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0022|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0085|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_METEOR |↑  |0.0731|±  |   N/A|
# | - refcoco_seg_test  |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0749|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0410|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0140|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0052|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0021|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0030|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_METEOR |↑  |0.0709|±  |   N/A|
# | - refcoco_seg_testA |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0730|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0412|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0151|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0061|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0027|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0127|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_METEOR |↑  |0.0757|±  |   N/A|
# | - refcoco_seg_testB |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0762|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_Bleu_1 |↑  |0.0412|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_Bleu_2 |↑  |0.0147|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_Bleu_3 |↑  |0.0058|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_Bleu_4 |↑  |0.0024|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_CIDEr  |↑  |0.0081|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_METEOR |↑  |0.0730|±  |   N/A|
# | - refcoco_seg_val   |0.0    |none  |     0|refcoco_ROUGE_L|↑  |0.0749|±  |   N/A|

# [rank0]:[W128 00:46:10.057525479 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
# evaluate.sh: line 15: M/logs: No such file or directory
# (blip3o) omkar@gigabyte-G492-HA0-00:~/ritesh/EvoLMM/Evaluation/lmms-eval$ clear







# 026-01-28 07:34:04 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_aggregated:188 - Saving results aggregated
# qwen2_5_vl_our (pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/home/omkar/ritesh/EvoLMM/checkpoints/solver,max_pixels=12845056,interleave_visuals=False), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
# |   Tasks   |Version|     Filter     |n-shot|        Metric         |   |Value |   |Stderr|
# |-----------|-------|----------------|-----:|-----------------------|---|-----:|---|------|
# |ifeval     |      2|none            |     0|inst_level_loose_acc   |↑  |0.7590|±  |   N/A|
# |ifeval     |      2|none            |     0|inst_level_strict_acc  |↑  |0.7314|±  |   N/A|
# |ifeval     |      2|none            |     0|prompt_level_loose_acc |↑  |0.6747|±  |0.0202|
# |ifeval     |      2|none            |     0|prompt_level_strict_acc|↑  |0.6488|±  |0.0205|
# |realworldqa|Yaml   |flexible-extract|     0|exact_match            |↑  |0.6837|±  |0.0168|

# [rank0]:[W128 07:34:06.878443129 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
# evaluate.sh: line 16: lse: command not found
# (blip3o) omkar@gigabyte-G492-HA0-00:~/ritesh/EvoLMM/Evaluation/lmms-eval$ 


# 026-01-28 07:34:04 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_aggregated:188 - Saving results aggregated
# qwen2_5_vl_our (pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/home/omkar/ritesh/EvoLMM/checkpoints/solver,max_pixels=12845056,interleave_visuals=False), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
# |   Tasks   |Version|     Filter     |n-shot|        Metric         |   |Value |   |Stderr|
# |-----------|-------|----------------|-----:|-----------------------|---|-----:|---|------|
# |ifeval     |      2|none            |     0|inst_level_loose_acc   |↑  |0.7590|±  |   N/A|
# |ifeval     |      2|none            |     0|inst_level_strict_acc  |↑  |0.7314|±  |   N/A|
# |ifeval     |      2|none            |     0|prompt_level_loose_acc |↑  |0.6747|±  |0.0202|
# |ifeval     |      2|none            |     0|prompt_level_strict_acc|↑  |0.6488|±  |0.0205|
# |realworldqa|Yaml   |flexible-extract|     0|exact_match            |↑  |0.6837|±  |0.0168|

# [rank0]:[W128 07:34:06.878443129 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
# evaluate.sh: line 16: lse: command not found
# (blip3o) omkar@gigabyte-G492-HA0-00:~/ritesh/EvoLMM/Evaluation/lmms-eval$ 




# 2026-01-28 09:50:41 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_aggregated:188 - Saving results aggregated
# qwen2_5_vl_our (pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/home/omkar/ritesh/EvoLMM/checkpoints/solver,max_pixels=12845056,interleave_visuals=False), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
# |    Tasks     |Version|Filter|n-shot|     Metric      |   |Value |   |Stderr|
# |--------------|-------|------|-----:|-----------------|---|------|---|------|
# |docvqa        |    N/A|      |      |                 |   |      |   |      |
# | - docvqa_test|Yaml   |none  |     0|submission       |↑  |N/A   |±  |   N/A|
# | - docvqa_val |Yaml   |none  |     0|anls             |↑  |0.9439|±  |0.0029|
# |ocrbench      |Yaml   |none  |     0|ocrbench_accuracy|↑  |0.8440|±  |   N/A|




# 2026-01-28 10:10:20 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_aggregated:188 - Saving results aggregated
# qwen2_5_vl_our (pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/home/omkar/ritesh/EvoLMM/checkpoints/solver,max_pixels=12845056,interleave_visuals=False), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
# |    Tasks     |Version|Filter|n-shot|    Metric    |   |Value |   |Stderr|
# |--------------|-------|------|-----:|--------------|---|-----:|---|------|
# |pope_full     |    N/A|      |      |              |   |      |   |      |
# | - pope_adv   |Yaml   |none  |     0|pope_accuracy |↑  |0.8660|±  |   N/A|
# | - pope_adv   |Yaml   |none  |     0|pope_f1_score |↑  |0.8543|±  |   N/A|
# | - pope_adv   |Yaml   |none  |     0|pope_precision|↑  |0.9357|±  |   N/A|
# | - pope_adv   |Yaml   |none  |     0|pope_recall   |↑  |0.7860|±  |   N/A|
# | - pope_adv   |Yaml   |none  |     0|pope_yes_ratio|↑  |0.5000|±  |   N/A|
# | - pope_pop   |Yaml   |none  |     0|pope_accuracy |↑  |0.8770|±  |   N/A|
# | - pope_pop   |Yaml   |none  |     0|pope_f1_score |↑  |0.8647|±  |   N/A|
# | - pope_pop   |Yaml   |none  |     0|pope_precision|↑  |0.9609|±  |   N/A|
# | - pope_pop   |Yaml   |none  |     0|pope_recall   |↑  |0.7860|±  |   N/A|
# | - pope_pop   |Yaml   |none  |     0|pope_yes_ratio|↑  |0.5000|±  |   N/A|
# | - pope_random|Yaml   |none  |     0|pope_accuracy |↑  |0.8880|±  |   N/A|
# | - pope_random|Yaml   |none  |     0|pope_f1_score |↑  |0.8753|±  |   N/A|
# | - pope_random|Yaml   |none  |     0|pope_precision|↑  |0.9874|±  |   N/A|
# | - pope_random|Yaml   |none  |     0|pope_recall   |↑  |0.7860|±  |   N/A|
# | - pope_random|Yaml   |none  |     0|pope_yes_ratio|↑  |0.5000|±  |   N/A|

# [rank0]:[W128 10:10:23.530622661 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
# (blip3o) omkar@gigabyte-G492-HA0-00:~/ritesh/EvoLMM/Evaluation/lmms-eval$ 