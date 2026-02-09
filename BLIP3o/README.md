# 🌌 BLIP3-o

BLIP3-o is a unified multimodal model that combines the reasoning and instruction following strength of autoregressive models with the generative power of diffusion models. Unlike prior works that diffuse VAE features or raw pixels, BLIP3-o diffuses semantically rich **CLIP image features**, enabling a powerful and efficient architecture for both image understanding and generation.

## 📖 [Arxiv](http://arxiv.org/abs/2505.09568)


## ✨ Highlights


- **Fully Open-Source:**
  - **Pretraining Data:** [27 Million Detailed Captions](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption), [5 Million Short Captions](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption), [4 Million JourneyDB](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB)
  - **Instruction Tuning Data:** [60 k GPT-4o Distilled Instruction Tuning Data](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)
  - **Model Weights:** [4 B](https://huggingface.co/BLIP3o/BLIP3o-Model-4B)  [8 B](https://huggingface.co/BLIP3o/BLIP3o-Model-8B)
  - **Training Code**


- **Unified Architecture:** for both image understanding and generation.
- **CLIP Feature Diffusion:** Directly diffuses semantic vision features for stronger alignment and performance.
- **State-of-the-art performance:** across a wide range of image understanding and generation benchmarks.


## Update

- [2025/05/29] 🔥 Check out new branch [Qwen3-Siglip2](https://github.com/JiuhaiChen/BLIP3o/tree/Qwen3-Siglip2), it supports SigLIP2 as the image understanding vision encoder and Qwen3 as the AR backbone, with flexible training strategies—either sequential training or joint training—for both image understanding and generation.


- [2025/05/22] 🔥 Evaluation for image understanding and generation, please check folder [eval](https://github.com/JiuhaiChen/BLIP3o/tree/main/eval).

- [2025/05/20] 🔥 Welcome to discuss with us if you have any questions.
Discord: https://discord.gg/SsVYdV84bw
or Wechat
<p align="center">
<img src="figure/wechat_2.jpg" width="256">
</p>

- [2025/05/19] 🔥 We understand this is a large codebase, we shared a high-level overview of its [Code Structure](https://github.com/JiuhaiChen/BLIP3o/issues/11#issuecomment-2891930000), feel free to open an issue if you encounter any problems.


- [2025/05/16] 🔥 We’ve published a dataset of 24 million images with detailed captions [BLIP3o Pretrain Long Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption) and 5 million images with short caption [BLIP3o Pretrain Short Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption). All images and their captions are compressed into tar archives, **no separate image url downloads or manual unzipping required**. 




- [2025/05/16] 🔥 We’ve reorganized and cleaned up the repository to ensure a clear, well-structured codebase. Please give the training and inference scripts a try, and feel free to leave an issue if you run into any problems. We apologize for any confusion caused by our original codebase release.






<!-- <p align="center">
  <img src="figure/arch.png" alt="BLIP3-U Overview Figure" width="700"/>
</p>

*Figure: Overview of the BLIP3-U architecture. We use Flow Matching Loss to predict the ground truth CLIP embeddings. At inference, the autoregressive model first generates a sequence of visual tokens from the given conditioning, and those visual tokens are then passed to a diffusion transformer that decodes them into the final image.* -->


---

## Demo



You can try out BLIP3-o in your browser using our interactive [Demo](https://blip3o.salesforceresearch.ai/). 



Install package for tranining
```Shell
conda create -n blip3o python=3.11 -y
conda activate blip3o
pip install --upgrade pip  setuptools
pip install -r requirements.txt
```

## Model Checkpoint

### BLIP3o-4B [4B](https://huggingface.co/BLIP3o/BLIP3o-Model-4B)

### BLIP3o-8B [8B](https://huggingface.co/BLIP3o/BLIP3o-Model-8B)

## Inference

You can  download our checkpoint:

```Shell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BLIP3o/BLIP3o-Model-8B', repo_type='model'))"
```

and run the inference code

```Shell
python inference.py  /HF_model/checkpoint/path/
```
## Training
We include two scripts: **slurm.sh** for multi-node training on Slurm clusters, and **run.sh** for debugging.

For both **slurm.sh** and **run.sh**, you need to import huggingface home **HF_HOME**, training data folder **IMG_FOLDER** and output model save folder **OUTPUT_FOLDER**. 

For our open source model training, we combine the pretraining dataset, including [long caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption), [short caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption) and [JourneyDB](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB). 

You can download all datasets and then modify the [dataloader](https://github.com/JiuhaiChen/BLIP3o/blob/main/blip3o/train/train.py#L587) in the code to load whichever subset you need.

```
data_files = glob.glob(os.path.join('/long/caption/path', "*.tar")) + glob.glob(os.path.join('/short/caption/path', "*.tar")) + glob.glob(os.path.join('/JourneyDB/caption/path', "*.tar")) 
train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128)
```

When training the diffusion transformer from scratch, we recommend using a large number of training steps (at least **150k steps**) along with a cosine annealing learning rate schedule that decays from 1×10⁻⁴ down to 1×10⁻⁵.

When finetuning with our BLIP3o-60k, we recommend using a large number of training steps (at least **10k steps**) along with a cosine annealing learning rate schedule that decays from 1×10⁻⁴ down to 0.


## CLIP + Diffusion (Encoder + Decoder)
We also provide two CLIP + Diffusion: 

**[EVA-CLIP + SDXL]**: The model checkpoint already includes the diffusion decoder [diffusion-decoder](https://huggingface.co/BLIP3o/BLIP3o-Model/tree/main/diffusion-decoder). The EVA-CLIP vision tower weights can be downloaded here [EVA-CLIP](https://huggingface.co/jiuhai/eva_clip_vision_tower), the preprocess of EVA-CLIP is in the training code [EVA-CLIP-preprocess](https://github.com/JiuhaiChen/BLIP3o/tree/main/blip3o/model/multimodal_encoder/eva_clip).

**[SigLIP2 + SANA]**: The model checkpoint is available here [SigLIP2_SANA ](https://huggingface.co/BLIP3o/SigLIP2_SANA). 

First, download the model checkpoint:

```Shell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BLIP3o/SigLIP2_SANA', repo_type='model'))"
```
and update [img_path = 'fig.jpg'](https://github.com/JiuhaiChen/BLIP3o/blob/2e6775baaa6703fd8be835b658fb20c31e3ca365/siglip2_sana/inference.py#L53) to any local image you like. Run the inference code

```Shell
cd siglip2_sana
python inference.py  /HF_model/SigLIP2_SANA/path/
```

And you will get reconstruction.png.

## Supported Tasks

- **Text → Text**  
- **Image → Text** (Image Understanding) 
- **Text → Image** (Image Generation)  
- **Image → Image** (Image Editing)  
- **Multitask Training** (Image generation and undetstanding mix training)


## Supported Image Generation Methods

- **CLIP + MSE**  
- **CLIP + Flow Matching** 
- **VAE + Flow Matching**  



## Supported Autoregressive Backbones

- **Qwen-2.5-VL**  
- **LLaMA 3**
  
We suggest to use Qwen-2.5-VL as the backbone, we are fixing some tokenizer issues for LLama3.

## Supported Dataset Format

- **Webdataset**  
- **Json**


## Data Loading

Most of our training data use Huggingface datasets to load **WebDataset**. To download the datasets:

### T2I Pretraining Dataset  
#### 👉 [25 Million Detailed Captions](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption), [5 Million Short Captions](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption), [4 Million JourneyDB](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB)

You can download the datasets by
```Shell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BLIP3o/BLIP3o-Pretrain', repo_type='dataset'))"
```
And load them directly with HuggingFace WebDataset
```Shell
train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128)
```
### Super High Quality T2I Instruction Tuning Data Prompted From GPT-4o   
#### 👉 [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)
You can download the datasets by
```Shell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BLIP3o/BLIP3o-60k', repo_type='dataset'))"
```
### 💥 In general, BLIP3o-60k can help pretrained T2I models achieve a 5–7 point absolute score improvement on the GenEval and DPG benchmarks.

## Qualitative results of BLIP3-o
![BLIP3-o Overview Figure](figure/image.png)
*Figure: Qualitative results of BLIP3-o.*


## Image Benchmark Performance

| Model               | Pretrain Data                                             | GenEval | DBP    | WISE |
|---------------------|-----------------------------------------------------------|---------|--------|------|
| 4B (open source)    | 30 million open-source data                           | 0.81    | 79.36  | 0.50 |
| 8B (open source)    | 30 million open-source data                           | 0.83    | 80.73  | 0.52 |
| 8B (paper reported) | 30 million open-source + 30 million proprietary data  | 0.84    | 81.60  | 0.62 |






### Citation
To cite the paper and model
```
@article{chen2025blip3,
  title={BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset},
  author={Chen, Jiuhai and Xu, Zhiyang and Pan, Xichen and Hu, Yushi and Qin, Can and Goldstein, Tom and Huang, Lifu and Zhou, Tianyi and Xie, Saining and Savarese, Silvio and others},
  journal={arXiv preprint arXiv:2505.09568},
  year={2025}
}
```


