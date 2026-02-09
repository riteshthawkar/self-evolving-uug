## Image Understanding

We use lmms-eval for all the image understanding benchmark evaluation, please install 

```
cd lmms-eval
pip install -e .
```
And then modify the `understanding_eval.sh` with `--model_args pretrained="your/model/path/"` and `--tasks  mme `


## Image Understanding

For Geneval, you need to do the image generation firstly, please modify HF_HOME and model path in `https://github.com/JiuhaiChen/BLIP3o/blob/main/eval/geneval/generation.sh`. 
You can also set number of GPU used in the generation `N_CHUNKS=8`.

After you generate all images, you need to use [Geneval](https://github.com/djghosh13/geneval) to do the final  evaluation. 
