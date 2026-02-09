#!/bin/bash


conda  activate  blip3o


export HF_HOME=/your/HF/home/path


MODEL="/your/mode/path/"



# Total number of GPUs/chunks.
N_CHUNKS=8

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python generate.py --model "$MODEL" --index $i --n_chunks $N_CHUNKS &
done

# Wait for all background processes to finish.
wait
echo "All background processes finished."


