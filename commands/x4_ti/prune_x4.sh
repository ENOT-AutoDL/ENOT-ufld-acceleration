#!/usr/bin/env bash

python prune.py \
    configs/tusimple_res18.py \
    --log_path runs/ti/x4/prune \
    --latency_type server \
    --acceleration 1.12 \
    --n_search_steps 200 \
    --host localhost \
    --port 15003 \
    --model_ckpt checkpoints/x3_jetson/model_best.pth \
    --ti_compatible