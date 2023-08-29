#!/usr/bin/env bash

python prune.py \
    configs/tusimple_res18.py \
    --log_path runs/jetson/x2/prune \
    --latency_type server \
    --acceleration 2.0 \
    --n_search_steps 200 \
    --host localhost \
    --port 15003 \
    --model_ckpt runs/baseline/model_best.pth
