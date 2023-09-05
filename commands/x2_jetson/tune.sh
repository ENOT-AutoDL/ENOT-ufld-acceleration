#!/usr/bin/env bash

python train.py \
    configs/tusimple_res18_tune.py \
    --log_path runs/jetson/x2/tune \
    --model_ckpt runs/jetson/x2/prune/model_best.pth \
    --teacher runs/baseline/model_best.pth \
    --distill_loss 2.0 \
    --epoch 200
