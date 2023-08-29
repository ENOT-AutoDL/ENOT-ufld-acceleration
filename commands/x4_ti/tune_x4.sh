#!/usr/bin/env bash

python train.py \
    configs/tusimple_res18_tune.py \
    --log_path runs/ti/x4/tune \
    --model_ckpt runs/ti/x4/prune/model_best.pth \
    --teacher checkpoints/baseline/model_best.pth \
    --distill_loss 2.0 \
    --epoch 200
