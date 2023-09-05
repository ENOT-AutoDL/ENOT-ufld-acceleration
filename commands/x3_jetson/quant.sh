#!/usr/bin/env bash

python quant.py \
    configs/tusimple_res18.py \
    --log_path runs/jetson/x3/quant \
    --model_ckpt checkpoints/x3_jetson/model_best.pth \
    --epoch 0