#!/usr/bin/env bash

python test.py \
    configs/tusimple_res18.py \
    --model_ckpt runs/jetson/x3/tune/model_best.pth
