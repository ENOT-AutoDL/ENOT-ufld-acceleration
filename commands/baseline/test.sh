#!/usr/bin/env bash

python test.py \
    configs/tusimple_res18.py \
    --model_ckpt runs/baseline/model_best.pth
