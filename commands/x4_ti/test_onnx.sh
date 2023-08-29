#!/usr/bin/env bash

python test.py \
    configs/tusimple_res18.py \
    --onnx_path runs/ti/x4/tune/model_best.onnx \
    --batch_size 1