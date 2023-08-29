#!/usr/bin/env bash

python test.py \
    configs/tusimple_res18.py \
    --onnx_path runs/jetson/x3/tune/model_best.onnx \
    --batch_size 1