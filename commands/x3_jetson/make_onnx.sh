#!/usr/bin/env bash

python export.py \
    --checkpoint runs/jetson/x3/tune/model_best.pth \
    --output_onnx runs/jetson/x3/tune/model_best.onnx
