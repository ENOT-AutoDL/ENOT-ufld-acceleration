#!/usr/bin/env bash

python export.py \
    --checkpoint runs/jetson/x2/tune/model_best.pth \
    --output_onnx runs/jetson/x2/tune/model_best.onnx
