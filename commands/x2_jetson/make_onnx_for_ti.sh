#!/usr/bin/env bash

python export.py \
    --checkpoint runs/jetson/x2/tune/model_best.pth \
    --output_onnx runs/ti/x3/tune/model_best.onnx \
    --ti_compatible
