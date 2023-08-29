#!/usr/bin/env bash

python export.py \
    --checkpoint runs/ti/x4/tune/model_best.pth \
    --output_onnx runs/ti/x4/tune/model_best.onnx \
    --ti_compatible
