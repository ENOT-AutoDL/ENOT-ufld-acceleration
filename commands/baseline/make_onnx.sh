#!/usr/bin/env bash

python export.py \
    --checkpoint runs/baseline/model_best.pth \
    --output_onnx runs/baseline/model_best.onnx
