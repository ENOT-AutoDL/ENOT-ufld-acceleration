#!/usr/bin/env bash

python measure.py \
    --onnx runs/jetson/x3/quant/quantized_model.onnx \
    --host localhost \
    --port 15003
