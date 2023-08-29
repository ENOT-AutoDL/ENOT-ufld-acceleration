#!/usr/bin/env bash

python measure.py \
    --model_ckpt runs/jetson/x2/tune/model_best.pth \
    --host localhost \
    --port 15003
