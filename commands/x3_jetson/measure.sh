#!/usr/bin/env bash

python measure.py \
    --model_ckpt runs/jetson/x3/tune/model_best.pth \
    --host localhost \
    --port 15003
