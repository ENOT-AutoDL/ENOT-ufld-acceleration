#!/usr/bin/env bash

python measure.py \
    --model_ckpt runs/baseline/model_best.pth \
    --host localhost \
    --port 15003
