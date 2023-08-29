#!/usr/bin/env bash

python measure.py \
    --model_ckpt runs/ti/x4/tune/model_best.pth \
    --host localhost \
    --port 15003 \
    --ti_server
