import os

import torch

from evaluation.eval_wrapper import eval_on_pickles
from utils.common import merge_config

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    cfg.distributed = False

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    print(f"Start validation using pickles from {args.ti_inference_results}...")

    eval_on_pickles(args.ti_inference_results, cfg)
