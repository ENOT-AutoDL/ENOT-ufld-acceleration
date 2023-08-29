import os
import shutil

from evaluation.eval_wrapper import create_data_pickles
from utils.common import merge_config

if __name__ == "__main__":
    args, cfg = merge_config()
    cfg.distributed = False

    print(f"Start creating directory {args.pickle_data_path}")

    if os.path.exists(args.pickle_data_path) and os.path.isdir(args.pickle_data_path):
        shutil.rmtree(args.pickle_data_path)

    os.mkdir(args.pickle_data_path)

    create_data_pickles(data_root=args.pickle_data_path, cfg=cfg)
