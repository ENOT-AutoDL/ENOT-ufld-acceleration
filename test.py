import os

import onnxruntime as ort
import torch

from evaluation.eval_wrapper import eval_lane
from utils.common import CallableSession
from utils.common import get_model
from utils.common import merge_config

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1
    cfg.distributed = distributed
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if cfg.onnx_path:
        providers = ["CUDAExecutionProvider"]
        provider_options = [{}]

        ort_session_options = ort.SessionOptions()
        session = ort.InferenceSession(
            cfg.onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=ort_session_options,
        )
        net = CallableSession(session)
    else:
        net = get_model(cfg)

        if cfg.model_ckpt:
            net = torch.load(cfg.model_ckpt, map_location="cpu")["model_ckpt"].cuda()
        else:
            state_dict = torch.load(cfg.test_model, map_location="cpu")["model"].cuda()
            compatible_state_dict = {}
            for k, v in state_dict.items():
                if "module." in k:
                    compatible_state_dict[k[7:]] = v
                else:
                    compatible_state_dict[k] = v

            net.load_state_dict(compatible_state_dict, strict=True)

        if distributed:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    eval_lane(net, cfg)
