import datetime
import os
import time

import torch
from enot.quantization import DefaultQuantizationDistiller
from enot.quantization import TrtFakeQuantizedModel

from evaluation.eval_wrapper import eval_lane
from utils.common import ModelWithReshape
from utils.common import get_logger
from utils.common import get_model
from utils.common import get_train_loader
from utils.common import get_work_dir
from utils.common import merge_config
from utils.dist_utils import dist_print
from utils.dist_utils import synchronize
from utils.factory import get_loss_dict
from utils.factory import get_metric_dict
from utils.factory import get_optimizer
from utils.factory import get_scheduler


class CudaLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.dataloader.reset()
        for data in self.dataloader:
            yield data["images"].cuda()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    if args.local_rank == 0:
        work_dir = get_work_dir(cfg)

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        if args.local_rank == 0:
            with open(".work_dir_tmp_file.txt", "w") as f:
                f.write(work_dir)
        else:
            while not os.path.exists(".work_dir_tmp_file.txt"):
                time.sleep(0.1)
            with open(".work_dir_tmp_file.txt") as f:
                work_dir = f.read().strip()

    synchronize()
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed

    if args.local_rank == 0:
        os.system("rm .work_dir_tmp_file.txt")

    dist_print(datetime.datetime.now().strftime("[%Y/%m/%d %H:%M:%S]") + " start training...")
    dist_print(cfg)
    assert cfg.backbone in ["18", "34", "50", "101", "152", "50next", "101next", "50wide", "101wide", "34fca"]

    train_loader = get_train_loader(cfg)
    resume_epoch = 0
    # resume now work as model ckpt
    if cfg.model_ckpt is not None:
        net = torch.load(cfg.model_ckpt, map_location="cpu")["model_ckpt"]
    else:
        net = get_model(cfg)

    # resume, finetune and distributed are not supported for quantization

    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print("finetune from ", cfg.finetune)
        state_all = torch.load(cfg.finetune, map_location="cpu")["model"]
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if "model" in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    epoch = 1

    max_res = 0
    res = None

    net.cuda()
    res = eval_lane(net, cfg, ep=epoch, logger=logger)

    net.without_reshape = True
    fake_quantized_model = TrtFakeQuantizedModel(net).cuda()

    distiller = DefaultQuantizationDistiller(
        quantized_model=fake_quantized_model,
        dataloader=CudaLoader(train_loader),
        device="cuda",
        logdir=work_dir,
        verbose=2,
        n_epochs=cfg.epoch,
    )

    distiller.n_epochs = cfg.epoch  # Increase the number of threshold fine-tuning epochs.
    distiller.scheduler.T_max *= cfg.epoch  # Fix learning rate schedule.

    distiller.distill()

    fake_quantized_model.enable_quantization_mode(True)
    fake_quantized_model.cpu()

    torch.onnx.export(
        f=f"{cfg.log_path}/quantized_model.onnx",
        model=fake_quantized_model,
        args=torch.ones(
            (1, 3, cfg.train_height, cfg.train_width),
            dtype=torch.float32,
        ),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,  # Minimal opset for quant/dequant nodes
    )

    fake_quantized_model.cuda()
    fake_quantized_model = ModelWithReshape(
        fake_quantized_model,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
    )
    res = eval_lane(fake_quantized_model, cfg, ep=epoch, logger=logger)

    logger.close()
