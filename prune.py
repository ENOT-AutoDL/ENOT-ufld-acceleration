import datetime
import os
import time
from copy import deepcopy
from functools import partial
from typing import Dict

import onnx
import torch
from enot.pruning.label_selector import OptimalPruningLabelSelector
from enot.pruning.prune import prune_model
from enot.pruning.prune_calibrator import PruningCalibrator
from enot_latency_server.client import measure_latency_remote
from fvcore.nn import FlopCountAnalysis

import utils.dist_utils
from evaluation.eval_wrapper import eval_lane
from export import TiCompatibleClsLinear
from utils.common import calc_loss
from utils.common import get_logger
from utils.common import get_model
from utils.common import get_train_loader
from utils.common import get_work_dir
from utils.common import inference
from utils.common import merge_config
from utils.common import save_model
from utils.dist_utils import dist_print
from utils.dist_utils import dist_tqdm
from utils.dist_utils import synchronize
from utils.factory import get_loss_dict
from utils.factory import get_metric_dict
from utils.factory import get_optimizer
from utils.factory import get_scheduler


def calibrate(
    net: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_dict: Dict,
    logger: utils.dist_utils.DistSummaryWriter,
    epoch: int,
    dataset: torch.utils.data.Dataset,
):
    net.eval()
    pruning_calibrator = PruningCalibrator(model=net)
    progress_bar = dist_tqdm(train_loader)
    with pruning_calibrator:
        for b_idx, data_label in enumerate(progress_bar):
            global_step = epoch * len(data_loader) + b_idx
            results = inference(net, data_label, dataset)

            loss = calc_loss(
                loss_dict=loss_dict,
                results=results,
                logger=logger,
                global_step=global_step,
                epoch=epoch,
            )
            loss.backward()

    return pruning_calibrator.pruning_info


def tune_bn(net, data_loader, dataset):
    net.train()
    progress_bar = dist_tqdm(data_loader)
    for b_idx, data_label in enumerate(progress_bar):
        _ = inference(net, data_label, dataset)

    return net


def measure_latency_on_server(model, device, image_size, port, host, ti_server=False):
    model = deepcopy(model)
    model.eval()
    if ti_server:
        opset = 9
        model.cls[3] = TiCompatibleClsLinear(linear=model.cls[3]).to(device)
    else:
        opset = 11

    torch.onnx.export(
        model=model,
        args=torch.ones((1, 3, *image_size), device=device),
        f="model.onnx",
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
    )

    onnx_model = onnx.load("model.onnx")
    result = measure_latency_remote(onnx_model.SerializeToString(), host=host, port=port)
    if isinstance(result, float):
        return result
    print(result)
    return result["latency"]


def measure_flops(model):
    model.eval()
    flops = FlopCountAnalysis(model, torch.ones((1, 3, cfg.train_height, cfg.train_width)))
    flops = flops.total()
    return flops


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
        ValueError("--model_ckpt should be passed to pruning script.")

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
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

    if cfg.latency_type == "MAC":
        latency_measurement_func = measure_flops
    elif cfg.latency_type == "server":
        latency_measurement_func = partial(
            measure_latency_on_server,
            device="cpu",
            image_size=(cfg.train_height, cfg.train_width),
            host=cfg.host,
            port=cfg.port,
            ti_server=cfg.ti_compatible,
        )
    else:
        raise ValueError(f"latency_type {cfg.latency_type} is not supported.")

    net.cpu()
    baseline_latency = latency_measurement_func(net)
    dist_print("baseline latency:", baseline_latency)
    net.cuda()

    pruning_info = calibrate(
        net=net,
        data_loader=train_loader,
        loss_dict=loss_dict,
        logger=logger,
        epoch=epoch,
        dataset=cfg.dataset,
    )

    net.cpu()
    label_selector = OptimalPruningLabelSelector(
        model=net,
        latency_calculation_function=latency_measurement_func,
        target_latency=baseline_latency / cfg.acceleration,
        n_search_steps=cfg.n_search_steps,
        architecture_optimization_strategy=lambda x: (8, 1),
    )
    labels = label_selector.select(pruning_info)

    pruned_model = prune_model(model=net, pruning_info=pruning_info, prune_labels=labels)

    train_loader.reset()
    net.cuda()
    tune_bn(net=pruned_model, data_loader=train_loader, dataset=cfg.dataset)

    res = eval_lane(pruned_model, cfg, ep=epoch, logger=logger)

    pruned_model.cpu()
    pruned_model_latency = latency_measurement_func(pruned_model)
    dist_print("pruned model latency:", pruned_model_latency)
    dist_print("acceleration:", baseline_latency / pruned_model_latency)

    save_model(pruned_model, optimizer, epoch, work_dir, distributed)
    logger.add_scalar("CuEval/X", max_res, global_step=epoch)

    logger.close()
