import datetime
import os
import time
from typing import Callable
from typing import Dict

import torch
from enot.optimize import GTBaselineOptimizer

from evaluation.eval_wrapper import eval_lane
from utils.common import ExponentialMovingAverage
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
from utils.metrics import reset_metrics
from utils.metrics import update_metrics


def train(
    net: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_dict: Dict[str, Callable],
    optimizer,
    scheduler,
    logger,
    epoch: int,
    metric_dict: Dict[str, Callable],
    dataset: str,
    teacher: torch.nn.Module = None,
    distill_loss_weight: float = None,
    distill_loss_fn: Callable = None,
    model_ema: ExponentialMovingAverage = None,
):
    net.train()
    progress_bar = dist_tqdm(data_loader)
    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(data_loader) + b_idx

        results = None
        common_loss = None
        task_loss = None
        distill_loss = None

        optimizer.zero_grad()

        def closure():
            nonlocal results
            nonlocal common_loss
            nonlocal task_loss
            nonlocal distill_loss
            results = inference(net, data_label, dataset, teacher=teacher)
            task_loss = calc_loss(loss_dict, results, logger, global_step, epoch)
            if teacher:
                distill_loss = distill_loss_fn(results["student_out"], results["teacher_out"]) * distill_loss_weight
                common_loss = task_loss + distill_loss
            else:
                common_loss = task_loss
            common_loss.backward()

            if model_ema and b_idx % cfg.model_ema_steps == 0:
                model_ema.update_parameters(net)
                if epoch < args.ema_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)

            return common_loss

        optimizer.step(closure)
        scheduler.step(global_step)

        if global_step % 20 == 0:
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)
            for me_name, me_op in zip(metric_dict["name"], metric_dict["op"]):
                logger.add_scalar("metric/" + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar("meta/lr", optimizer.param_groups[0]["lr"], global_step=global_step)
            logger.add_scalar("train/task_loss", task_loss, global_step=global_step)
            if teacher:
                logger.add_scalar("train/distill_loss", distill_loss, global_step=global_step)
            logger.add_scalar("train/common_loss", common_loss, global_step=global_step)

            if hasattr(progress_bar, "set_postfix"):
                kwargs = {
                    me_name: "%.3f" % me_op.get() for me_name, me_op in zip(metric_dict["name"], metric_dict["op"])
                }
                new_kwargs = {}
                for k, v in kwargs.items():
                    if "lane" in k:
                        continue
                    new_kwargs[k] = v
                progress_bar.set_postfix(loss="%.3f" % float(common_loss), **new_kwargs)


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
    net = get_model(cfg)
    if args.model_ckpt is not None:
        net = torch.load(args.model_ckpt, map_location="cpu")["model_ckpt"].cuda()
    optimizer = get_optimizer(net, cfg)
    # resume now work as model ckpt
    if cfg.resume is not None:
        dist_print("==> Resume model from " + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location="cpu")
        net.load_state_dict(resume_dict["model"])
        net.cuda()
        if "optimizer" in resume_dict.keys():
            optimizer.load_state_dict(resume_dict["optimizer"])

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    optimizer = GTBaselineOptimizer(model=net, optimizer=optimizer, rho=0.05)
    model_ema = None
    if cfg.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = 1 * cfg.batch_size * cfg.model_ema_steps / cfg.epoch
        alpha = 1.0 - cfg.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        print(1.0 - alpha)
        model_ema = ExponentialMovingAverage(
            net,
            decay=1.0 - alpha,
            device="cuda",
        )

    if cfg.finetune is not None:
        dist_print("finetune from ", cfg.finetune)
        state_all = torch.load(cfg.finetune, map_location="cpu")["model"]
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if "model" in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    if cfg.teacher:
        teacher = torch.load(cfg.teacher, map_location="cpu")["model_ckpt"].cuda()
    else:
        teacher = None

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    # cp_projects(cfg.auto_backup, work_dir)
    max_res = 0
    res = None
    for epoch in range(resume_epoch, cfg.epoch):
        train(
            net=net,
            data_loader=train_loader,
            loss_dict=loss_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            epoch=epoch,
            metric_dict=metric_dict,
            dataset=cfg.dataset,
            teacher=teacher,
            distill_loss_weight=cfg.distill_loss_weight,
            distill_loss_fn=torch.nn.MSELoss() if teacher else None,
            model_ema=model_ema,
        )
        train_loader.reset()

        if cfg.model_ema:
            res = eval_lane(model_ema, cfg, ep=epoch, logger=logger)
        else:
            res = eval_lane(net, cfg, ep=epoch, logger=logger)

        if res is not None and res > max_res:
            max_res = res
            if cfg.model_ema:
                save_model(
                    net=model_ema,
                    optimizer=optimizer,
                    epoch=epoch,
                    save_path=work_dir,
                    distributed=distributed,
                )
            else:
                save_model(
                    net=net,
                    optimizer=optimizer,
                    epoch=epoch,
                    save_path=work_dir,
                    distributed=distributed,
                )
        logger.add_scalar("CuEval/X", max_res, global_step=epoch)

    logger.close()
