# Adapted from https://github.com/kekmodel/FixMatch-pytorch
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from utils.constants import N_CHANNELS, CENTRAL_RANGE, ACG_LEN

from utils.models import MLP
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import utils.h5_utils as h5

from utils.cerebellum import get_cerebellum_dataset
from utils.misc import AverageMeter, accuracy
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
best_acc = 0
SEED = None


def save_checkpoint(state, is_best, checkpoint, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def train(
    args,
    labeled_trainloader,
    unlabeled_trainloader,
    test_loader,
    model,
    optimizer,
    ema_model,
    scheduler,
):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (
                    inputs_u_w,
                    inputs_u_s,
                ) = (
                    unlabeled_iter.next()
                )  #! weak and strong augment are in the unlabelled iter
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                inputs_u_w, inputs_u_s = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
            ).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(
                args.threshold
            ).float()  # ge is greater or equal elementwise. To compare with set threshold

            Lu = (
                F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask
            ).mean()

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg,
                    )
                )
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, f1 = test(args, test_loader, test_model, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = (
                    ema_model.ema.module
                    if hasattr(ema_model.ema, "module")
                    else ema_model.ema
                )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                    "ema_state_dict": ema_to_save.state_dict()
                    if args.use_ema
                    else None,
                    "acc": test_acc,
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best,
                args.out,
            )

            test_accs.append(test_acc)
            logger.info("Best top-1 acc: {:.2f}".format(best_acc))
            logger.info("Mean top-1 acc: {:.2f}".format(np.mean(test_accs[-20:])))
            logger.info("F1 score: {:.2f} \n".format(f1))

    return test_acc, f1


def test(args, test_loader, model, epoch):
    global SEED
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    true = []
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))

            predicted.extend(outputs.argmax(1).cpu().numpy().squeeze().ravel().tolist())
            true.extend(targets.cpu().numpy().ravel().tolist())

            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec2.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top2: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    )
                )
        if not args.no_progress:
            test_loader.close()

    f1 = f1_score(true, predicted, average="macro")
    if epoch == (args.epochs - 1):
        h5.save(f"fixm_targets_{SEED}.pkl", true)
        h5.save(f"fixm_outputs_{SEED}.pkl", predicted)

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-2 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg, f1


def main():
    global SEED

    class args:
        gpu_id = 0
        num_workers = 4
        dataset = "full"
        num_labeled = 24
        expand_labels = True
        arch = "cerebellum_full"
        total_steps = 2**11
        eval_step = 2**6
        start_epoch = 0
        batch_size = 40
        lr = 0.004265516314362439
        warmup = 0
        wdecay = 5e-4
        nesterov = True
        use_ema = True
        mu = 7  #! FixMatch hyperparam: coefficient of unlabeled batch size: int
        lambda_u = 1  #! FixMatch hyperparam: coefficient of unlabeled loss: float
        T = 1  #! FixMatch hyperparam: pseudo label temperature
        threshold = 0.95  #! FixMatch hyperparam: threshold for pseudo label generation
        out = "./results"
        resume = ""  # path to tar file if we want to resume training
        amp = False  # use 16-bit (mixed) precision through NVIDIA apex AMP
        opt_level = "01"  # apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html
        seed = None
        local_rank = -1  # For distributed training: local_rank
        no_progress = False
        ema_decay = 0.999

    global best_acc

    #! Dataset behaviour
    if args.dataset == "full":
        args.input_size = N_CHANNELS * CENTRAL_RANGE + ACG_LEN
        args.wvf_only = False
        args.acg_only = False
        args.num_classes = 6

    elif args.dataset == "acg":
        args.input_size = ACG_LEN
        args.wvf_only = False
        args.acg_only = True
        args.num_classes = 6

    elif args.dataset == "wvf":
        args.input_size = N_CHANNELS * CENTRAL_RANGE
        args.wvf_only = True
        args.acg_only = False
        args.num_classes = 6

    #! Change model architecture here!
    def create_model(args):
        if args.arch == "cerebellum_full":
            model = MLP(args.input_size, args.num_classes)
        else:
            print("Passed model architecture not supported yet")
            raise NotImplementedError

        logger.info(
            "Total params: {:.2f}M".format(
                sum(p.numel() for p in model.parameters()) / 1e6
            )
        )
        return model

    #! CUDA
    if args.local_rank == -1:
        device = (
            torch.device("cuda", args.gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",
    )

    # Seeding
    if args.seed is not None:
        SEED = h5.set_seed(args.seed)
    else:
        SEED = h5.set_seed()

    # Still cuda settings
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Load data
    labeled_dataset, unlabeled_dataset, test_dataset = get_cerebellum_dataset(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.wdecay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.RMSprop(
        grouped_parameters,
        lr=args.lr,
    )

    #! Mind how FixMatch epochs are calculated
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps
    )

    if args.use_ema:
        from utils.ema import ModelEMA

        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    if args.amp:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    test_acc, f1 = train(
        args,
        labeled_trainloader,
        unlabeled_trainloader,
        test_loader,
        model,
        optimizer,
        ema_model,
        scheduler,
    )
    return test_acc, f1


if __name__ == "__main__":
    accuracies = (
        h5.load("fixmatch_accuracies.pkl")
        if os.path.exists("fixmatch_accuracies.pkl")
        else []
    )
    f1_scores = h5.load("fixmatch_f1.pkl") if os.path.exists("fixmatch_f1.pkl") else []
    for _ in range(20):
        acc, f1 = main()
        accuracies.append(acc)
        f1_scores.append(f1)
        h5.save(f"fixmatch_accuracies.pkl", accuracies)
        h5.save(f"fixmatch_f1.pkl", f1_scores)
