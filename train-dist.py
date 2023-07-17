import os
import argparse
import torch
import torch.distributed as dist

from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel

from gnetmdk.layers import GNetDet, GNetDetLoss, GNetDetWithRegLoss
from gnetmdk.dist import comm, launch, check_env
from gnetmdk.dataset import get_gnetdet_dataloader
from gnetmdk.checkpoint import GNetDetCheckPointer
from gnetmdk.evaluation import Evaluator, DistEvaluator, Tensorboard
from gnetmdk.utils.experiment import progress_bar, ProgressBar
from gnetmdk.utils.fuse import fuse_bn, contain_bn
from gnetmdk.solver import get_lr_scheduler
from configs import get_config

parser = argparse.ArgumentParser(
    epilog=f"""
Example:

Run on a single machine:
    $ python3 %(prog)s --gpus 1
    $ python3 %(prog)s --gpus 2

Run on multiple machines:
    (machine0) $ python3 %(prog)s --gpus 1 --world-size 2 --rank 0 --dist-url <MASTER_IP:MASTER_PORT>
    (machine1) $ python3 %(prog)s --gpus 1 --world-size 2 --rank 1 --dist-url <MASTER_IP:MASTER_PORT>
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# Args for distributed training
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per each node.")
parser.add_argument("--world-size", type=int, default=1, help="Number of nodes for distributed training.")
parser.add_argument("--rank", type=int, default=0, help="Node rank for distributed training.")
parser.add_argument("--dist-url", type=str, metavar="<IP>:<PORT>", default="tcp://127.0.0.1:8888",
                    help="IP address and port along with protocol to connect with master machine. "
                         "(Default tcp://127.0.0.1:8888 [NOTE: Make sure port 8888 is free to use!]) "
                         "If --dist-url==env://, then following environment variables must be provided: "
                         "[1] WORLD_SIZE: Total number of machines (nodes); "
                         "[2] RANK: Machine rank (index), start from 0; "
                         "[3] MASTER_ADDR: IP address of master machine ranked at 0; "
                         "[4] MASTER_PORT: Port number of master machine ranked at 0.")

# Args to override default cfg settings
parser.add_argument("-s", "--step", type=int, help="Training step. If not specified, use default step in configs.py")
parser.add_argument("--cal", action="store_true", default=False, help="If present, train one epoch for `Calibration`.")
parser.add_argument("--fuse-bn", action="store_true", default=False, help="Fuse BN before step2 training. Don't use.")
parser.add_argument("--finetune", type=int, default=0, help="Freeze GNetDet backbone.")
parser.add_argument("opts", nargs=argparse.REMAINDER, help="Modify config options through command line.")


# State variables
BEST_LOSS: float
LEARNING_RATE: float


def main_worker(args):
    """
    Main function.
    """
    ##########
    # Configs
    ##########
    cfg = get_config(args.step, args.cal)
    cfg.gpus = args.gpus
    assert args.finetune >= 0, "Non-negative epochs for --finetune!"
    cfg.finetune = args.finetune
    cfg.merge_opts_with_known_conf(args)
    cfg.rescale_to_distributed()
    cfg.freeze()

    # TODO: BN layer unstable
    if cfg.batch_norm:
        print(f"[WARNING] GNetDet with BN is unstable, which performs `BAD` during quantization!!")
        print(f"[WARNING] GNetDet with BN is unstable, which performs `BAD` during quantization!!")

    #############
    # DataLoader
    #############
    train_sampler, train_loader = get_gnetdet_dataloader(cfg, train=True)
    valid_sampler, valid_loader = get_gnetdet_dataloader(cfg, train=False)

    ###########
    # GNetDet
    ###########
    print_by_rank(f"Instantiating GNetDet...", rank=cfg.local_rank)
    model = GNetDet(cfg).to(cfg.device)
    if cfg.finetune:
        print(f"Finetune is ON for first {cfg.finetune} epochs.")
        model.freeze_backbone()
        finetune_flag = True
    else:
        finetune_flag = False
    print(f"Instantiation complete. GNetDet created and moved to device: {cfg.device}")

    ############
    # BN Fusion
    ############
    if args.fuse_bn:
        print(f"[WARNING] Fuse BN is under testing, which most likely causes unstable results!!")
        print(f"[WARNING] Fuse BN is under testing, which most likely causes unstable results!!")
        if cfg.local_rank == 0:
            # BN must be fused into CONV before quantization
            assert "step1" in cfg.checkpoint_path, f"BN Fusion on STEP1 model only: {cfg.checkpoint_path}"
            print_by_rank("\nFuse BN ...", rank=cfg.local_rank)
            print_by_rank("Reading checkpoint from:", os.path.relpath(cfg.checkpoint_path), rank=cfg.local_rank)

            # Load step1 ckpt
            model.load_state_dict(torch.load(cfg.checkpoint_path))
            model = fuse_bn(model, cfg.unfreeze())
            print_by_rank("BN Fusion Complete\n", rank=cfg.local_rank)

            # Save Fused model ckpt
            save_path = os.path.join(os.path.dirname(cfg.best_save_path), "fused.pth")
            print_by_rank(f'Fused model saved:', os.path.relpath(save_path), rank=cfg.local_rank)
            state_dict = model.state_dict()
            torch.save(state_dict, save_path)
        comm.synchronize()
        exit(0)

    # BN is not allowed since step2 training
    if contain_bn(model) and cfg.step >= 2:
        print("[ERROR] Current model has BN layer, which is not allowed for step >= 2 training. "
              "Please fuse BN layers by specify --fuse-bn argument before step2 training.")
        exit(-1)

    ############
    # DDP Model
    ############
    if cfg.world_size > 1:
        print(f"[RANK: {cfg.global_rank}] DDP Model Created")
        model = DistributedDataParallel(model, device_ids=[cfg.device])

    if cfg.num_workers > 0:
        # If num_worker is greater than 0, than the program most likely throws an error:
        # "unable to open shared memory object </torch_5962_408180057> in read-write mode"
        # To walk around this problem, one solution is set `num_worker=0`;
        # or you can also run this command `ulimit -n 64000` in terminal.
        # NOTE: `64000` is kind of a random number, change this value if needed.
        pass

    ################################
    # Loss, Optimizer, Scheduler
    ################################
    criterion = GNetDetWithRegLoss(cfg, model)
    if isinstance(criterion, GNetDetWithRegLoss):
        cfg.unfreeze()
        cfg.weight_decay = 0.0
        cfg.freeze()

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        cfg.learning_rate, cfg.lr_momentum, weight_decay=cfg.weight_decay,
    )
    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    ###############
    # Checkpointer
    ###############
    print_by_rank("Reading checkpoint from:", os.path.relpath(cfg.checkpoint_path), rank=cfg.local_rank)
    checkpointer = GNetDetCheckPointer(cfg, model=model)
    checkpointer.load()

    ############
    # Evaluator
    ############
    if cfg.world_size > 1:
        evaluator = DistEvaluator()
    else:
        evaluator = Evaluator()

    ##############
    # Tensorboard
    ##############
    # Only logs info by master node's rank 0 process.
    if cfg.global_rank == 0 and not cfg.cal:
        tensorboard = Tensorboard(step=cfg.step, model=model)
        print("Tensorboard logger instantiated.")
        print("Logging directory:", tensorboard.logdir)
    else:
        tensorboard = None

    #########
    # Others
    #########
    # print some info
    print_by_rank(rank=cfg.local_rank)
    print_by_rank(f"batch-norm: {cfg.batch_norm}", rank=cfg.local_rank)
    print_by_rank(f"image-format: {cfg.image_format}", rank=cfg.local_rank)
    print_by_rank(f"loss-weights: {cfg.loss_weights}", rank=cfg.local_rank)
    print_by_rank(f"Total images: {len(train_loader) * cfg.batch_size * cfg.world_size}", rank=cfg.local_rank)
    if train_sampler is not None:
        print_by_rank(f"Per-GPU images: {len(train_sampler)}", rank=cfg.local_rank)

    # Set global state variables
    global BEST_LOSS, LEARNING_RATE
    BEST_LOSS = float("inf")

    ################
    # Training loop
    ################
    for epoch in range(0, cfg.num_epochs):
        # Freeze backbone if finetune is on
        if finetune_flag and epoch >= cfg.finetune:
            print_by_rank("Model unfreeze", rank=cfg.local_rank)
            model.unfreeze()
            finetune_flag = False

        # Share lr to global
        LEARNING_RATE = lr_scheduler.get_last_lr()[0]

        # Shuffle dataloader
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print_by_rank('\n\nEpoch %d / %d' % (epoch + 1, cfg.num_epochs), rank=cfg.local_rank)
        print_by_rank(f"Learning Rate: {LEARNING_RATE:.2e}", rank=comm.get_local_rank())

        # training loop
        train(model, train_loader, criterion, optimizer, epoch, cfg, tensorboard)

        # validation loop
        validate(model, valid_loader, criterion, evaluator, checkpointer, epoch, cfg, tensorboard)

        # update learning rate
        lr_scheduler.step()

        # For calibration step, we need to reduce cap values.
        if cfg.cal and cfg.world_size > 1:
            cap_values = torch.tensor(model.module.cap, dtype=torch.float32, device=cfg.device)
            # TODO: Reduce at every forward pass?
            dist.all_reduce(cap_values, dist.ReduceOp.MAX)
            model.module.cap = cap_values.cpu().tolist()
            model.module.write_cap_txt(cfg.cap_txt_path)


def train(model, train_loader, criterion, optimizer, epoch, cfg, tensorboard):
    """
    One epoch training step.
    """
    model.train()
    losses = defaultdict(float)

    monitor = ProgressBar(len(train_loader), cfg.log_freq, cfg.local_rank)
    for i, (images, target, *_) in enumerate(train_loader):
        images = images.to(cfg.device, non_blocking=True)
        target = target.to(cfg.device, non_blocking=True)

        # forward
        pred = model(images)
        data_losses, reg_losses = criterion(pred, target)
        data_loss = sum(data_losses.values())
        reg_loss = sum(reg_losses.values())
        total_loss = data_loss + reg_loss

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # log info
        # `train_loss` has not been synchronized yet, so they are different
        # across each process. They will be reduced after training.
        # train_loss += (data_loss.data.item() + reg_loss.data.item())
        losses["data_loss"] += data_loss.data.item()
        losses["reg_loss"] += reg_loss.data.item()
        info_str = monitor.format_loss(data_losses, loss=sum(losses.values()) / (i + 1))
        monitor.update(i, info_str)

    # Reduce Training loss.
    # train_loss /= len(train_loader)
    data_loss = losses["data_loss"] / len(train_loader)
    reg_loss = losses["reg_loss"] / len(train_loader)
    train_loss = data_loss + reg_loss
    ratio = data_loss / train_loss
    train_loss = comm.average(train_loss)

    # log tensorboard
    if tensorboard is not None:
        tensorboard.log_scalars(epoch, "Loss", {f"train_lr:{cfg.learning_rate:.2e}": train_loss})
        tensorboard.log_scalars(epoch, "Data+Reg", {"data_loss": data_loss, "reg_loss": reg_loss, "ratio": ratio})
        tensorboard.log_scalar(epoch, "LR", LEARNING_RATE)

    # Sync all processes
    comm.synchronize()


def validate(model, valid_loader, criterion, evaluator, checkpointer, epoch, cfg, tensorboard):
    """
    Validation step.
    """
    if cfg.cal: return

    global BEST_LOSS
    print_by_rank("Evaluation...", rank=cfg.local_rank)
    model.eval()
    validation_loss: float = 0.

    monitor = ProgressBar(len(valid_loader), cfg.log_freq, cfg.local_rank)
    with torch.no_grad(), evaluator:
        for i, (images, target, gt_bboxes, whs, fnames) in enumerate(valid_loader):
            images = images.to(cfg.device, non_blocking=True)
            target = target.to(cfg.device, non_blocking=True)

            # forward
            if i == 0: model.register_activation_hooks(tensorboard, epoch)
            pred = model(images)
            if i == 0: model.remove_activation_hooks()
            data_losses, reg_losses = criterion(pred, target)
            data_loss = sum(data_losses.values())
            reg_loss = sum(reg_losses.values())
            validation_loss += (data_loss.data.item() + reg_loss.data.item())

            # mAP evaluation
            evaluator.eval(pred, gt_bboxes, whs, fnames)

            # log info
            info_str = monitor.format_loss(data_losses, loss=validation_loss / (i + 1))
            monitor.update(i, info_str)

        # Reduce validation loss
        validation_loss /= len(valid_loader)
        validation_loss = comm.average(validation_loss, cfg)

        # Save weights
        # TODO: Only save ckpt if rank==0? Since DPP broadcast parameters.
        print_by_rank("validation loss:", validation_loss, rank=cfg.local_rank)
        if validation_loss < BEST_LOSS:
            BEST_LOSS = validation_loss
            print_by_rank("Get best test loss %.5f" % BEST_LOSS, rank=cfg.local_rank)
            print_by_rank(f'Saving best checkpoint to:', os.path.relpath(cfg.best_save_path), rank=cfg.local_rank)
            checkpointer.save(cfg.best_save_path)
        print_by_rank(f'Saving last checkpoint to:', os.path.relpath(cfg.last_save_path), rank=cfg.local_rank)
        checkpointer.save(cfg.last_save_path)

    # Log info to Tensorboard
    if tensorboard is not None:
        tensorboard.log_scalars(epoch, "Loss", {f"validation_lr:{cfg.learning_rate:.2e}": validation_loss})
        tensorboard.log_scalar(epoch, "mAP", evaluator.mAP)
        tensorboard.log_weights(epoch, model)

    # Sync all processes
    comm.synchronize()


def print_by_rank(*args, **kwargs):
    """Print message by the first rank within each node."""
    if not kwargs.pop("rank", 0):
        print(*args, **kwargs)


def print_ex_rank(*args, **kwargs):
    """Print message for all ranks except the given rank."""
    if kwargs.pop("rank", 0):
        print(*args, **kwargs)


def set_env(args):
    assert torch.cuda.is_available(), "No GPU available, dist training stop! Use `train.py` instead."
    assert torch.cuda.device_count() >= args.gpus, f"{torch.cuda.device_count()} GPUs detected, " \
                                                   f"which is less than {args.gpus}"
    print(f"[INFO] Number GPUs: {args.gpus}.")
    visible_devices = ','.join(map(str, range(args.gpus)))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    if args.dist_url == "env://":
        check_env()
        args.world_size = os.environ["WORLD_SIZE"]
        args.rank = os.environ["RANK"]
        args.dist_url = f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"


if __name__ == '__main__':
    args = parser.parse_args()
    set_env(args)
    launch(
        main_func=main_worker,
        num_gpus_per_node=args.gpus,
        num_machines=args.world_size,
        machine_rank=args.rank,
        dist_url=args.dist_url,
        args=(args,),
    )
