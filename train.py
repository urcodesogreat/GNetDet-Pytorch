import os
import numpy as np
import torch

from gnetmdk.layers import GNetDet, GNetDetLoss
from gnetmdk.dataset import get_gnetdet_dataloader
from gnetmdk.checkpoint import GNetDetCheckPointer
from gnetmdk.evaluation import Evaluator, Tensorboard
from gnetmdk.utils.experiment import progress_bar
from configs import get_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
cfg = get_config()

# logging string
LOGGING_FORMAT = "- loss: {loss:<.4f} " \
                 "- loc_loss: {loc_loss:<.4f} " \
                 "- cls_loss: {cls_loss:<.4f} " \
                 "- cont_loss: {contain_loss:<.4f} " \
                 "- nocont_loss: {not_contain_loss:<.4f} " \
                 "- noobj_loss: {noobj_loss:<.4f} "

# Build GNetDet
model = GNetDet(cfg)

if use_gpu:
    model.cuda()
    print('\ncuda:', torch.cuda.current_device(), "NUM GPUs:", torch.cuda.device_count())

# Checkpointer
print("Reading checkpoint from:", os.path.relpath(cfg.checkpoint_path))
checkpointer = GNetDetCheckPointer(cfg, model=model)
checkpointer.load()

# Get save and load paths
best_save_path, last_save_path = cfg.get_ckpt_path()

# Loss and Evaluator
criterion = GNetDetLoss(cfg)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=cfg.learning_rate,
                            momentum=cfg.lr_momentum,
                            weight_decay=cfg.weight_decay)

evaluator = Evaluator()                     # mAP evaluator
tensorboard = Tensorboard(step=cfg.step)    # Tensorboard logger
tensorboard.log_graph(model, device="cpu" if not use_gpu else "cuda")  # Add model graph

# Load dataset
_, train_loader = get_gnetdet_dataloader(cfg, train=True)
_, valid_loader = get_gnetdet_dataloader(cfg, train=False)
print('the dataset has %d images' % (len(train_loader) * cfg.batch_size))
print('the batch_size is %d' % cfg.batch_size)

# Start Training
best_test_loss = np.inf
for epoch in range(0, cfg.num_epochs):
    if (epoch + 1) % 10 == 0:
        cfg.learning_rate = cfg.learning_rate * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.learning_rate
    print('\n\nEpoch %d / %d' % (epoch + 1, cfg.num_epochs))
    print('Learning Rate: {:.3e}'.format(cfg.learning_rate))

    model.train()
    total_loss = 0.
    monitor = progress_bar(cfg.log_freq)
    for i, (images, target, *_) in enumerate(train_loader):
        if use_gpu:
            images = images.to("cuda")
            target = target.to("cuda")

        # forward
        pred = model(images)
        losses = criterion(pred, target)
        loss = sum(losses.values())
        total_loss += loss.data.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log info
        info_str = LOGGING_FORMAT.format(loss=total_loss / (i + 1), **losses)
        monitor(i, len(train_loader), info_str)

    # validation
    print("Evaluation...")
    model.eval()
    validation_loss = 0.0
    monitor = progress_bar(cfg.log_freq)
    with torch.no_grad(), evaluator:
        for i, (images, target, gt_bboxes, whs, fnames) in enumerate(valid_loader):
            if use_gpu:
                images = images.to("cuda", non_blocking=True)
                target = target.to("cuda", non_blocking=True)

            # forward
            pred = model(images)
            losses = criterion(pred, target)
            loss = sum(losses.values())
            validation_loss += loss.data.item()

            # mAP evaluation
            evaluator.eval(pred, gt_bboxes, whs, fnames)

            # log info
            info_str = LOGGING_FORMAT.format(loss=total_loss / (i + 1), **losses)
            monitor(i, len(valid_loader), info_str)

        validation_loss /= len(valid_loader)
        print("validation loss:", validation_loss)
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            print('Saving best checkpoint to ', os.path.relpath(best_save_path))
            checkpointer.save(best_save_path)

        print('Saving last checkpoint to ', os.path.relpath(last_save_path))
        checkpointer.save(last_save_path)

    # Log info to Tensorboard
    tensorboard.log_scalars(epoch, "Loss", {"train": total_loss / len(train_loader), "validation": validation_loss})
    tensorboard.log_scalar(epoch, "LR", cfg.learning_rate)
    tensorboard.log_scalar(epoch, "mAP", evaluator.mAP)
    tensorboard.log_weights(epoch, model)
