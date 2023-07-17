import os
import sys
import time
import torch

from torch.utils.tensorboard import SummaryWriter


class Tensorboard:
    """Support Tensorboard logging.

    @Usage:

    model = gnetdet(*args, **kwargs)

    tensorboard = Tensorboard(logdir, step=cfg.step)
    tensorboard.log_graph(model)  # log graph immediately

    # train loop
    for epoch in range(epochs):

        train_loop()
        validation_step()

        # tensorboard logging after validation step
        tensorboard.log_scalars(epoch, "Loss", {"validation": validation_loss})
        tensorboard.log_scalar(epoch, "LR", LEARNING_RATE)
        tensorboard.log_scalar(epoch, "mAP", evaluator.mAP)
        tensorboard.log_hist(epoch, model)

    During training, this class will log the above information into `logdir (default is ./log)`.
    To check those logging info:
        1. open a terminal, type the command:
            tensorboard --logdir=log
        2. open a browser, type the url address:
            localhost:6006
    """
    def __init__(self, logdir="log", step: int = 1, model=None, suffix=""):
        timestamp = time.strftime("%Y-%m-%d-%X", time.localtime())
        self.logdir = os.path.join(logdir, f"Step-{step}", timestamp)
        self._writer = SummaryWriter(self.logdir)
        self._flag_added_graph = False
        self.step = step

        if model is not None:
            self.log_graph(model)

    def log_graph(self, model: torch.nn.Module, input_shape=None, device="cuda"):
        from torch.nn.parallel import DistributedDataParallel
        if not self._flag_added_graph:
            if input_shape is None:
                input_tensor = torch.zeros([1, 3, 448, 448])
            else:
                input_tensor = torch.zeros(input_shape)
            input_tensor = input_tensor.to(dtype=torch.float32, device=device)
            if isinstance(model, DistributedDataParallel):
                model = model.module
            self._writer.add_graph(model, input_tensor)
            self._flag_added_graph = True

    def log_scalars(self, epoch, main_tag, value_dict):
        self._writer.add_scalars(main_tag, value_dict, global_step=epoch)

    def log_scalar(self, epoch, tag, value):
        self._writer.add_scalar(tag, value, global_step=epoch)

    def log_hist(self, epoch, tag, values):
        self._writer.add_histogram(tag, values, global_step=epoch)

    def log_weights(self, epoch, model: torch.nn.Module):
        for name, param in model.named_parameters():
            try:
                if param.requires_grad and param.grad is not None:
                    self.log_hist(epoch, name + "_grad", param.grad)
                self.log_hist(epoch, name + "_data", param)
            except Exception as e:
                sys.stderr.write(f"{e}\n")
                sys.stderr.write(f"Epoch: {epoch}, {name}:\n{param}\n{param.grad}\n")
