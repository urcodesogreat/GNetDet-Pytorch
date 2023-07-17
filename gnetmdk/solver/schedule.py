import torch
import inspect

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

REGISTERED_SCHEDULERS = {
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
}


def _get_registered_lr_scheduler(name: str, optimizer: Optimizer, *args, **kwargs):
    """
    Return a registered lr scheduler. The signature of scheduler's
    construct method must be correctly given by args and kwargs.

    Examples:
        # StepLR
        scheduler = _get_registered_lr_scheduler("StepLR", optimizer, step_size, gamma=0.1)

        # MultiStepLR
        scheduler = _get_registered_lr_scheduler("MultiStepLR", optimizer, milestones, gamma=0.1)

        # CosineAnnealingLR
        scheduler = _get_registered_lr_scheduler("CosineAnnealingLR", optimizer, T_max, eta_min=0)

        # CosineAnnealingWarmRestarts
        scheduler = _get_registered_lr_scheduler("CosineAnnealingWarmRestarts", optimizer, T_0, T_mult=1, eta_min=0)
    """
    assert (
        name in REGISTERED_SCHEDULERS
    ), f"Supported LRScheduler: {list(REGISTERED_SCHEDULERS.keys())}. Got: {name}"

    return REGISTERED_SCHEDULERS[name](optimizer, *args, **kwargs)


def get_lr_scheduler(
    cfg, optimizer: Optimizer,
    warmup_epochs: int = -1,
    min_lr: float = 1e-10,
    **kwargs,
):
    """
    Get a Warmup LR Scheduler, the name of scheduler must be registered.

    Registered LR Schedulers are:
        StepLR          - Default
        MultiStepLR
        CosineAnnealingLR
        CosineAnnealingWarmRestarts

    Usage Example

        # Instantiate a config object
        cfg = get_config()
        cfg.lr_scheduler = "MultiStepLR"   # Change the name of lr_scheduler

        # To get a lr scheduler without warmup
        lr_scheduler = get_lr_scheduler(cfg, optimizer)

        # Get a LR Scheduler with a few epochs of warmup
        lr_scheduler = get_lr_scheduler(cfg, optimizer, warmup_epochs=10)
    """
    # Introspection of init method
    assert cfg.lr_scheduler in REGISTERED_SCHEDULERS, f"Not a registered scheduler: {cfg.lr_scheduler}"
    scheduler = REGISTERED_SCHEDULERS[cfg.lr_scheduler]
    signature = inspect.signature(scheduler)
    scheduler_kwargs = {}
    for param in signature.parameters.values():
        if param.name == "optimizer":
            continue
        # If kwargs contains argument, then use it
        value = kwargs.pop(param.name, None)
        if value is not None:
            scheduler_kwargs[param.name] = value
            continue
        # Assume cfg has no attribute valued None
        value = getattr(cfg, param.name, None)
        if value is None and param.default != param.empty:
            scheduler_kwargs[param.name] = param.default
        else:
            scheduler_kwargs[param.name] = value

    # check if any None in kwargs
    null_args = list(k for k, v in scheduler_kwargs.items() if v is None)
    if any(null_args):
        msg = f"{cfg.lr_scheduler} requires cfg has attributes: {', '.join(null_args)}"
        raise AttributeError(msg)

    # Get LR Scheduler
    lr_schedule = _get_registered_lr_scheduler(cfg.lr_scheduler, optimizer, **scheduler_kwargs)

    # Warmup epochs
    if warmup_epochs <= 0:
        warmup_epochs = getattr(cfg, "warmup_epochs", 0)
    if warmup_epochs > 0:
        lr_schedule = WarmupLRScheduler(lr_schedule, warmup_epochs, min_lr)

    return lr_schedule


class WarmupLRScheduler(_LRScheduler):
    """
    Create a warmup scheduler and concatenate with another scheduler, such that
    lr will be linearly increased to the initial value of the latter one.
    """

    def __init__(self, after_scheduler: _LRScheduler, warmup_epochs: int, min_lr: float = 0.):
        assert isinstance(after_scheduler, _LRScheduler), after_scheduler
        self.warmup_epochs = max(0, warmup_epochs)
        self.after_scheduler = after_scheduler
        self.min_lr = min_lr
        super(WarmupLRScheduler, self).__init__(after_scheduler.optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.min_lr + self.last_epoch * (base_lr - self.min_lr) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            return self.after_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super(WarmupLRScheduler, self).step(epoch)
        else:
            if epoch is None:
                self.after_scheduler.step()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
            self._last_lr = self.after_scheduler.get_last_lr()


if __name__ == '__main__':
    from gnetmdk.config import BaseConfig


    def show_lr(scheduler, optimizer, title=""):
        import matplotlib.pyplot as plt
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            print(lrs[-1], scheduler.get_last_lr())
            optimizer.step()
            scheduler.step()

        plt.plot(lrs)
        plt.title(title)
        plt.show()


    cfg = BaseConfig()
    cfg.lr_scheduler = "CosineAnnealingWarmRestarts"
    cfg.T_0 = 20

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    scheduler = get_lr_scheduler(cfg, optimizer)
    show_lr(scheduler, optimizer, cfg.lr_scheduler)
