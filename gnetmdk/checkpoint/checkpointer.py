import os
import torch
import weakref
import collections
import itertools

from torch.nn.parallel import DistributedDataParallel

from gnetmdk.dist import comm
from gnetmdk.config import configurable


class GNetDetCheckPointer(object):
    """Load checkpoint for all ranks but only save ckpt for local_rank==0"""

    best_loss: float

    @configurable
    def __init__(self, model, ckpt_path: str, edit_gain: bool = False, cap_txt: str = ""):
        assert os.path.isfile(ckpt_path), f"Wrong path: {ckpt_path}"
        assert (not edit_gain or os.path.isfile(cap_txt)), f"Wrong path: {cap_txt}"
        self.model: torch.nn.Module = weakref.proxy(model)
        self.ckpt_path = ckpt_path
        self.edit_gain = edit_gain
        self.cap_txt = cap_txt
        self.local_rank = comm.get_local_rank()
        self.map_location = f"cuda:{self.local_rank}"
        self.is_ddp_module = isinstance(model, DistributedDataParallel)
        self.is_ddp_ckpt = False

    def load(self):
        """Load checkpoint from file."""
        ckpt_state_dict = torch.load(self.ckpt_path, map_location=self.map_location)
        self.best_loss = ckpt_state_dict.pop("best_loss", float("inf"))

        if self.edit_gain:
            model_state_dict = collections.OrderedDict()
            relu_caps = list(map(float, open(self.cap_txt, 'r').readlines()))
            relu_caps = itertools.chain.from_iterable(zip(relu_caps, relu_caps))
            prev_cap = 31.
            for ckpt_k, ckpt_v in ckpt_state_dict.items():

                # DDP layers's name starts with "module."
                if ckpt_k.startswith("module.") and not self.is_ddp_ckpt:
                    self.is_ddp_ckpt = True

                if ckpt_k.endswith(".weight") and "conv" in ckpt_k:
                    w_gain = prev_cap / next(relu_caps)
                    ckpt_v *= w_gain

                elif ckpt_k.endswith(".bias") and "conv" in ckpt_k:
                    prev_cap = next(relu_caps)
                    b_gain = 31. / prev_cap
                    ckpt_v *= b_gain

                # Write to new state dict
                ckpt_k = self._correct_key(ckpt_k)
                model_state_dict[ckpt_k] = ckpt_v
        else:
            model_state_dict = self.model.state_dict()
            for ckpt_k, ckpt_v in ckpt_state_dict.items():

                # DDP layer's name starts with "module."
                if ckpt_k.startswith("module.") and not self.is_ddp_ckpt:
                    self.is_ddp_ckpt = True

                ckpt_k = self._correct_key(ckpt_k)
                if model_state_dict[ckpt_k].shape == ckpt_v.shape:
                    model_state_dict[ckpt_k] = ckpt_v

        # Load state dict
        self.model.load_state_dict(model_state_dict, strict=False)

    def _correct_key(self, ckpt_key):
        if self.is_ddp_module and not self.is_ddp_ckpt:
            return f"module.{ckpt_key}"
        elif not self.is_ddp_module and self.is_ddp_ckpt:
            return ckpt_key[7:]
        else:
            return ckpt_key

    def save(self, dest_path: str):
        """Save checkpoint to destination."""
        if self.local_rank == 0:
            if self.is_ddp_module:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            torch.save(state_dict, dest_path)

    @classmethod
    def from_config(cls, cfg, model):
        return {
            "model": model,
            "ckpt_path": cfg.checkpoint_path,
            "edit_gain": cfg.edit_gain,
            "cap_txt": cfg.cap_txt_path,
        }
