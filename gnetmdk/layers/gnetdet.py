import math
import os

import torch
import collections
import functools
import numpy as np

from gnetmdk import ROOT
from gnetmdk.layers import Conv2d, ReLU
from gnetmdk.config import configurable


class GNetDet(torch.nn.Module):

    @configurable
    def __init__(self, features, grid_num=14, cal=False, scale=31.0, cap_txt="relu_cap.txt"):
        super(GNetDet, self).__init__()
        self.features = features
        self.grid_num = grid_num
        self.cal = cal
        self.scale = scale / 31.0
        self.cap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cap_txt = cap_txt
        self._initialize_weights()
        self._hook_handlers = []
        if self.cal:
            self._register_cal_hooks()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def freeze_backbone(self):
        """
        Freeze all layers except last CONV layer, such that the layers
        are untrainable.
        """
        count = 0
        for name, layer in self.named_modules():
            if isinstance(layer, (Conv2d, torch.nn.BatchNorm2d)) and count < 15:
                count += 1
                try:
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                except Exception:
                    pass
        print("GNetDet Backbone Frozen.")

    def unfreeze(self):
        """
        Unfreeze all layers.
        """
        for layer in self.modules():
            if isinstance(layer, (Conv2d, torch.nn.BatchNorm2d)):
                try:
                    layer.weight.requires_grad = True
                    layer.bias.requires_grad = True
                except Exception:
                    pass

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1, self.grid_num, self.grid_num)
        # x.cpu().detach().numpy().astype(np.uint8).tofile("layers.bin")
        # step4
        x = x * self.scale
        x = x.permute(0, 2, 3, 1)

        if self.cal:
            self.write_cap_txt(self.cap_txt)
        return x

    def write_cap_txt(self, file):
        file_handle = open(file, mode='w')
        for cap in self.cap:
            file_handle.write(str(cap) + "\n")

    def _register_cal_hooks(self):
        """
        Register a calibration hook that records 99 percentile of the outputs
        from each conv layer.
        """
        def _cal_hook(module, inputs, outputs, i=0):
            old_cap = self.cap[i]
            # CAUTION: torch.quantile() limits tensor size, and slower than np.percentile()
            self.cap[i] = max(old_cap, np.percentile(outputs.cpu().detach(), 99))

        cap_index = 0
        for name, layer in self.named_modules():
            if isinstance(layer, Conv2d):
                layer.register_forward_hook(functools.partial(_cal_hook, i=cap_index))
                cap_index += 1

    def register_activation_hooks(self, logger, epoch=None):
        if logger is None: return

        def _log_relu(module, inputs, outputs, epoch=None, name=""):
            logger.log_hist(epoch, name+"_out", outputs)

        for name, layer in self.named_modules():
            if isinstance(layer, Conv2d):
                handler = layer.register_forward_hook(functools.partial(_log_relu, epoch=epoch, name=name))
                self._hook_handlers.append(handler)

    def remove_activation_hooks(self):
        for handler in self._hook_handlers:
            handler.remove()
        self._hook_handlers.clear()

    @classmethod
    def from_config(cls, cfg):
        image_size = cfg.image_size
        if image_size == 448:
            struct = [
                64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'C', 256, 256,
                cfg.grid_depth
            ]
        elif image_size == 224:
            struct = [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'C', 256, 256,
                cfg.grid_depth
            ]
        else:
            raise ValueError("only support 224 or 448 size.")

        return {
            "features": _make_layers(cfg, struct),
            "grid_num": cfg.grid_size,
            "cal": cfg.cal,
            "scale": cfg.scale,
            "cap_txt": cfg.cap_txt_path,
        }


def _make_layers(cfg, model_structure):
    if not os.path.exists(cfg.cap_txt_path):
        cap_name = os.path.basename(cfg.cap_txt_path)
        cap_txt_path = os.path.join(ROOT, "model", cap_name)
        os.system(f"touch {cap_txt_path}")
    layers = collections.OrderedDict()
    major_layer = 1
    file_handle = open(cfg.cap_txt_path)
    cap = None
    conv_id, pool_id, relu_id, norm_id = 0, 0, 0, 0
    in_channels = 3
    for v in model_structure:
        if v == 'M':
            major_layer += 1
            pool_id += 1
            layers["pool" + str(pool_id)] = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        elif v == 'C':
            major_layer += 1
        else:
            if cfg.quant_r:
                cap = file_handle.readline()
                cap = float(cap)
            if cfg.quant_31:
                cap = 31
            mask_bit = 3
            if cfg.image_size == 448 and major_layer == 7:
                mask_bit = 8
            if cfg.image_size == 224 and major_layer == 6:
                mask_bit = 8

            conv2d = Conv2d(quantize=cfg.quant_w, chip=cfg.chip, mask_bit=mask_bit,
                            in_channels=in_channels, out_channels=v,
                            kernel_size=3, padding=1, dilation=1, stride=1, groups=1)
            if cfg.batch_norm:
                layers["conv" + str(conv_id)] = conv2d
                conv_id += 1
                layers["BatchNorm" + str(norm_id)] = torch.nn.BatchNorm2d(v)
                norm_id += 1
                layers["relu" + str(relu_id)] = ReLU(quantize=cfg.quant_r, cap=cap, max_act=31)
                relu_id += 1
            else:
                layers["conv" + str(conv_id)] = conv2d
                conv_id += 1
                layers["relu" + str(relu_id)] = ReLU(quantize=cfg.quant_r, cap=cap, max_act=31)
                relu_id += 1
            in_channels = v
    file_handle.close()
    return torch.nn.Sequential(layers)
