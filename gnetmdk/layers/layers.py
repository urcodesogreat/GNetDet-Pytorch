import torch
import torch.nn.functional as F

from gnetmdk.gti import quantize as Q


class Conv2d(torch.nn.Conv2d):
    def __init__(self, quantize=False, chip="5801", mask_bit=None, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        self.quantize = quantize
        self.chip = chip
        self.mask_bit = mask_bit

    def forward(self, x):
        if self.quantize:
            shift = Q.compute_shift(
                self.weight,
                self.bias,
                self.chip,
                self.mask_bit
            ).item()
            tmp_weight = Q.quantize_weight(
                self.weight,
                self.mask_bit,
                shift,
            )
            tmp_bias = Q.QuantizeShift.apply(self.bias, shift)
            return F.conv2d(
                x,
                tmp_weight,
                tmp_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class ReLU(torch.nn.ReLU):
    def __init__(self, quantize=False, cap=31.0, max_act=31, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.quantize = quantize
        self.cap = cap
        self.max_act = max_act

    def forward(self, x):
        if self.quantize:
            out = 0.5 * (torch.abs(x) - torch.abs(x - self.cap) + self.cap)
            factor = (self.max_act / self.cap)  # .item() #uses less GPU RAM
            return Q.Round.apply(out * factor) / factor
        return F.relu(x, inplace=self.inplace)
