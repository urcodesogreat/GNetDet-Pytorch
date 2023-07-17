'''
All Rights Reserved.

Copyright (c) 2017-2019, Gyrfalcon technology Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch
from torch.autograd import Function

from gnetmdk.gti.chip import spec

# epsilon to avoid zero division 
_EPSILON = 1e-6


def quantize_weight(weight, mask_bit, shift):
    if mask_bit == 1:
        return _Quantize1Bit.apply(weight, shift)
    elif mask_bit == 2:
        return _Quantize2Bit.apply(weight, shift)
    elif mask_bit == 3:
        return _Quantize3Bit.apply(weight, shift)
    elif mask_bit == 5:
        return _Quantize5Bit.apply(weight, shift)
    elif mask_bit == 8 or mask_bit == 12:
        return _QuantizeMoreBit.apply(weight, shift)
    else:
        raise ValueError('Unsupported {}-bit quantization'.format(mask_bit))


class _Quantize1Bit(Function):
    @staticmethod
    def forward(ctx, input, shift):
        mean_abs = torch.mean(torch.abs(input), (2, 3), keepdim=True)
        mean_abs = QuantizeShift.apply(mean_abs, shift)  # NxMx1x1
        return torch.where(input >= 0, mean_abs, -mean_abs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _Quantize2Bit(Function):
    @staticmethod
    def forward(ctx, input, shift):
        abs_input = torch.abs(input)
        mean_abs = torch.mean(abs_input, (2, 3), keepdim=True)
        mean_abs = QuantizeShift.apply(mean_abs, shift)  # NxMx1x1
        output = torch.where(abs_input >= mean_abs / 4., mean_abs,
                             torch.tensor(0, device=input.device, dtype=input.dtype))
        return torch.where(input >= 0, output, -output)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _Quantize3Bit(Function):
    @staticmethod
    def forward(ctx, input, shift):
        abs_input = torch.abs(input)
        mean_abs = torch.mean(abs_input, (2, 3), keepdim=True)  # NxMx1x1
        device = input.device
        dtype = input.dtype
        # TODO: maybe use < _eps instead?
        # TODO: replace with max like function?
        step = torch.where(mean_abs == 0, torch.tensor(_EPSILON, device=device, dtype=dtype), mean_abs) / 4.
        coef = (abs_input / step).int().float()  # NxMx3x3 #TODO: verify
        step = QuantizeShift.apply(step, shift)  # TODO: why quant after?
        output = torch.where(coef >= 3., torch.tensor(4, device=device, dtype=dtype), coef)
        return torch.where(input >= 0, output * step, output * -step)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _Quantize5Bit(Function):
    @staticmethod
    def forward(ctx, input, shift):
        max_abs, _ = torch.max(torch.abs(input), (2, 3), keepdim=True)  # NxMx1x1
        device = input.device
        dtype = input.dtype
        # TODO: replace with max like function?
        step = torch.where(max_abs == 0, torch.tensor(_EPSILON, device=device, dtype=dtype), max_abs) / 15.
        coef = Round.apply(input / step).int().float()
        step = QuantizeShift.apply(step, shift)
        return coef * step

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# TODO: refactor later
class _QuantizeMoreBit(Function):
    @staticmethod
    def forward(ctx, input, shift):
        return QuantizeShift.apply(input, shift)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# output is same shape & dtype as input
class QuantizeShift(Function):
    @staticmethod
    def forward(ctx, input, shift):
        return Round.apply(input * (2 ** shift)) / (2 ** shift)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# computes shift for a tensor; not clipped
# output is 0 dim tensor, int casted to float32
# _EPSILON is necessary because on cuda, .int() maps inf -> max_int32
# whereas on cpu, .int() maps inf -> min_int32
# TODO(Yin): refactor later
def _shift_helper(x, bits):
    return torch.log2((2. ** (bits - 1) - 1) / (_EPSILON + torch.max(torch.abs(x)))).int().float()


def compute_shift(weight, bias, chip, mask_bit):
    weight_bits, bias_bits = spec.schemes[chip][mask_bit]
    weight_shift = _shift_helper(weight, weight_bits)
    # print(mask_bit,'==========================')
    if bias is None:
        return torch.clamp(weight_shift, spec.MIN_SHIFT, spec.MAX_SHIFT)
    bias_shift = _shift_helper(bias, bias_bits)
    return torch.clamp(torch.min(weight_shift, bias_shift), spec.MIN_SHIFT, spec.MAX_SHIFT)


class Round(Function):
    """Simulate chip rounding (away from 0): 2.5 -> 3, not using default half-to-even 2.5 -> 2.
    gpu version of torch.round already does this
    cpu version seems to depend on instruction set"""

    @staticmethod
    def forward(ctx, input):
        return torch.where(input >= 0, torch.floor(input + 0.5), torch.ceil(input - 0.5))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
