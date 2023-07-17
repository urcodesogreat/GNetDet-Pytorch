#!/usr/bin/env python3
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

"""Conversion from trained checkpoint to chip layers"""
import os
import sys
import torch
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnetmdk.gti import converter
from gnetmdk.utils.experiment import silent
from configs import get_config

with silent():
    cfg = get_config()

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(module)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

chip = cfg.chip
img_size = cfg.image_size
checkpoint = cfg.convert_model_path
dat_json = cfg.dat_json_path
model_json = cfg.model_json_path
out_model = cfg.out_model_path
dump_mode = cfg.dump_mode
output_channel = cfg.chip_depth

if img_size == 448:
    dat_json = cfg.dat_json_448_path
    model_json = cfg.model_json_448_path


def check_pattern():
    cfg = get_config()

    num_pre_grid = cfg.grid_depth
    chip_output = cfg.chip_depth

    net = torch.load(cfg.best_ckpt_path, map_location="cpu")
    weight = np.zeros(chip_output * 256 * 3 * 3).reshape(chip_output, 256, 3, 3)
    bias = np.zeros(chip_output).reshape(chip_output)

    weight[0:num_pre_grid, :, :, :] = net["features.conv15.weight"].cpu().numpy()
    bias[0:num_pre_grid] = net["features.conv15.bias"].cpu().numpy()

    net["features.conv15.weight"] = torch.tensor(weight)
    net["features.conv15.bias"] = torch.tensor(bias)

    torch.save(net, cfg.convert_model_path)


def main():
    """Given a chip compatible checkpoint, convert it to a format the chip understands

    Args:
        checkpoint (str): full file path to checkpoint
        net_dir (str): dir to place layers file/store intermediate files
        layers (str): architecture (vgg16/mobilenet/etc) to convert
        chip (str): name of GTI chip for checkpoint to be deployed on

    Returns:
        None. Converts checkpoint and write it to disk.
    """

    check_pattern()
    net_config_lst = converter.convert(
        chip=chip,
        checkpoint=checkpoint,
        dat_json=dat_json,
        model_json=model_json,
        out_model=out_model,
        dump_mode=dump_mode
    )


if __name__ == "__main__":
    main()
