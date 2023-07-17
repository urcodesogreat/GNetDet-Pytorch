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

"""Convert Pytorch checkpoint to chip format."""

import json
import os
import shutil
import numpy as np
import torch
import datetime
import logging

from gnetmdk.gti.utils import update_dat_json, update_model_json


# When on-chip net performance is not satisfactory, you may set _DEBUG_CONVERSION to True to see
# more details during conversion by setting environment variable before running conversion script:
#   GTI_DEBUG_CONVERSION=True python convert_to_chip.py
_DEBUG_CONVERSION = os.environ.get("GTI_DEBUG_CONVERSION") == "True"
_CONVERSION_LOG_LEVEL = logging.DEBUG if _DEBUG_CONVERSION else logging.INFO
_logger = logging.getLogger(__name__)
_logger.setLevel(_CONVERSION_LOG_LEVEL)

from gnetmdk.gti.chip import driver
from gnetmdk.gti.config import gticonfig
import gnetmdk.gti.quantize as Q


def convert(
        chip,
        checkpoint,
        dat_json,
        model_json,
        out_model,
        dump_mode
):
    """Convert checkpoint to chip-compatible .net
        Generate output net and write to disk.

    Args:
        chip (str): chip type
        net (str): type of net corresponding to checkpoint
        checkpoint (str): path of checkpoint, e.g. checkpoints/best/2801_step1.pt
        dat_json (str): path of DAT definition JSON
        model_json (str): path of MODEL definition JSON
        out_model (str): path of output net to be generated
        dump_mode (bool): if True, keep all intermediate files; else dump them
        classifier_mode (bool): if False, dumps host layers responsible
            for doing classification based on chip output
        dump_mode and classifier_mode cannot be both True

    Returns:
        net_config_lst (list of dicts): list of dat_json, each stored as dict
    """

    state_dict = torch.load(checkpoint, map_location='cpu')  # ["model_state_dict"]

    _, state_dict = separate_state_dict(state_dict)

    debug_dir = os.path.join(os.path.dirname(out_model), "debug")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)

    data_files, net_config_lst = convert_chip_layers(
        chip=chip,
        state_dict=state_dict,
        dat_json_prefix=dat_json,
        save_dir=debug_dir,
        dump_mode=dump_mode
    )  # creates chip.dat, fills in data_files dictionary, updates dat.json

    model_json_out = os.path.join(debug_dir, os.path.basename(model_json))

    update_model_json(
        net_config_lst,
        model_json,
        data_files,
        model_json_out,
        dump_mode
    )

    if os.path.exists(out_model):
        _logger.warning("{} already exists and will be overwritten".format(out_model))

    driver.compose_model(json_file=model_json_out, model_file=out_model)
    if not (_DEBUG_CONVERSION or dump_mode):
        _logger.info("Removing intermediate files generated during conversion")
        shutil.rmtree(debug_dir)
    _logger.info("Successfully generated {}".format(out_model))
    return net_config_lst


def convert_chip_layers(
        chip,
        state_dict,
        dat_json_prefix,
        save_dir,
        dump_mode=False
):
    """Convert chip layers into .DAT file

    Args:
        chip (str): chip type
        net (str): type of net
        state_dict (dict): net state dictionary of checkpoint
        dat_json_prefix (str): path of DAT definition JSON
        save_dir (str): directory to save intermediate files

    Returns:
        data files dictionary consisting of paths to data files; look up by key:
            "dat0", or "dat1", "dat2" (multiple chip cases)
        Main purpose of function is to generate these files.
        This function also generates filter#.txt and bias#.txt.
    """
    bias_keys = [key for key in state_dict.keys() if 'bias' in key]
    partitions = partition_by_chip(bias_keys)

    filter_prefix = os.path.join(save_dir, "filter")
    bias_prefix = os.path.join(save_dir, "bias")
    dat_prefix = os.path.join(save_dir, "chip")
    data_files = {}

    # get filter, bias, shifts
    num_chips = len(partitions)

    # chunhe
    num_chips = 1

    net_config_lst = [0] * num_chips
    # for chip_idx, key_list in enumerate(partitions):

    filter_file = filter_prefix + ".txt"
    bias_file = bias_prefix + ".txt"
    dat_out = dat_prefix + ".dat"

    flat_filter = []
    flat_bias = []
    bit_shifts = []

    for key_idx, key in enumerate(partitions):

        key = key[0]
        bias = state_dict[key].float()
        key_prefix = key[:-4]

        weight = state_dict[key_prefix + "weight"].float()
        if key_idx == len(partitions) - 1:
            w_shape = weight.shape
            w_zeros = np.zeros([256 - w_shape[0], w_shape[1], w_shape[2], w_shape[3]])
            weight = torch.tensor(np.concatenate((weight, w_zeros), axis=0))
            b_shape = bias.shape
            b_zeros = np.zeros([256 - b_shape[0], ])
            bias = torch.tensor(np.concatenate((bias, b_zeros), axis=0))

        mask_bit = 3
        if int(key_idx) >= 13:
            mask_bit = 8

        shift = Q.compute_shift(
            weight=weight,
            bias=bias,
            chip=chip,
            mask_bit=mask_bit
        )
        weight = Q.quantize_weight(
            weight=weight,
            mask_bit=mask_bit,
            shift=shift
        )
        bias = Q.QuantizeShift.apply(bias, shift)
        weight = weight.detach().numpy()
        bias = bias.detach().numpy()

        # Log detailed information for layer gains and parameter magnitudes
        _logger.debug("Layer: {}".format(get_layer_name(key)))
        _logger.debug(
            "|W|max: {}, |B|max: {}, Shift: {}".format(
                np.amax(np.absolute(weight)), np.amax(np.absolute(bias)), shift
            )
        )
        _logger.debug("")

        # gnetfc special handling
        # pad the last convolutional layer weight and bias output channels with 0
        flat_filter.append(weight.ravel())
        flat_bias.append(bias.ravel())
        bit_shifts.append(shift)

    _logger.info("Converting convolutional layers to .DAT file")
    flat_filter = np.concatenate(flat_filter)
    flat_bias = np.concatenate(flat_bias)
    flat_filter.tofile(filter_file, sep="\n", format="%.16e")
    flat_bias.tofile(bias_file, sep="\n", format="%.16e")
    dat_json_out = os.path.join(save_dir, os.path.basename(dat_json_prefix))

    net_config = update_dat_json(
        dat_json=dat_json_prefix,
        new_shifts=bit_shifts,
        dat_json_out=dat_json_out,
        dump_mode=dump_mode
    )
    net_config_lst[0] = net_config
    # now that dat_json if updated, filter/bias files are written to disk
    # write chip.dat to disk @ dat_out
    # print("b4 config")
    gticonfig(
        dat_json=dat_json_out,
        filter_file=filter_file,
        bias_file=bias_file,
        dat_out=dat_out,
        save_dir=save_dir
    )
    # print("after config")
    data_files["dat"] = os.path.realpath(dat_out)
    return data_files, net_config_lst


# Misc
# separates chip layer vars from host layer vars
# host layer vars removed from original state_dict and returned separately
# warns about keys that are not obviously host/chip layer
def separate_state_dict(state_dict):
    host_state_dict = {key: value for key, value in state_dict.items() if "loc" in key or "conf" in key}
    for key in host_state_dict:
        del state_dict[key]
    """
    for key in state_dict:
        if "chip" not in key:
            _logger.warning("Unrecognized key in state_dict: " + key)
    """
    return host_state_dict, state_dict


# by default, pytorch registers parameters in the state_dict in the order of creation
# this function assumes that chip_layer0 (and all its associated parameters)
# are created before chip_layer1, chip_layer2, etc
# each entry in partitions is a list of keys associated with a chip_layer
# so returns [[keys in chip_layer0], [keys in chip_layer1]...]
def partition_by_chip(key_list):
    partitions = []
    current_chip = get_chip_name(key_list[0])
    keys_in_current_chip = []
    for key in key_list:
        new_chip = get_chip_name(key)
        if current_chip == new_chip:
            keys_in_current_chip.append(key)
        else:
            current_chip = new_chip
            partitions.append(keys_in_current_chip)
            keys_in_current_chip = [key]
    partitions.append(keys_in_current_chip)
    return partitions


# key names are generally module.chip_layer#.#.#.operation.parameter#
# returns the chip_layer# part
def get_chip_name(key):
    return key.split('.', 2)[1]


def get_layer_name(key):
    key = key.split('.', 2)[2]
    return key.rsplit('.', 2)[0]


def padding_zeros(weight, bias, out_channels=256):
    weight = np.pad(
        array=weight,
        pad_width=(
            (0, out_channels - weight.shape[0]),
            (0, 0),
            (0, 0),
            (0, 0)
        ),
        mode="constant"
    )
    bias = np.pad(
        array=bias,
        pad_width=(0, out_channels - bias.shape[0]),
        mode="constant"
    )
    return weight, bias
