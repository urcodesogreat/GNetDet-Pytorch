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

"""Model util functions."""

import logging
import json

_logger = logging.getLogger(__name__)


# does not touch the other vars -> assumes they're already correct
def update_model_json(
        net_config_lst,
        model_json,
        data_files,
        model_json_out,
        dump_mode=False
):
    """Update full MODEL JSON with newly generated data file paths:
        dat0, dat1... (chip layers)

    Args:
        net_config_lst (list of dicts): list of dat jsons (as python dicts)
        model_json (str): path of layers definition JSON
        data_files (dict str:str): name:file path
        dat_json_out (str): path to write modified layers JSON
        dump_mode (bool): if True, chip will dump activations of all minor layers

    Returns:
        None"""

    with open(model_json, "r+") as f:
        model_def = json.load(f)

    # dump all host layers (because SDK 5.0)
    # TODO: clean all the layers.jsons?
    tmp = []
    for layer in model_def["layer"]:
        print("layer: ", layer)
        if layer["operation"] in ["GTICNN", "IMAGEREADER"]:
            tmp.append(layer)
    model_def["layer"] = tmp

    count_dat = 0
    for layer in model_def["layer"]:
        if layer["operation"] == "GTICNN":
            layer["data file"] = data_files["dat"]
            count_dat += 1
            if layer["device"]["chip"] == "2801" and dump_mode:
                layer["mode"] = 1

    # add SDK 5.0 support
    chip_nums = len(net_config_lst)
    chip_type = net_config_lst[0]['model'][0]['ChipType']

    fullmodel = FullModelConfig(model_def)

    for idx, net_config in enumerate(net_config_lst):
        model_def = fullmodel.update_fullmodel(idx + 1, net_config, chip_nums)

    with open(model_json_out, "w") as outf:
        json.dump(model_def, outf, indent=4, separators=(',', ': '), sort_keys=True)


# does not touch the other vars -> assumes they're already correct
# most of the other vars can be easily read/computed from the checkpoint
# image_size for each layer can be computed (knowing input size), but is annoying
# pooling information is not in the checkpoint
def update_dat_json(dat_json, new_shifts, dat_json_out, dump_mode):
    """Update DAT JSON with newly calculated bit shifts/scaling factors from checkpoint.

    Args:
        dat_json (str): path of DAT definition JSON
        new_shifts (list(int)): list of new shifts
        dat_json_out (str): path to write modified DAT JSON
        dump_mode (bool): if True, chip will dump activations of all minor layers

    Returns:
        net_config (dict): updated dat_json in dict form
    """

    with open(dat_json) as f:
        net_config = json.load(f)

    # add MajorLayerNumber
    net_config['model'][0]['MajorLayerNumber'] = len(net_config['layer'])

    # add major_layer and shift values to layers.json
    idx = 0
    for i, layer in enumerate(net_config['layer']):
        layer['major_layer'] = i + 1
        layer['scaling'] = []
        for j in range(layer['sublayer_number']):
            layer['scaling'].append(int(new_shifts[idx]))
            idx += 1

        # change layers.json learning mode to do the conversion
        if dump_mode:
            layer['learning'] = True
        else:
            if 'learning' in layer:
                layer['learning'] = False
            # else not present -> using default false

    with open(dat_json_out, 'w') as f:
        json.dump(net_config, f, indent=4, separators=(',', ': '), sort_keys=True)
    return net_config


class FullModelConfig(object):
    def __init__(self, fullmodel_config):
        self.fullmodel_config = fullmodel_config
        if "version" not in self.fullmodel_config:
            self.fullmodel_config['version'] = 100

        self.net_confg = None
        self.layer_idx = 0
        self.chip_type = 0

    def update_fullmodel(self, layer_idx, net_config, chip_nums):
        self.net_config = net_config
        self.layer_idx = layer_idx
        self.chip_nums = chip_nums

        cnn_layer = self.fullmodel_config['layer'][self.layer_idx]
        # add inputs array to cnn layer
        cnn_layer['inputs'] = [{
            "format": "byte",
            "prefilter": "interlace_tile_encode",
            "shape": [
                self.net_config['layer'][0]['image_size'],
                self.net_config['layer'][0]['image_size'],
                self.net_config['layer'][0]['input_channels']
            ]
        }]
        # add outputs array to cnn layer, consider learning mode, the implementation vary by chip type
        cnn_layer['outputs'] = []
        DEFAULT_TILE_SIZE = 14
        NUM_ENGINES = 16

        for idx, layer in enumerate(self.net_config['layer']):
            image_size = layer['image_size']
            output_channels = layer['output_channels']
            layer_scaledown = 0
            if self.chip_nums > 1 and self.layer_idx < self.chip_nums:
                layer_scaledown = -3
            upsample_mode = 0
            if 'upsample_enable' in layer and layer['upsample_enable']:
                image_size <<= 1
                output_channels = ((NUM_ENGINES - 1 + output_channels) / NUM_ENGINES) * NUM_ENGINES
                upsample_mode = 1
            output_format = 'byte'
            filter_type = "interlace_tile_decode"
            if 'ten_bits_enable' in layer and layer['ten_bits_enable']:
                output_format = 'float'
                filter_type = 'interlace_tile_10bits_decode'
            tile_size = image_size if image_size < DEFAULT_TILE_SIZE else DEFAULT_TILE_SIZE
            output_size = image_size * image_size * output_channels * 32 // 49
            if 'learning' in layer and layer['learning']:
                # check fake layer
                sublayers = layer['sublayer_number'] + 1 if self.need_fake_layer(layer) else layer['sublayer_number']
                for i in range(sublayers):
                    sub_output_channels = output_channels
                    # handle mobilenet one by one convolution
                    if i == 0 and self.depth_enabled(layer):
                        sub_output_channels = layer['input_channels']
                        sub_output_size = image_size * image_size * sub_output_channels * 32 // 49
                        cnn_layer["outputs"].append({
                            "format": output_format,
                            "postfilter": filter_type,
                            "shape": [
                                image_size,
                                image_size,
                                sub_output_channels,
                                tile_size * tile_size,
                                sub_output_size
                            ],
                            "layer scaledown": layer_scaledown,
                            "upsampling": upsample_mode
                        })
                    else:
                        cnn_layer["outputs"].append({
                            "format": output_format,
                            "postfilter": filter_type,
                            "shape": [
                                image_size,
                                image_size,
                                output_channels,
                                tile_size * tile_size,
                                output_size
                            ],
                            "layer scaledown": layer_scaledown,
                            "upsampling": upsample_mode
                        })
            elif 'last_layer_out' in layer and layer['last_layer_out']:
                cnn_layer["outputs"].append({
                    "format": output_format,
                    "postfilter": filter_type,
                    "shape": [
                        image_size,
                        image_size,
                        output_channels,
                        tile_size * tile_size,
                        output_size
                    ],
                    "layer scaledown": layer_scaledown,
                    "upsampling": upsample_mode
                })
            elif idx + 1 == len(self.net_config['layer']):  # add the last layer output
                if 'pooling' in layer and layer['pooling']:
                    image_size >>= 1
                    tile_size = DEFAULT_TILE_SIZE >> 1
                    if image_size == 7:  # fc_mode
                        filter_type = "fc77_decode"

                if filter_type == "fc77_decode":
                    output_size = image_size * image_size * output_channels * 64 // 49
                    layer_scaledown = 3
                else:
                    output_size = image_size * image_size * output_channels * 32 // 49
                cnn_layer["outputs"].append({
                    "format": output_format,
                    "postfilter": filter_type,
                    "shape": [
                        image_size,
                        image_size,
                        output_channels,
                        tile_size * tile_size,
                        output_size
                    ],
                    "layer scaledown": layer_scaledown,
                    "upsampling": upsample_mode
                })
        return self.fullmodel_config

    def need_fake_layer(self, layer):
        return 'resnet_shortcut_start_layers' in layer and 'pooling' in layer and layer['pooling'] and \
               layer['sublayer_number'] == layer['resnet_shortcut_start_layers'][-1] + 1

    def depth_enabled(self, layer):
        return 'depth_enable' in layer and layer['depth_enable'] \
               and 'one_coef' in layer and len(layer['one_coef']) > 0 \
               and layer['one_coef'][0] == 0
