import torch
import functools
import collections
from torch.nn import BatchNorm2d, Parameter

from gnetmdk.config import BaseConfig
from gnetmdk.layers import GNetDet, Conv2d, ReLU


def get_folded_weights(conv: Conv2d, bn: BatchNorm2d):
    """
    Fold BN to previous CONV weights and bias.
    """
    conv_w, conv_b = conv.weight, conv.bias
    mean, var, gamma, beta, eps = bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
    with torch.no_grad():
        sigma = torch.sqrt(var + eps)
        new_conv_w = conv_w * (gamma / sigma).view(-1, 1, 1, 1)
        new_conv_b = (conv_b - mean) * gamma / sigma + beta
    return Parameter(new_conv_w), Parameter(new_conv_b)


def fuse_bn(model: GNetDet, cfg: BaseConfig):
    """
    Fuse GNetDet BN layers into previous CONV layers.
    """
    assert contain_bn(model), f"No BN to fuse!"
    if not cfg.batch_norm:
        raise TypeError("cfg.batch_norm is False!")
    
    def iter_gnetdet_without_bn(model):
        for layer in model.modules():
            if isinstance(layer, Conv2d):
                yield layer
    
    def get_shortname(layer_name: str):
        short_name = layer_name.split('.')[-1]
        return short_name
        
    # Create a fused model
    cfg = cfg.clone()
    cfg.batch_norm = False
    fused_model = type(model)(cfg)
    fused_model_iter = iter_gnetdet_without_bn(fused_model)
    
    # Fold BN to CONV one layer at a time
    last_conv = None
    for name, layer in model.named_modules():
        if isinstance(layer, Conv2d):
            last_conv = (name, layer)
        elif isinstance(layer, BatchNorm2d):
            fused_conv = next(fused_model_iter)
            fused_conv.weight, fused_conv.bias = get_folded_weights(last_conv[1], layer)
            print(f"FUSE\t{get_shortname(last_conv[0]):>8}\t<---\t{get_shortname(name):<10}\t\tCOMPLETE")
    return fused_model


def contain_bn(model):
    """
    Returns a bool flag indicating whether the model contains BN layer.
    """
    has_bn = False
    for layer in model.modules():
        if isinstance(layer, BatchNorm2d):
            has_bn = True
            break
    return has_bn


def debug_outputs(bn_model: GNetDet, fused_model: GNetDet, cls):
    output_bn_model = collections.OrderedDict()
    output_fused_model = collections.OrderedDict()
    
    def hook_layer_output(model, inputs, outputs, store_dict: dict, layer_name: str):
        store_dict[layer_name] = outputs
    
    # Register hook
    for name, layer in bn_model.named_modules():
        if isinstance(layer, cls):
            layer.register_forward_hook(
                functools.partial(hook_layer_output, store_dict=output_bn_model, layer_name=name)
            )
    
    # Register hook
    for name, layer in fused_model.named_modules():
        if isinstance(layer, cls):
            layer.register_forward_hook(
                functools.partial(hook_layer_output, store_dict=output_fused_model, layer_name=name)
            )
    
    bn_model.eval()
    fused_model.eval()
    
    def call(input_tensor):
        with torch.no_grad():
            bn_out = bn_model(input_tensor)
            fused_out = fused_model(input_tensor)
            print(bn_out.reshape(-1)[:10])
            print(fused_out.reshape(-1)[:10])
            print()
            
        for k in output_bn_model:
            print(k)
            print("\t", output_bn_model[k].view(-1)[:10])
            print("\t", output_fused_model[k].view(-1)[:10])
            print("\t", output_bn_model[k].view(-1)[-10:])
            print("\t", output_fused_model[k].view(-1)[-10:])
            
        return output_bn_model, output_fused_model
    
    return call


if __name__ == '__main__':
    import random
    import numpy as np
    
    pass
    
    # from configs import  Config
    #
    # torch.use_deterministic_algorithms(True)
    # torch.random.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)
    #
    # # Fake image
    # fake_image = torch.randn([1, 3, 448, 448]) * 31.
    # print("fake image: ", fake_image.view(-1)[:10])
    # print()
    #
    # # Test step1 model and step2 model
    # cfg1 = Config(step=1)
    # cfg1.batch_norm = True
    # cfg1.checkpoint_path = r"../../checkpoint/step1/best.pth"
    # model1 = GNetDet(cfg1)
    # model1.load_state_dict(torch.load(cfg1.checkpoint_path))
    #
    # cfg2 = Config(step=2)
    # cfg2.batch_norm = True
    # cfg2.checkpoint_path = r"../../checkpoint/step1/best.pth"
    # model2 = GNetDet(cfg2)
    # model2.load_state_dict(torch.load(cfg2.checkpoint_path))
    #
    # # debugger = debug_outputs(model1, model2, Conv2d)
    # # debugger(fake_image)
    #
    # # Test bn-model and fused-model
    # fused_model = fuse_bn(model1, cfg1)
    #
    # debugger = debug_outputs(model1, fused_model, ReLU)
    # debugger(fake_image)
    