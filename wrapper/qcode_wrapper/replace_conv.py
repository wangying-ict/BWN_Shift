r"""
    Replace `conv` with `convq`;
    Replace `Linear` with `LinearQ`
"""

from models.modules import Conv2dQv2, LinearQv2, ActQv2
from models.modules import Conv2dBP, LinearBP
import torch.nn as nn
import ipdb


def quantize_scale_and_bias(model, bias_bits=8, scale_bits=8):
    for module_name, module in model.named_modules():
        if isinstance(module, ActQv2):
            if bias_bits > 0:
                module.set_out_scale(True)
            module.set_scale_bits(nbits=scale_bits)
        elif isinstance(module, LinearQv2) or isinstance(module, Conv2dQv2):
            module.set_scale_bits(nbits=scale_bits)
            module.set_bias_bits(nbits=bias_bits)
    return model


def add_radix_position(model):
    for module_name, module in model.named_modules():
        ipdb.set_trace()
        if isinstance(module, nn.Linear):
            m = LinearBP()
            pass
        if isinstance(module, nn.Conv2d):
            m = Conv2dBP()
            pass
