import numpy as np
import torch.nn as nn

import models.modules as my_nn
import torch
import ipdb

__all__ = ['debug_graph_hooks', 'save_inner_hooks']


def save_inner_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, my_nn.Conv2dQ) or isinstance(module, my_nn.LinearQ) \
                or isinstance(module, my_nn.ActQ) or isinstance(module, nn.MaxPool2d) \
                or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) \
                or isinstance(module, my_nn.LinearBP) or isinstance(module, my_nn.Conv2dBP) \
                or isinstance(module, my_nn.Conv2dQv2) or isinstance(module, my_nn.LinearQv2):
            # TODO: ReLU(inplace=false) MaxPool ????
            module.name = name
            module.register_forward_hook(save_inner_data)


def save_inner_data(self, input, output):
    if len(output) == 2:
        out = output[0]
    else:
        out = output
    print('saving {} shape: {}'.format(self.name + 'out', out.size()))
    nu = out.detach().cpu().numpy()
    np_save = nu.reshape(-1, nu.shape[-1])
    np.savetxt('{}_out.txt'.format(self.name), np_save, delimiter=' ', fmt='%.8f')
    np.save('{}_out'.format(self.name), nu)

    in_data = input
    while not isinstance(in_data, torch.Tensor):
        in_data = in_data[0]
    print('saving {} shape: {}'.format(self.name + 'in', in_data.size()))
    nu = in_data.detach().cpu().numpy()
    np_save = nu.reshape(-1, nu.shape[-1])
    np.savetxt('{}_in.txt'.format(self.name), np_save, delimiter=' ', fmt='%.8f')
    np.save('{}_in'.format(self.name), nu)


def debug_graph(self, input, output):
    print('{}: type:{} input:{} ==> output:{} (max: {})'.format(self.name, type_str(self), [i.size() for i in input],
                                                                output.size(), output.max()))


def debug_graph_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or \
                isinstance(module, nn.MaxPool2d) or isinstance(module, my_nn.Concat) or \
                isinstance(module, nn.ZeroPad2d) or isinstance(module, my_nn.UpSample) or \
                isinstance(module, my_nn.CReLU):
            module.name = name
            module.register_forward_hook(debug_graph)


def type_str(module):
    if isinstance(module, nn.Conv2d):
        return 'Conv2d'
    if isinstance(module, nn.MaxPool2d):
        return 'MaxPool2d'
    if isinstance(module, nn.Linear):
        return 'Linear'
    if isinstance(module, my_nn.Concat):
        return 'Concat'
    if isinstance(module, nn.ZeroPad2d):
        return 'ZeroPad2d'
    if isinstance(module, my_nn.UpSample):
        return 'Upsample'
    if isinstance(module, my_nn.CReLU):
        return 'CReLU'
    return 'Emtpy'
