import math
from enum import Enum
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

__all__ = ['Conv2dBP', 'Conv2dBPv2', 'LinearBP', 'Conv2dBNBP', 'LinearBPv2', 'count_bit', 'truncation', 'bit_sparse',
           'FunctionBitPruningSTE']


class q_modes(Enum):
    layer_wise = 1
    kernel_wise = 2


class FunctionStopGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, stopGradientMask):
        ctx.save_for_backward(stopGradientMask)
        return weight

    @staticmethod
    def backward(ctx, grad_outputs):
        stopGradientMask, = ctx.saved_tensors
        grad_inputs = grad_outputs * stopGradientMask
        return grad_inputs, None


bit_code1 = [0, 1, 2, 4, 8, 16, 32, 64]
bit_code1_threshold = [0.0] + [(bit_code1[i] + bit_code1[i + 1]) / 2 for i in range(len(bit_code1) - 1)] + [128]
bit_code2 = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40,
             48, 64, 65, 66, 68, 72, 80, 96]
bit_code2_threshold = [0.0] + [(bit_code2[i] + bit_code2[i + 1]) / 2 for i in range(len(bit_code2) - 1)] + [128]


# ================bit pruning core==================
class FunctionBitPruningSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, radix_position, log=False):
        weights_bp = bit_pruning_with_truncation(weights, radix_position, log)
        return weights_bp

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None


def bit_pruning_with_truncation(weights, radix_position, log=False, nbits=8):
    scale_factor = 2 ** radix_position
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data_int = linear_quantize_clamp(weights, scale_factor, clamp_min, clamp_max)
    negtive_index = q_data_int < 0
    q_data_int = q_data_int.abs()
    # if False:
    #     # stochastic round when
    #     pass
    if log:
        bit_code = bit_code1
        bit_code_threshold = bit_code1_threshold
    else:
        bit_code = bit_code2
        bit_code_threshold = bit_code2_threshold

    for i in range(len(bit_code) - 1):
        case = (bit_code_threshold[i] <= q_data_int) * (q_data_int < bit_code_threshold[i + 1])
        q_data_int = torch.where(case, torch.zeros_like(q_data_int).fill_(bit_code[i]), q_data_int)
    q_data_int = torch.where(negtive_index, -q_data_int, q_data_int)
    q_data = linear_dequantize(q_data_int, scale_factor)
    return q_data


# ================bit pruning core end===============


def expected_bit_sparsity_func(factor, input_bit_sparsity):
    return input_bit_sparsity + (1 - input_bit_sparsity) * factor


class Conv2dBPv2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 nbits=8, mode=q_modes.layer_wise, complement=False, log=False):
        super(Conv2dBPv2, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)
        if nbits <= 0:
            self.nbits = None
            return
        self.nbits = nbits
        self.log = log
        self.complement = complement  # use tows complement representation, otherwise true form
        self.stopGradientMask = None
        self.register_buffer('init_state', torch.zeros(1))
        if mode == q_modes.layer_wise:
            self.register_buffer('radix_position', torch.zeros(1))
        else:
            raise NotImplementedError

    def forward(self, input):
        if self.nbits is None:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.init_state == 0:  # need to set radix position
            # set radix position
            self.init_state.data.fill_(1)
            il = torch.log2(torch.max(self.weight.max(), self.weight.min().abs())) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            # quantize weight_fold
            weight_q, weight_int = truncation(self.weight, self.radix_position)
            if self.training:
                self.stopGradientMask = (weight_int > 0).float()
        weight_mask = FunctionStopGradient.apply(self.weight, self.stopGradientMask)
        # STE for quantized weight.
        weight_bp = FunctionBitPruningSTE.apply(weight_mask, self.radix_position, self.log)
        out = F.conv2d(input, weight_bp, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def extra_repr(self):
        s_prefix = super(Conv2dBPv2, self).extra_repr()
        if self.nbits is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}, quan_log={}'.format(
            s_prefix, self.nbits, self.log)


class LinearBPv2(nn.Linear):
    """
        Linear with Bit Pruning
    """

    def __init__(self, in_features, out_features, bias=True,
                 nbits=8, mode=q_modes.layer_wise,
                 complement=False, no_bp=False, log=False):
        super().__init__(in_features, out_features, bias)
        if nbits <= 0:
            self.nbits = None
            return
        self.nbits = nbits
        self.no_bp = no_bp
        self.log = log
        self.complement = complement  # use tows complement representation, otherwise true form
        self.stopGradientMask = None
        self.register_buffer('init_state', torch.zeros(1))
        if mode == q_modes.layer_wise:
            self.register_buffer('radix_position', torch.zeros(1))
        else:
            raise NotImplementedError

    def forward(self, input):
        if self.nbits is None:
            return F.linear(input, self.weight, self.bias)
        if self.init_state == 0:
            self.init_state.data.fill_(1)
            il = torch.log2(torch.max(self.weight.max(), self.weight.min().abs())) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            # quantize weight
            weight_q, weight_int = truncation(self.weight, self.radix_position)
            if self.training:
                self.stopGradientMask = (weight_int > 0).float()

        weight_mask = FunctionStopGradient.apply(self.weight, self.stopGradientMask)
        # STE for quantized weight.
        if not self.no_bp:
            weight_bp = FunctionBitPruningSTE.apply(weight_mask, self.radix_position, self.log)
        else:
            weight_bp = weight_mask
        return F.linear(input, weight_bp, self.bias)

    def extra_repr(self):
        s_prefix = super(LinearBPv2, self).extra_repr()
        if self.nbits is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}, quan_log={}, no_bp: {}'.format(
            s_prefix, self.nbits, self.log, self.no_bp)


class LinearBP(nn.Linear):
    """
        Linear with Bit Pruning
    """

    def __init__(self, in_features, out_features, bias=True,
                 nbits=8, mode=q_modes.layer_wise,
                 expected_bit_sparsity=None, increase_factor=1 / 3, complement=False, no_bp=False):
        super().__init__(in_features, out_features, bias)
        self.expected_bit_sparsity = expected_bit_sparsity
        if nbits <= 0:
            self.nbits = None
            return
        self.nbits = nbits
        self.no_bp = no_bp
        self.complement = complement
        self.stopGradientMask = None
        self.increase_factor = increase_factor
        assert (0 <= increase_factor < 1), 'increase factor ranges in [0, 1)'
        self.expected_bit_sparsity_func = partial(expected_bit_sparsity_func, increase_factor)
        self.register_buffer('init_state', torch.zeros(1))
        if mode == q_modes.layer_wise:
            self.register_buffer('radix_position', torch.zeros(1))
        else:
            raise NotImplementedError
        self.register_buffer('weight_int', torch.zeros(self.weight.shape, dtype=torch.int8))
        self.register_buffer('weight_old', torch.zeros(self.weight.shape))

    def forward(self, input):
        if self.nbits is None:
            return F.linear(input, self.weight, self.bias)
        if self.init_state == 0:
            self.init_state = torch.ones(1)
            il = torch.log2(torch.max(self.weight.max(), self.weight.min().abs())) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position += self.nbits - il
            # quantize weight_fold
            weight_q, weight_int = truncation(self.weight, self.radix_position)
            # quantize weight
            self.weight_int.data.copy_(weight_int)
            self.weight_old.data.copy_(self.weight)
            if self.training:
                if self.expected_bit_sparsity is None:
                    bit_cnt = count_bit(weight_int, self.complement)
                    original_bit_sparsity = bit_sparse(bit_cnt, self.complement)
                    self.expected_bit_sparsity = self.expected_bit_sparsity_func(original_bit_sparsity)
                    print('original: {:.3f} expected: {:.3f}'.format(original_bit_sparsity, self.expected_bit_sparsity))
                self.stopGradientMask = (weight_int > 0).float()
        elif self.no_bp:
            weight_q, weight_int = truncation(self.weight, self.radix_position)
        else:
            # quantize weight
            weight_q, weight_int = truncation(self.weight, self.radix_position)
            if self.training:
                bit_cnt_old = count_bit(self.weight_int).to(self.weight_int.device)
                # bit_sparsity_new = 1 - bit_cnt_new.sum().float() / (8 * weight_int.view(-1).shape[0])
                bit_sparsity_old = bit_sparse(bit_cnt_old, self.complement)
                if bit_sparsity_old < self.expected_bit_sparsity:
                    # need bit pruning
                    bit_cnt_new = count_bit(weight_int).to(self.weight_int.device)
                    bit_increase = bit_cnt_new - bit_cnt_old
                    case = (bit_increase > 0)
                    weight_q = torch.where(case, self.weight_int.float() * 2 ** (-self.radix_position),
                                           weight_q)
                    # don't work
                    self.weight.data.copy_(torch.where(case, self.weight_old, self.weight))
                    self.weight_old.data.copy_(self.weight)
                    self.weight_int.data.copy_(weight_q * 2 ** self.radix_position)
                else:  # don't need bit pruning
                    # print('do not need bit pruning')
                    # use new weights
                    self.weight_old.data.copy_(self.weight)
                    self.weight_int.data.copy_(weight_int)
            else:  # inference
                weight_q = self.weight_int.data.float() * 2 ** (-self.radix_position)
        weight_mask = FunctionStopGradient.apply(self.weight, self.stopGradientMask)
        # STE for quantized weight.
        weight_bp = weight_q.detach() + weight_mask - weight_q.detach()
        return F.linear(input, weight_bp, self.bias)

    def extra_repr(self):
        s_prefix = super(LinearBP, self).extra_repr()
        if self.nbits is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}, increase_factor:{}, no_bp: {}'.format(
            s_prefix, self.nbits, self.increase_factor, self.no_bp)


class Conv2dBP(nn.Conv2d):
    """
        Conv2d with Bit Pruning
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 nbits=8, mode=q_modes.layer_wise,
                 expected_bit_sparsity=None, increase_factor=1 / 3, complement=False):
        super(Conv2dBP, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.expected_bit_sparsity = expected_bit_sparsity
        if nbits <= 0:
            self.nbits = None
            return
        self.nbits = nbits
        self.complement = complement  # use tows complement representation, otherwise true form
        self.stopGradientMask = None
        self.increase_factor = increase_factor
        assert (0 <= increase_factor < 1), 'increase factor ranges in [0, 1)'
        self.expected_bit_sparsity_func = partial(expected_bit_sparsity_func, increase_factor)
        self.register_buffer('init_state', torch.zeros(1))
        if mode == q_modes.layer_wise:
            self.register_buffer('radix_position', torch.zeros(1))
        else:
            raise NotImplementedError
        self.register_buffer('weight_int', torch.zeros(self.weight.shape, dtype=torch.int8))
        self.register_buffer('weight_old', torch.zeros(self.weight.shape))

    def forward(self, input):
        if self.nbits is None:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = weight_fold.reshape([weight_fold.shape[0], -1]).transpose(0, 1)
        if self.init_state == 0:  # need to set radix position
            # set radix position
            self.init_state.data.fill_(1)
            il = torch.log2(torch.max(self.weight.max(), self.weight.min().abs())) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            # quantize weight_fold
            weight_q, weight_int = truncation(self.weight, self.radix_position)
            self.weight_int.data.copy_(weight_int)
            self.weight_old.data.copy_(self.weight)
            if self.training:
                if self.expected_bit_sparsity is None:
                    bit_cnt = count_bit(weight_int, self.complement)
                    original_bit_sparsity = bit_sparse(bit_cnt, self.complement)
                    self.expected_bit_sparsity = self.expected_bit_sparsity_func(original_bit_sparsity)
                    print('original: {:.3f} expected: {:.3f}'.format(original_bit_sparsity, self.expected_bit_sparsity))
                self.stopGradientMask = (weight_int > 0).float()
        else:
            # quantize weight
            weight_q, weight_int = truncation(self.weight, self.radix_position)
            if self.training:
                bit_cnt_old = count_bit(self.weight_int, self.complement)
                # bit_sparsity_new = bit_sparse(bit_cnt_new, self.complement)
                bit_sparsity_old = bit_sparse(bit_cnt_old, self.complement)
                if bit_sparsity_old < self.expected_bit_sparsity:
                    # need bit pruning
                    bit_cnt_new = count_bit(weight_int, self.complement)
                    bit_increase = bit_cnt_new - bit_cnt_old
                    case = (bit_increase > 0)
                    weight_q = torch.where(case, self.weight_int.float() * 2 ** (-self.radix_position),
                                           weight_q)
                    # don't work
                    self.weight.data.copy_(torch.where(case, self.weight_old, self.weight))
                    self.weight_old.data.copy_(self.weight)
                    self.weight_int.data.copy_(weight_q * 2 ** self.radix_position)
                else:  # don't need bit pruning
                    # print('do not need bit pruning')
                    # use new weights
                    self.weight_old.data.copy_(self.weight)
                    self.weight_int.data.copy_(weight_int)
            else:  # inference
                weight_q = self.weight_int.data.float() * 2 ** (-self.radix_position)
        weight_mask = FunctionStopGradient.apply(self.weight, self.stopGradientMask)
        # STE for quantized weight.
        weight_bp = weight_q.detach() + weight_mask - self.weight.detach()
        out = F.conv2d(input, weight_bp, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def extra_repr(self):
        s_prefix = super(Conv2dBP, self).extra_repr()
        if self.nbits is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}, increase_factor:{}'.format(
            s_prefix, self.nbits, self.increase_factor)


class Conv2dBNBP(nn.Conv2d):
    """
    quantize weights after fold BN to conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 nbits=8, mode=q_modes.layer_wise,
                 expected_bit_sparsity=None, increase_factor=1 / 3, complement=False):
        super(Conv2dBNBP, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)
        self._bn = nn.BatchNorm2d(out_channels, eps, momentum, affine, track_running_stats)
        self.expected_bit_sparsity = expected_bit_sparsity
        if nbits <= 0:
            self.nbits = None
            self.register_buffer('init_state', None)
            self.register_buffer('radix_position', None)
            return
        self.nbits = nbits
        self.complement = complement
        self.stopGradientMask = None
        self.increase_factor = increase_factor
        assert (0 <= increase_factor < 1), 'increase factor ranges in [0, 1)'
        self.expected_bit_sparsity_func = partial(expected_bit_sparsity_func, increase_factor)
        self.register_buffer('init_state', torch.zeros(1))
        if mode == q_modes.layer_wise:
            self.register_buffer('radix_position', torch.zeros(1))
        else:
            raise NotImplementedError
        self.register_buffer('weight_fold_int', torch.zeros(self.weight.shape, dtype=torch.int8))
        self.register_buffer('weight_old', torch.zeros(self.weight.shape))

    def forward(self, input):
        if self._bn.training:
            print('Please Freeze BN when using bit pruning')
            raise PermissionError
            # conv_out = F.conv2d(input, self.weight, self.bias, self.stride,
            #                     self.padding, self.dilation, self.groups)
            # # calculate mean and various
            # # fake_out = self._bn(conv_out)
            # conv_out = conv_out.transpose(1, 0).contiguous()
            # conv_out = conv_out.view(conv_out.size(0), -1)
            # mu = conv_out.mean(dim=1)  # it is the same as mean calculated in _bn.
            # var = torch.var(conv_out, dim=1, unbiased=False)
        else:  # inference
            mu = self._bn.running_mean
            var = self._bn.running_var
        if self._bn.affine:
            gamma = self._bn.weight
            beta = self._bn.bias
        else:
            gamma = torch.ones(self.out_channels).to(var.device)
            beta = torch.zeros(self.out_channels).to(var.device)

        A = gamma.div(torch.sqrt(var + self._bn.eps))
        A_expand = A.expand_as(self.weight.transpose(0, -1)).transpose(0, -1)

        weight_fold = self.weight * A_expand
        if self.bias is None:
            bias_fold = (- mu) * A + beta
        else:
            bias_fold = (self.bias - mu) * A + beta
        if self.nbits is None:
            return F.conv2d(input, weight_fold, bias_fold, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = weight_fold.reshape([weight_fold.shape[0], -1]).transpose(0, 1)
        update = False
        if self.init_state == 0:  # need to set radix position
            # set radix position
            self.init_state.data.fill_(1)
            il = torch.log2(torch.max(weight_fold.max(), weight_fold.min().abs())) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position += self.nbits - il
            # quantize weight_fold
            weight_fold_q, weight_fold_int = truncation(weight_fold, self.radix_position)
            self.weight_fold_int.data.copy_(weight_fold_int)
            self.weight_old.data.copy_(self.weight)
            if self.training:
                if self.expected_bit_sparsity is None:
                    bit_cnt = count_bit(weight_fold_int, self.complement)
                    original_bit_sparsity = bit_sparse(bit_cnt, self.complement)
                    self.expected_bit_sparsity = self.expected_bit_sparsity_func(original_bit_sparsity)
                    print('original: {:.3f} expected: {:.3f}'.format(original_bit_sparsity, self.expected_bit_sparsity))
                self.stopGradientMask = (weight_fold_int > 0).float()
        else:
            # quantize weight_fold
            weight_fold_q, weight_fold_int = truncation(weight_fold, self.radix_position)
            if self.training:
                bit_cnt_old = count_bit(self.weight_fold_int, self.complement)
                # bit_sparsity_new = bit_sparse(bit_cnt_new, self.complement)
                bit_sparsity_old = bit_sparse(bit_cnt_old, self.complement)
                if bit_sparsity_old < self.expected_bit_sparsity:
                    # need bit pruning
                    bit_cnt_new = count_bit(weight_fold_int, self.complement)
                    bit_increase = bit_cnt_new - bit_cnt_old
                    case = (bit_increase > 0)
                    weight_fold_q = torch.where(case, self.weight_fold_int.float() * 2 ** (-self.radix_position),
                                                weight_fold_q)
                    # don't work
                    self.weight.data.copy_(torch.where(case, self.weight_old, self.weight))
                    self.weight_old.data.copy_(self.weight)
                    self.weight_fold_int.data.copy_(weight_fold_q * 2 ** self.radix_position)
                    update = True
                else:  # don't need bit pruning
                    # print('do not need bit pruning')
                    # use old weights
                    # weight_fold_q = self.weight_int.data.float() * 2 ** (-self.radix_position)
                    # self.weight.data.copy_(self.weight_old)
                    # use new weights
                    self.weight_old.data.copy_(self.weight)
                    self.weight_fold_int.data.copy_(weight_fold_int)
            else:  # inference
                weight_fold_q = self.weight_fold_int.data.float() * 2 ** (-self.radix_position)
        if False and self.training and update:  # keep weight_fold 
            if self._bn.training:
                print('Please Freeze BN when using bit pruning')
                raise PermissionError
                # new_weight
                # conv_out = F.conv2d(input, self.weight, self.bias, self.stride,
                #                     self.padding, self.dilation, self.groups)
                # # calculate mean and various
                # fake_out = self._bn(conv_out)
                # conv_out = conv_out.transpose(1, 0).contiguous()
                # conv_out = conv_out.view(conv_out.size(0), -1)
                # mu = conv_out.mean(dim=1)  # it is the same as mean calculated in _bn.
                # var = torch.var(conv_out, dim=1, unbiased=False)
            else:  # inference
                mu = self._bn.running_mean
                var = self._bn.running_var
            if self._bn.affine:
                gamma = self._bn.weight
                beta = self._bn.bias
            else:
                gamma = torch.ones(self.out_channels).to(var.device)
                beta = torch.zeros(self.out_channels).to(var.device)

            A = gamma.div(torch.sqrt(var + self._bn.eps))
            A_expand = A.expand_as(self.weight.transpose(0, -1)).transpose(0, -1)
            weight_fold = self.weight * A_expand
            if self.bias is None:
                bias_fold = (- mu) * A + beta
            else:
                bias_fold = (self.bias - mu) * A + beta
        # STE for quantized weight.
        weight_fold_mask = FunctionStopGradient.apply(weight_fold, self.stopGradientMask)
        weight_fold_bp = weight_fold_q.detach() + weight_fold_mask - weight_fold_mask.detach()
        out = F.conv2d(input, weight_fold_bp, bias_fold, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def extra_repr(self):
        s_prefix = super(Conv2dBNBP, self).extra_repr()
        if self.nbits is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}, increase_factor:{}'.format(
            s_prefix, self.nbits, self.increase_factor)


def count_bit(w_int, complement=False):
    if complement:
        w_int = torch.where(w_int < 0, 256 + w_int, w_int).int().to(w_int.device)
        bit_cnt = torch.zeros(w_int.shape).int().to(w_int.device)
        for i in range(8):
            bit_cnt += w_int % 2
            w_int /= 2
    else:
        w_int = torch.abs(w_int.float()).int()
        bit_cnt = torch.zeros(w_int.shape).int().to(w_int.device)
        for i in range(8):
            bit_cnt += w_int % 2
            w_int /= 2
    return bit_cnt


def bit_sparse(bit_cnt, complement=False):
    if complement:
        return 1 - bit_cnt.sum().float() / (8 * bit_cnt.view(-1).shape[0])
    else:
        return 1 - bit_cnt.sum().float() / (7 * bit_cnt.view(-1).shape[0])


class Conv2dBN(nn.Conv2d):
    """
    quantize weights after fold BN to conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(Conv2dBN, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self._bn = nn.BatchNorm2d(out_channels, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        if self._bn.training:
            conv_out = F.conv2d(input, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
            # calculate mean and various
            fake_out = self._bn(conv_out)
            conv_out = conv_out.transpose(1, 0).contiguous()
            conv_out = conv_out.view(conv_out.size(0), -1)
            mu = conv_out.mean(dim=1)  # it is the same as mean calculated in _bn.
            var = torch.var(conv_out, dim=1, unbiased=False)
        else:
            mu = self._bn.running_mean
            var = self._bn.running_var
        if self._bn.affine:
            gamma = self._bn.weight
            beta = self._bn.bias
        else:
            gamma = torch.ones(self.out_channels).to(var.device)
            beta = torch.zeros(self.out_channels).to(var.device)

        A = gamma.div(torch.sqrt(var + self._bn.eps))
        A_expand = A.expand_as(self.weight.transpose(0, -1)).transpose(0, -1)

        weight_fold = self.weight * A_expand
        if self.bias is None:
            bias_fold = (- mu) * A + beta
        else:
            bias_fold = (self.bias - mu) * A + beta
        out = F.conv2d(input, weight_fold, bias_fold, self.stride,
                       self.padding, self.dilation, self.groups)
        return out


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def truncation(fp_data, radix_position, nbits=8):
    scale_factor = 2 ** radix_position
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data_int = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data_int, scale_factor)
    return q_data, q_data_int
