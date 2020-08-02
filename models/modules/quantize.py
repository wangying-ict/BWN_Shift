from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
import math
from torch.nn.modules.dropout import _DropoutNd
# from .config import config

import ipdb

__all__ = ['ActQ', 'LinearQ', 'Conv2dQ', 'q_modes', 'PACT', 'Conv2dQv2', 'ActQv2', 'LinearQv2', 'DropoutScale']


class q_modes(Enum):
    layer_wise = 1
    kernel_wise = 2


class Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=q_modes.kernel_wise, l2=True, scale_bits=-1, bias_bits=-1, ema_decay=0.99):
        """
        Quantize Conv2d
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param nbits:  weights' bits
        :param mode:
        :param l2:
        :param scale_bits: scale bits
        :param bias_bits:  bias bits(need upper actq's scale)
        """
        super(Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)
        if nbits < 0:
            self.register_buffer('running_scale', None)
            return
        self.nbits = nbits
        self.q_model = mode
        self.l2 = l2
        self.scale_bits = scale_bits
        self.bias_bits = bias_bits
        self.ema_decay = ema_decay
        if mode == q_modes.kernel_wise:
            self.register_buffer('running_scale', torch.zeros(out_channels))
            self.is_layer_wise = False
        else:
            self.register_buffer('running_scale', torch.zeros(1))
            self.is_layer_wise = True
        self.register_buffer('init_state', torch.zeros(1))
        self.reset_running_stats()

    def set_scale_bits(self, nbits=8):
        self.scale_bits = nbits

    def set_bias_bits(self, nbits=8):
        self.bias_bits = nbits

    def reset_running_stats(self):
        self.running_scale.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            # init running scale
            self.running_scale.data.copy_(w_reshape.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        scale = self.running_scale.detach()
        if self.scale_bits > 0:
            scale, _ = truncation(scale, self.scale_bits)
        if self.bias_bits > 0:
            input, scale_a = input[0], input[1]
            scale_bias = scale * scale_a
            if self.scale_bits > 0:
                scale_bias, _ = truncation(scale_bias, self.scale_bits)
            # quantize bias b_q = round(b / (scale_a * scale))
            bias_clip = (self.bias / scale_bias).clamp(- 2 ** (self.bias_bits - 1),
                                                       2 ** (self.bias_bits - 1) - 1)
            bias_q = bias_clip.round()
            bias_q = bias_q * scale_bias
            bq = bias_q.detach() + self.bias - self.bias.detach()
        else:
            bq = self.bias
        error, _, y = ln_error(w_reshape, self.nbits, scale, is_act=self.is_layer_wise, l2=self.l2)
        if self.training:
            with torch.no_grad():
                b, s = update_running_scale(w_reshape, self.nbits, scale, error, self.is_layer_wise, l2=self.l2)
                self.running_scale = torch.where(b, scale * self.ema_decay + (1 - self.ema_decay) * scale * 2, scale)
                self.running_scale = torch.where(s, scale * self.ema_decay + (1 - self.ema_decay) * scale / 2,
                                                 self.running_scale)

        wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(input, wq, bq, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s_prefix = super(Conv2dQ, self).extra_repr()
        if self.running_scale is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits=({},bias:{},scale:{}), qmode=({}, {})'.format(
            s_prefix, self.nbits, self.bias_bits,
            self.scale_bits, self.q_model,
            'l2' if self.l2 else 'l1')


class Conv2dQv2(nn.Conv2d):
    """
        Update running scale using simulated grad.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=q_modes.kernel_wise, l2=True, scale_bits=-1, bias_bits=-1):

        super(Conv2dQv2, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        if nbits < 0:
            self.register_parameter('running_scale', None)
            return
        self.nbits = nbits
        self.q_model = mode
        self.l2 = l2
        self.scale_bits = scale_bits
        self.bias_bits = bias_bits
        # self.ema_decay = scale_decay
        if mode == q_modes.kernel_wise:
            self.running_scale = nn.Parameter(torch.Tensor(out_channels))
            self.is_layer_wise = False
        else:
            self.running_scale = nn.Parameter(torch.Tensor(1))
            self.is_layer_wise = True
        self.register_buffer('init_state', torch.zeros(1))
        self.reset_running_stats()

    def set_scale_bits(self, nbits=8):
        self.scale_bits = nbits

    def set_bias_bits(self, nbits=8):
        self.bias_bits = nbits

    def reset_running_stats(self):
        self.running_scale.data.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            # init running scale
            if self.is_layer_wise:
                self.running_scale.data.copy_(w_reshape.detach().abs().max() / 2 ** (self.nbits - 1))
            else:
                self.running_scale.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        if self.training and self.running_scale.grad is None:
            loss = self.running_scale.sum()
            loss.backward()
            self.running_scale.grad.zero_()
        # self.running_scale.data.abs_()
        scale = self.running_scale.detach()
        if self.scale_bits > 0:
            scale, _ = truncation(scale, self.scale_bits)
        if self.bias_bits > 0:
            input, scale_a = input[0], input[1]
            scale_bias = scale * scale_a
            if self.scale_bits > 0:
                scale_bias, _ = truncation(scale_bias, self.scale_bits)
            # quantize bias b_q = round(b / (scale_a * scale))
            bias_clip = (self.bias / scale_bias).clamp(- 2 ** (self.bias_bits - 1),
                                                       2 ** (self.bias_bits - 1) - 1)
            bias_q = bias_clip.round()
            bias_q = bias_q * scale_bias
            bq = bias_q.detach() + self.bias - self.bias.detach()
        else:
            bq = self.bias
        error, _, y = ln_error(w_reshape, self.nbits, scale, is_act=self.is_layer_wise, l2=self.l2)
        if self.training and self.running_scale.grad is not None:
            with torch.no_grad():
                b, s = update_running_scale(w_reshape, self.nbits, scale, error, self.is_layer_wise, l2=self.l2)
                self.running_scale.grad.data.add_(
                    torch.where(b, -1 * (self.running_scale ** 2),
                                torch.zeros_like(self.running_scale)))
                self.running_scale.grad.data.add_(
                    torch.where(s, self.running_scale ** 2,
                                torch.zeros_like(self.running_scale)))

        wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(input, wq, bq, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s_prefix = super(Conv2dQv2, self).extra_repr()
        if self.running_scale is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits=({},bias:{},scale:{}), qmode=({}, {})'.format(
            s_prefix, self.nbits, self.bias_bits,
            self.scale_bits, self.q_model,
            'l2' if self.l2 else 'l1')


class LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits=4, mode=q_modes.layer_wise, l2=True,
                 scale_bits=-1, bias_bits=-1, ema_decay=0.9):
        """
        Quantize Linear
        :param in_features:
        :param out_features:
        :param bias:
        :param nbits:
        :param mode:
        :param l2:
        :param scale_bits:
        :param bias_bits: bias bits(need upper actq's scale)
        """
        super(LinearQ, self).__init__(in_features, out_features, bias=bias)
        if nbits < 0:
            self.register_buffer('running_scale', None)
            return
        self.nbits = nbits
        self.q_mode = mode
        self.l2 = l2
        self.scale_bits = scale_bits
        self.bias_bits = bias_bits
        self.ema_decay = ema_decay
        if mode == q_modes.kernel_wise:
            self.register_buffer('running_scale', torch.zeros(out_features))
            self.is_layer_wise = False
        else:
            self.register_buffer('running_scale', torch.zeros(1))
            self.is_layer_wise = True
        self.reset_running_stats()

    def set_scale_bits(self, nbits=8):
        self.scale_bits = nbits

    def set_bias_bits(self, nbits=8):
        self.bias_bits = nbits

    def reset_running_stats(self):
        self.running_scale.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return F.linear(input, self.weight, self.bias)
        w_reshape = self.weight.transpose(0, 1)
        scale = self.running_scale.detach()

        if self.scale_bits > 0:
            scale, _ = truncation(scale, self.scale_bits)
        if self.bias_bits > 0:
            input, scale_a = input[0], input[1]
            scale_bias = scale * scale_a
            if self.scale_bits > 0:
                scale_bias, _ = truncation(scale_bias, self.scale_bits)
            # quantize bias b_q = round(b / (scale_a * scale))
            bias_clip = (self.bias / scale_bias).clamp(- 2 ** (self.bias_bits - 1),
                                                       2 ** (self.bias_bits - 1) - 1)
            bias_q = bias_clip.round()
            bias_q = bias_q * scale_bias
            bq = bias_q.detach() + self.bias - self.bias.detach()
        else:
            bq = self.bias

        error, _, y = ln_error(w_reshape, self.nbits, scale, is_act=self.is_layer_wise, l2=self.l2)
        if self.training:
            with torch.no_grad():
                b, s = update_running_scale(w_reshape, self.nbits, scale, error, self.is_layer_wise, l2=self.l2)
                self.running_scale = torch.where(b, scale * self.ema_decay + (1 - self.ema_decay) * scale * 2, scale)
                self.running_scale = torch.where(s, scale * self.ema_decay + (1 - self.ema_decay) * scale / 2,
                                                 self.running_scale)
        wq = y.transpose(0, 1).detach() + self.weight - self.weight.detach()
        return F.linear(input, wq, bq)

    def extra_repr(self):
        s_prefix = super(LinearQ, self).extra_repr()
        if self.running_scale is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits=({},bias:{},scale:{}), mode=({}, {})'.format(
            s_prefix, self.nbits, self.bias_bits,
            self.scale_bits, self.q_mode,
            'l2' if self.l2 else 'l1')


class LinearQv2(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits=4, mode=q_modes.layer_wise, l2=True,
                 scale_bits=-1, bias_bits=-1):

        super(LinearQv2, self).__init__(in_features, out_features, bias=bias)
        if nbits < 0:
            self.register_parameter('running_scale', None)
            return
        self.nbits = nbits
        self.q_mode = mode
        self.l2 = l2
        self.scale_bits = scale_bits
        self.bias_bits = bias_bits
        if mode == q_modes.kernel_wise:
            self.running_scale = nn.Parameter(torch.Tensor(out_features))
            self.is_layer_wise = False
        else:
            self.running_scale = nn.Parameter(torch.Tensor(1))
            self.is_layer_wise = True
        self.register_buffer('init_state', torch.zeros(1))
        self.reset_running_stats()

    def set_scale_bits(self, nbits=8):
        self.scale_bits = nbits

    def set_bias_bits(self, nbits=8):
        self.bias_bits = nbits

    def reset_running_stats(self):
        self.running_scale.data.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return F.linear(input, self.weight, self.bias)
        w_reshape = self.weight.transpose(0, 1)
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            self.running_scale.data.fill_(w_reshape.detach().abs().max() / 2 ** (self.nbits - 1))
        if self.training and self.running_scale.grad is None:
            loss = self.running_scale.sum()
            loss.backward()
            self.running_scale.grad.zero_()
        scale = self.running_scale.detach()

        if self.scale_bits > 0:
            scale, _ = truncation(scale, self.scale_bits)
        if self.bias_bits > 0:
            input, scale_a = input[0], input[1]
            scale_bias = scale * scale_a
            if self.scale_bits > 0:
                scale_bias, _ = truncation(scale_bias, self.scale_bits)
            # quantize bias b_q = round(b / (scale_a * scale))
            bias_clip = (self.bias / scale_bias).clamp(- 2 ** (self.bias_bits - 1),
                                                       2 ** (self.bias_bits - 1) - 1)
            bias_q = bias_clip.round()
            bias_q = bias_q * scale_bias
            bq = bias_q.detach() + self.bias - self.bias.detach()
        else:
            bq = self.bias

        error, _, y = ln_error(w_reshape, self.nbits, scale, is_act=self.is_layer_wise, l2=self.l2)
        if self.training and self.running_scale.grad is not None:
            with torch.no_grad():
                b, s = update_running_scale(w_reshape, self.nbits, scale, error, self.is_layer_wise, l2=self.l2)
                self.running_scale.grad.data.add_(
                    torch.where(b, -(self.running_scale ** 2), torch.zeros_like(self.running_scale)))
                self.running_scale.grad.data.add_(
                    torch.where(s, self.running_scale ** 2, torch.zeros_like(self.running_scale)))
        wq = y.transpose(0, 1).detach() + self.weight - self.weight.detach()
        return F.linear(input, wq, bq)

    def extra_repr(self):
        s_prefix = super(LinearQv2, self).extra_repr()
        if self.running_scale is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits=({},bias:{},scale:{}), mode=({}, {})'.format(
            s_prefix, self.nbits, self.bias_bits,
            self.scale_bits, self.q_mode,
            'l2' if self.l2 else 'l1')


class PACT(Module):
    # TODO: signed
    def __init__(self, nbits=4, signed=False, inplace=False):
        super(PACT, self).__init__()
        if nbits < 0:
            self.register_parameter('clip_value', None)
            return
        if signed:
            raise NotImplementedError
        self.nbits = nbits
        self.signed = signed
        self.inplace = inplace
        self.clip_value = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        if self.clip_value is None:
            return input
        if self.init_state == 0:
            self.init_state = torch.ones(1)
            self.clip_value.data.copy_(input.max())

        input = LearnedClippedLinearQuantizeSTE.apply(input, self.clip_value, self.nbits, True, self.inplace)
        return input

    def extra_repr(self):
        if self.clip_value is None:
            return 'fake'
        return 'signed={}, nbits=({}'.format(self.signed, self.nbits)


class LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val)
        if inplace:
            ctx.mark_dirty(input)
        scale_factor = asymmetric_linear_quantization_scale_factor(num_bits, 0, clip_val.data[0])
        output = clamp(input, 0, clip_val.data[0], inplace)
        output = linear_quantize(output, scale_factor, inplace)
        if dequantize:
            output = linear_dequantize(output, scale_factor, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(0)] = 0
        grad_input[input.ge(clip_val.data[0])] = 0

        grad_alpha = grad_output.clone()
        grad_alpha[input.lt(clip_val.data[0])] = 0
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        # Straight-through estimator for the scale factor calculation
        return grad_input, grad_alpha, None, None, None


def asymmetric_linear_quantization_scale_factor(num_bits, saturation_min, saturation_max):
    n = 2 ** num_bits - 1
    return n / (saturation_max - saturation_min)


class ActQv2(Module):
    def __init__(self, nbits=4, signed=False, l2=True, expand=False, split=False,
                 scale_bits=-1, out_scale=False):
        """
        Quantize activation
        :param nbits:
        :param signed:
        :param l2:     use l2 to optimize running scale
        :param expand:
        :param split:
        :param scale_bits: if scale bits > 0, then use qcode method to quantize scale
        :param out_scale: output = [output, scale]
        """
        # we can expand 8bit to high4 and low4
        super(ActQv2, self).__init__()
        if nbits < 0:
            self.register_parameter('running_scale', None)
            return
        self.nbits = nbits
        self.signed = signed
        self.expand = expand
        self.split = split
        self.l2 = l2
        self.scale_bits = scale_bits
        self.out_scale = out_scale
        if not signed:
            # We use signed to represent unsigned numbers.
            # e.g. int5 == uint4 when ActFun==ReLU
            self.nbits = nbits + 1
        self.running_scale = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.reset_running_stats()

    def set_scale_bits(self, nbits=8):
        self.scale_bits = nbits

    def set_out_scale(self, out_scale=True):
        self.out_scale = out_scale

    def reset_running_stats(self):
        self.running_scale.data.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return input
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            self.running_scale.data.fill_(input.detach().abs().max() / 2 ** (self.nbits - 1))
        if self.training and self.running_scale.grad is None:
            loss = self.running_scale.sum()
            loss.backward()
            self.running_scale.grad.zero_()
        scale = self.running_scale.detach()
        if self.scale_bits > 0:
            scale, _ = truncation(scale, nbits=self.scale_bits)
        error, x_clip, y = ln_error(input, self.nbits, scale, is_act=True, l2=self.l2)
        if self.training and self.running_scale.grad is not None:
            with torch.no_grad():
                b, s = update_running_scale(input, self.nbits, scale, error, is_act=True, l2=self.l2)
                self.running_scale.grad.data.add_(
                    torch.where(b, -(self.running_scale ** 2),
                                torch.zeros_like(self.running_scale)))
                self.running_scale.grad.data.add_(
                    torch.where(s, self.running_scale ** 2,
                                torch.zeros_like(self.running_scale)))

        output = y.detach() + x_clip - x_clip.detach()
        if self.expand is False and self.split is False:
            return [output, scale] if self.out_scale else output
        assert (self.expand and self.split) is False, \
            'The two parameters (expand, split) cannot be true at the same time.'
        if self.expand:
            if not self.signed:
                assert self.nbits == 9, 'Only support uint8 or int7'
                # output has already been rounded and clipped, so we just need split the input.
                wide_int = output / scale
                high_float = (wide_int / 2 ** 4)
                high_floor = high_float.floor()

                low_output = (wide_int - (high_floor.detach() + high_float - high_float.detach()) * 16) * scale

                high_float = high_float * scale
                high_floor = high_floor * scale
                high_output = high_float - high_float.detach() + high_floor.detach()

                low_high = torch.cat((low_output, high_output), dim=1)
                return low_high
            else:
                # TODO: quantize data to int7(int4+int4)
                assert self.nbits == 7, 'Only support int7 or uint8'
                assert NotImplementedError
        elif self.split:
            if not self.signed:
                assert self.nbits == 9, 'Only support uint8 or int7'
                # output has already been rounded and clipped, so we just need split the input.
                wide_int = output / scale
                high_float = (wide_int / 2 ** 4)
                high_floor = high_float.floor()

                low_output = (wide_int - (high_floor.detach() + high_float - high_float.detach()) * 16) * scale

                high_float = high_float * scale * 2 ** 4
                high_floor = high_floor * scale * 2 ** 4
                high_output = high_float - high_float.detach() + high_floor.detach()

                return high_output, low_output
            else:
                assert self.nbits == 7, 'Only support int7 or uint8'
                assert NotImplementedError

    def extra_repr(self):
        if self.running_scale is None:
            return 'fake'
        bit = self.nbits if self.signed else self.nbits - 1
        return 'signed={}, nbits=({}, scale:{}, out_scale:{}), mode={}, expand={}, split={}'.format(
            self.signed, bit,
            self.scale_bits,
            self.out_scale,
            'l2' if self.l2 else 'l1',
            self.expand,
            self.split)


class DropoutScale(_DropoutNd):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. The elements to zero are randomized on every forward call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input):
        if len(input) == 2:
            return [F.dropout(input[0], self.p, self.training, self.inplace), input[1]]
        return F.dropout(input, self.p, self.training, self.inplace)


class ActQ(Module):
    def __init__(self, nbits=4, signed=False, l2=True, expand=False, split=False,
                 scale_bits=-1, out_scale=False, ema_decay=0.999):
        """
        Quantize activation
        :param nbits:
        :param signed:
        :param l2:     use l2 to optimize running scale
        :param expand:
        :param split:
        :param scale_bits: if scale bits > 0, then use qcode method to quantize scale
        :param out_scale: output = [output, scale]
        """
        # we can expand 8bit to high4 and low4
        super(ActQ, self).__init__()
        if nbits < 0:
            self.register_buffer('running_scale', None)
            return
        self.nbits = nbits
        self.signed = signed
        self.expand = expand
        self.split = split
        self.l2 = l2
        self.scale_bits = scale_bits
        self.out_scale = out_scale
        self.ema_decay = ema_decay
        if not signed:
            # We use signed to represent unsigned numbers.
            # e.g. int5 == uint4 when ActFun==ReLU
            self.nbits = nbits + 1
        self.register_buffer('running_scale', torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.reset_running_stats()

    def set_scale_bits(self, nbits=8):
        self.scale_bits = nbits

    def set_out_scale(self, out_scale=True):
        self.out_scale = out_scale

    def reset_running_stats(self):
        self.running_scale.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return input
        if self.training and self.init_state == 0:
            # init running scale
            if self.signed:
                self.running_scale.data.copy_(input.max() / 2 ** (self.nbits - 1))
            else:
                self.running_scale.data.copy_(input.max() / 2 ** self.nbits)
            self.init_state.fill_(1)
        scale = self.running_scale.detach()
        if self.scale_bits > 0:
            scale, _ = truncation(scale, nbits=self.scale_bits)
        error, x_clip, y = ln_error(input, self.nbits, scale, is_act=True, l2=self.l2)
        if self.training:
            with torch.no_grad():
                b, s = update_running_scale(input, self.nbits, scale, error, is_act=True, l2=self.l2)
                self.running_scale = torch.where(b, scale * self.ema_decay + (1 - self.ema_decay) * scale * 2, scale)
                self.running_scale = torch.where(s, scale * self.ema_decay + (1 - self.ema_decay) * scale / 2,
                                                 self.running_scale)
        output = y.detach() + x_clip - x_clip.detach()
        if self.expand is False and self.split is False:
            return [output, scale] if self.out_scale else output
        assert (self.expand and self.split) is False, \
            'The two parameters (expand, split) cannot be true at the same time.'
        if self.expand:
            if not self.signed:
                assert self.nbits == 9, 'Only support uint8 or int7'
                # output has already been rounded and clipped, so we just need split the input.
                wide_int = output / scale
                high_float = (wide_int / 2 ** 4)
                high_floor = high_float.floor()

                low_output = (wide_int - (high_floor.detach() + high_float - high_float.detach()) * 16) * scale

                high_float = high_float * scale
                high_floor = high_floor * scale
                high_output = high_float - high_float.detach() + high_floor.detach()

                low_high = torch.cat((low_output, high_output), dim=1)
                return low_high
            else:
                # TODO: quantize data to int7(int4+int4)
                assert self.nbits == 7, 'Only support int7 or uint8'
                assert NotImplementedError
        elif self.split:
            if not self.signed:
                assert self.nbits == 9, 'Only support uint8 or int7'
                # output has already been rounded and clipped, so we just need split the input.
                wide_int = output / scale
                high_float = (wide_int / 2 ** 4)
                high_floor = high_float.floor()

                low_output = (wide_int - (high_floor.detach() + high_float - high_float.detach()) * 16) * scale

                high_float = high_float * scale * 2 ** 4
                high_floor = high_floor * scale * 2 ** 4
                high_output = high_float - high_float.detach() + high_floor.detach()

                return high_output, low_output
            else:
                assert self.nbits == 7, 'Only support int7 or uint8'
                assert NotImplementedError

    def extra_repr(self):
        if self.running_scale is None:
            return 'fake'
        bit = self.nbits if self.signed else self.nbits - 1
        return 'signed={}, nbits=({}, scale:{}, out_scale:{}), mode={}, expand={}, split={}'.format(
            self.signed, bit,
            self.scale_bits,
            self.out_scale,
            'l2' if self.l2 else 'l1',
            self.expand,
            self.split)


def update_running_scale(data_fp, nbits, scale_old, error, is_act, l2=True):
    s_error, _, _ = ln_error(data_fp, nbits, scale_old / 2, is_act=is_act,
                             l2=l2)
    b_error, _, _ = ln_error(data_fp, nbits, scale_old * 2, is_act=is_act,
                             l2=l2)
    a1 = error - s_error
    a2 = b_error - error
    g1 = a1 >= 0
    g2 = a2 > 0
    g3 = a1 + a2 >= 0
    """
                    g1  g2  g3  res
                    0   0   0   big
                    0   0   1   big
                    0   1   0   keep
                    0   1   1   keep
                    1   0   0   big
                    1   0   1   small
                    1   1   0   small
                    1   1   1   small
    """
    b = ((g1 == 0) * (g2 == 0) == 1) + ((g1 * (g2 == 0) * (g3 == 0)) > 0) > 0
    s = (((g1 * g2) > 0) + ((g1 * (g2 == 0) * g3) > 0)) > 0
    return b, s


def ln_error(x, nbits, scale, is_act, l2=True):
    x_clip = (x / scale).clamp(- 2 ** (nbits - 1), 2 ** (nbits - 1) - 1)
    x_q = x_clip.round()
    x_q = x_q * scale
    if is_act:
        if l2:
            error = ((x - x_q) ** 2).sum() / x.reshape(-1).size()[0]
        else:
            error = (x - x_q).abs().sum() / x.reshape(-1).size()[0]
    else:
        if l2:
            error = ((x - x_q) ** 2).sum(dim=0) / x.shape[0]
        else:
            error = (x - x_q).abs().sum(dim=0) / x.shape[0]
    x_clip = x_clip * scale
    return error, x_clip, x_q


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


def truncation(fp_data, nbits=8):
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)
    qcode = nbits - il
    scale_factor = 2 ** qcode
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode
