import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
#from models.modules import q_modes

__all__ = ['Conv2dBWN', 'LinearBWN','Conv2dBWN_Shift', 'LinearBWN_Shift', 'q_modes']
beta = -1e-5
class q_modes(Enum):
    layer_wise = 1
    kernel_wise = 2

#want_uneven = 0.3
class Conv2dBWN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, mode=q_modes.layer_wise):
        super(Conv2dBWN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.q_mode = mode
        if self.q_mode is q_modes.kernel_wise:
            raise NotImplementedError
        # self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        alpha = torch.mean(torch.abs(self.weight.data))
        #alpha = 1.0
        pre_quantized_weight = self.weight / alpha
        quantized_weight = alpha * Function_sign.apply(pre_quantized_weight)
        #print(alpha)
        return F.conv2d(x, quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearBWN(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearBWN, self).__init__(in_features, out_features, bias=bias)
        # self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        alpha = torch.mean(torch.abs(self.weight.data))
        #alpha = 1.0
        pre_quantized_weight = self.weight / alpha
        quantized_weight = alpha * Function_sign.apply(pre_quantized_weight)
        return F.linear(x, quantized_weight, self.bias)

class Function_sign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return torch.sign(weight)

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None

def compute_state(a):
    pos_num = (a>0).nonzero().size()[0]
    neg_num = (a<0).nonzero().size()[0]
    if pos_num >= neg_num:
        return 1.0
    else:
        return -1.0
def compute_threshold(weight):
    neuron_num = weight.numel()
    rate = 0.55
    threshold = weight.view(neuron_num).topk(int(neuron_num * rate))
    shift = threshold[0][int(neuron_num * rate - 1)]
    return -abs(shift)
import ipdb
'''
def Init_conv_shift(weight, out_channels, shift_2):
    beta = 0.1
    uneven = 0.0
    #shift_2 = torch.zeros(out_channels)
    for i in range(out_channels):
        shift_2[i] = torch.tensor(-abs(weight.mean())*compute_state(weight[i]))
    #shift_2 = shift_2.cuda()
    while(uneven < want_uneven):
        shift_2 = shift_2*beta
        w_reshape = weight.reshape([weight.shape[0], -1]).transpose(0, 1)
        indices = (((w_reshape > shift_2).to(weight.device).float() - 0.5)*2)
        quant =  indices.transpose(0, 1).reshape(weight.shape)
        pos_num = (quant+1).nonzero().size()[0]
        pos_rate = pos_num/quant.numel()
        uneven = abs(pos_rate - 0.5)
        if beta < 10000:
            beta += 0.1
        else:
            shift_2 = torch.ones(out_channels)*0.001
            break
    return shift_2
def Init_fc_shift(weight, shift_1):
#due to torch.autograd.Function Error:autograd
    beta = 0.1
    uneven = 0.0
    #shift = torch.zeros(1)
    #shift_1 = torch.zeros(1, requires_grad=False)
    shift_1 = -abs(weight.mean())*compute_state(weight)
    #shift_1 = shift_1.cuda()
    while(uneven < want_uneven):
        shift_1 = shift_1*beta
        w_reshape = weight.transpose(0, 1)
        indices = ((w_reshape > shift_1).to(weight.device).float() - 0.5)*2 #shift
        quant =  indices.transpose(0, 1).reshape(weight.shape)
        pos_num = (quant+1).nonzero().size()[0]
        pos_rate = pos_num/quant.numel()
        uneven = abs(pos_rate - 0.5)
        if beta < 10000:
            beta += 0.1
        else:
            shift_1 = torch.ones(out_channels)*0.001
            break
        # calculate pos_rate
    return shift_1
'''
# this is the fixed-shift version below.
class Conv2dBWN_Shift(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, mode=q_modes.layer_wise):
        super(Conv2dBWN_Shift, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.q_mode = mode
        self.alpha = nn.Parameter(torch.ones(out_channels))
        # self.shift = torch.zeros(out_channels, requires_grad=False)
        self.register_buffer('shift', torch.zeros(out_channels))
        self.register_buffer('init_state', torch.zeros(1))
        if self.q_mode is q_modes.kernel_wise:
            raise NotImplementedError

    def forward(self, x):
        if self.training and self.init_state == 0:
            # new scheme: calculate shift value by specified pos_rate
            out_channels = self.weight.shape[0]
            #shift = Init_conv_shift(self.weight, out_channels, self.shift)
            for i in range(self.shift.size()[0]):
                #self.shift[i] = torch.tensor(-abs(shift[i])*compute_state(self.weight[i])).type(torch.cuda.FloatTensor)
                #self.shift[i] = torch.tensor(compute_threshold(self.weight[i])*compute_state(self.weight[i])).type(torch.cuda.FloatTensor)
                #self.shift[i] = torch.tensor(-abs(self.weight.mean())*beta*compute_state(self.weight[i])).type(torch.cuda.FloatTensor)
                self.shift[i] = torch.tensor(beta*compute_state(self.weight[i])).type(torch.cuda.FloatTensor)
                # self.alpha += torch.mean(self.weight.reshape([self.out_channels, -1]), dim=1)
            self.init_state.fill_(1)
        quantized_weight = Function_sign_convshift.apply(self.weight, self.alpha, self.shift)
        return F.conv2d(x, quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearBWN_Shift(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearBWN_Shift, self).__init__(in_features, out_features, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1))
        self.register_buffer('shift', torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.training and self.init_state == 0:
            self.shift.fill_(beta * compute_state(self.weight))
            self.init_state.fill_(1)
        quantized_weight = Function_sign_fcshift.apply(self.weight, self.alpha, self.shift)
        return F.linear(x, quantized_weight, self.bias)




class Function_sign_fcshift(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scale, shift):
        w_reshape = weight.transpose(0, 1)
        indices = ((w_reshape > shift).float() - 0.5)*2 #shift
        indices = indices.to(weight.device).float()
        binary_weight = scale * indices
        ctx.save_for_backward(indices, scale)
        return binary_weight.transpose(0, 1)

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_reshape = grad_outputs.transpose(0, 1)
        indices, scale = ctx.saved_tensors
        pruned_indices = torch.ones_like(indices).to(indices.device) - indices
        grad_scale = torch.mean(grad_reshape * indices, dim=0)
        grad_inputs = scale * grad_reshape * indices + \
                         grad_reshape * pruned_indices
        return grad_inputs.transpose(0, 1), \
               grad_scale, None
class Function_sign_convshift(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scale, shift):
        w_reshape = weight.reshape([weight.shape[0], -1]).transpose(0, 1)
        indices = (((w_reshape > shift).float() - 0.5)*2)
        indices = indices.to(weight.device).float()
        binary_weight = scale * indices
        ctx.save_for_backward(indices, scale)
        return binary_weight.transpose(0, 1).reshape(weight.shape)

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_reshape = grad_outputs.reshape([grad_outputs.shape[0], -1]).transpose(0, 1)
        indices, scale = ctx.saved_tensors
        pruned_indices = torch.ones(indices.shape).to(indices.device) - indices
        grad_scale = torch.mean(grad_reshape * indices, dim=0)
        grad_inputs = scale * grad_reshape * indices + \
                         grad_reshape * pruned_indices
        return grad_inputs.transpose(0, 1).reshape(grad_outputs.shape), \
               grad_scale, None




