'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Conv2dBWN, LinearBWN, Conv2dBWN_Shift, LinearBWN_Shift
__all__ = ['mnist_lenet', 'mnist_lenet_bwn_all', 'mnist_lenet_bwn_shift']

def conv5x5_bwn(in_planes, out_planes, stride=1):
    """5x5 convolution without padding!"""
    return Conv2dBWN(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=0, bias=False)

def conv5x5_bwn_shift(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return Conv2dBWN_Shift(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=0, bias=False)
def fc_bwn(in_features, out_features):
    """fc layer"""
    return LinearBWN(in_features, out_features, bias=False)
def fc_bwn_shift(in_features, out_features):
    """fc layer"""
    return LinearBWN_Shift(in_features, out_features, bias=False)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNetBWN_ALL(nn.Module):
    def __init__(self):
        super(LeNetBWN_ALL, self).__init__()
        self.conv1 = conv5x5_bwn(1, 6)
        self.conv2 = conv5x5_bwn(6, 16)
        self.fc1 = fc_bwn(16 * 5 * 5, 120)
        self.fc2 = fc_bwn(120, 84)
        self.fc3 = fc_bwn(84, 10)

    def forward(self, x):
        #print(x.size())
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNetBWN_Shift(nn.Module):
    def __init__(self):
        super(LeNetBWN_Shift, self).__init__()
        self.conv1 = conv5x5_bwn_shift(1, 6)
        self.conv2 = conv5x5_bwn_shift(6, 16)
        self.fc1 = fc_bwn_shift(16 * 5 * 5, 120)
        self.fc2 = fc_bwn_shift(120, 84)
        self.fc3 = fc_bwn_shift(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
def mnist_lenet(pretrained=True, **kwargs):

    model = LeNet()
    if pretrained:
        print("load!")
        info = torch.load("../scripts/logger/mnist_lenet_baseline/mnist_lenet_best.pth.tar")
        model.load_state_dict(info['state_dict'], strict=True)

    return model

def mnist_lenet_bwn_all(pretrained=True, **kwargs):

    model = LeNetBWN_ALL()
    if pretrained:
        print("load!")
        info = torch.load("../scripts/logger/mnist_lenet_baseline/mnist_lenet_best.pth.tar")
        model.load_state_dict(info['state_dict'], strict=True)

    return model

def mnist_lenet_bwn_shift(pretrained=True, **kwargs):

    model = LeNetBWN_Shift()
    if pretrained:
        print("load!")
        info = torch.load("../scripts/logger/mnist_lenet_baseline/mnist_lenet_best.pth.tar")
        model.load_state_dict(info['state_dict'], strict=False)
    return model


if __name__ == '__main__':
    net = mnist_lenet_bwn_all(True)
    params = net.state_dict()
    print(net)

