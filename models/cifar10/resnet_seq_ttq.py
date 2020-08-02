import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import ipdb

from models.modules import Conv2dBWN, LinearBWN, Conv2dBWN_Shift, LinearBWN_Shift
__all__ = ['ResNet', 'cifar10_resnet18', 'cifar10_resnet18_bwn_shift', 'cifar10_resnet18_bwn_all']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bwn(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2dBWN(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bwn_shift(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2dBWN_Shift(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1_bwn(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2dBWN(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1_bwn_shift(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2dBWN_Shift(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



def fc_bwn(in_features, out_features):
    """fc layer"""
    return LinearBWN(in_features, out_features, bias=False)
def fc_bwn_shift(in_features, out_features):
    """fc layer"""
    return LinearBWN_Shift(in_features, out_features, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1_seq = nn.Sequential(conv3x3(inplanes, planes, stride),
                                       nn.BatchNorm2d(planes),
                                       )
        self.relu = nn.ReLU(inplace=True)
        self.conv2_seq = nn.Sequential(conv3x3(planes, planes),
                                       nn.BatchNorm2d(planes))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_seq(x)
        out = self.relu(out)

        out = self.conv2_seq(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlockBWN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockBWN, self).__init__()
        self.conv1_seq = nn.Sequential(conv3x3_bwn(inplanes, planes, stride),
                                       nn.BatchNorm2d(planes),
                                       )
        self.relu = nn.ReLU(inplace=True)
        self.conv2_seq = nn.Sequential(conv3x3_bwn(planes, planes),
                                       nn.BatchNorm2d(planes))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_seq(x)
        out = self.relu(out)

        out = self.conv2_seq(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlockBWN_Shift(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockBWN_Shift, self).__init__()
        self.conv1_seq = nn.Sequential(conv3x3_bwn_shift(inplanes, planes, stride),
                                       nn.BatchNorm2d(planes),
                                       )
        self.relu = nn.ReLU(inplace=True)
        self.conv2_seq = nn.Sequential(conv3x3_bwn_shift(planes, planes),
                                       nn.BatchNorm2d(planes))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_seq(x)
        out = self.relu(out)

        out = self.conv2_seq(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1_seq = nn.Sequential(conv1x1(inplanes, planes),
                                       nn.BatchNorm2d(planes))
        self.conv2_seq = nn.Sequential(conv3x3(planes, planes, stride),
                                       nn.BatchNorm2d(planes))
        self.conv3_seq = nn.Sequential(conv1x1(planes, planes * self.expansion),
                                       nn.BatchNorm2d(planes * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_seq(x)
        out = self.relu(out)

        out = self.conv2_seq(out)
        out = self.relu(out)

        out = self.conv3_seq(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1_seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_seq(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class ResNetBWN_Shift(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetBWN_Shift, self).__init__()
        self.inplanes = 64
        self.conv1_seq = nn.Sequential(
            conv3x3_bwn_shift(3, 64, stride=1),
            nn.BatchNorm2d(64), )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_bwn_shift(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_bwn_shift(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_seq(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetBWN_ALL(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetBWN_ALL, self).__init__()
        self.inplanes = 64
        self.conv1_seq = nn.Sequential(
            conv3x3_bwn(3, 64, stride=1),
            nn.BatchNorm2d(64), )  # first layer in full precision
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_bwn(512 * block.expansion, num_classes)  # last layer in full precision

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_bwn(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_seq(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
def cifar10_resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    if pretrained:
        print("load!")
        info = torch.load("../scripts/logger/cifar10_resnet18_baseline/cifar10_resnet18_best.pth.tar")

    return model

def cifar10_resnet18_bwn_all(pretrained=False, **kwargs):

    model = ResNetBWN_ALL(BasicBlockBWN, [2, 2, 2, 2], num_classes=10)
    if pretrained:
        print("load!")
        info = torch.load("../scripts/logger/cifar10_resnet18_baseline/cifar10_resnet18_best.pth.tar")
        model.load_state_dict(info['state_dict'], strict=False)

    return model

def cifar10_resnet18_bwn_shift(pretrained=True, **kwargs):

    model = ResNetBWN_Shift(BasicBlockBWN_Shift, [2, 2, 2, 2], num_classes=10)
    if pretrained:
        print("load!")
        info = torch.load("../scripts/logger/cifar10_resnet18_baseline/cifar10_resnet18_best.pth.tar")
        model.load_state_dict(info['state_dict'], strict=False)

    return model



if __name__ == '__main__':
    net = cifar10_resnet18_bwn_shift(True)
    params = net.state_dict()
    print(net)

