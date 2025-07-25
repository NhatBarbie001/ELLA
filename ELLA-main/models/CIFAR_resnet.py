'''
Reference:
https://github.com/khurramjaved96/incremental-learning/blob/autoencoders/model/resnet32.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3, num_classes=10):
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        # Thêm alias để tương thích với ResNet chuẩn
        self.layer1 = self.stage_1
        self.layer2 = self.stage_2
        self.layer3 = self.stage_3
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        self.num_classes = num_classes
        self.linear = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x):
        # Trích xuất đặc trưng trước FC layer cuối cùng
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x_1 = self.stage_1(x)
        x_2 = self.stage_2(x_1)
        x_3 = self.stage_3(x_2)
        pooled = self.avgpool(x_3)
        features = pooled.view(pooled.size(0), -1)
        return features

    def logits(self, features):
        # Đưa đặc trưng qua FC layer cuối cùng
        return self.linear(features)

    def forward(self, x):
        # Chuỗi hóa hai bước trên
        features = self.features(x)
        logits = self.logits(features)
        return logits

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


def resnet20mnist():
    """Constructs a ResNet-20 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 20, 1, num_classes=1)
    return model


def resnet32mnist():
    """Constructs a ResNet-32 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 32, 1, num_classes=1)
    return model


def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 or CIFAR-100."""
    model = CifarResNet(ResNetBasicblock, 20, num_classes=num_classes)
    return model


def resnet32(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 or CIFAR-100."""
    model = CifarResNet(ResNetBasicblock, 32, num_classes=num_classes)
    return model


def resnet44(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 44, num_classes=num_classes)
    return model


def resnet56(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 56, num_classes=num_classes)
    return model


def resnet110(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 110, num_classes=num_classes)
    return model

def resnet14(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 14, num_classes=num_classes)
    return model

def resnet26(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 26, num_classes=num_classes)
    return model