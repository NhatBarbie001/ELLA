"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.functional import relu, avg_pool2d
from models.CIFAR_resnet import resnet32 as cifar_resnet32
from torchvision.models import resnet18 as tv_resnet18, resnet34 as tv_resnet34, resnet50 as tv_resnet50, resnet101 as tv_resnet101, resnet152 as tv_resnet152


#from models.CIFAR_resnet import ResNet32

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.nclasses = num_classes
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)
        # self.linear = nn.Linear(512, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.nclasses == 100: #cifar100, imagenet100 (subset)
            out = avg_pool2d(out, 4)
        elif self.nclasses == 74: #vfn74
            out = avg_pool2d(out, 28) 
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits


def Reduced_ResNet18(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

def ResNet18(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)


#def Resnet32() is implemented in CIFAR_resnet.py, applied directly in the code below

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)

def ResNet50(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)


def ResNet101(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)


def ResNet152(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)


def ResNet20(nclasses, nf=16, bias=True):
    return ResNet(BasicBlock, [3, 3, 3, 3], nclasses, nf, bias)

def ResNet56(nclasses, nf=16, bias=True):
    return ResNet(BasicBlock, [9, 9, 9, 9], nclasses, nf, bias)

def ResNet110(nclasses, nf=16, bias=True):
    return ResNet(BasicBlock, [18, 18, 18, 18], nclasses, nf, bias)

def get_encoder(backbone, nclass):
    if backbone == 'reduced_resnet18':
        return Reduced_ResNet18(nclass)
    elif backbone == 'resnet18':
        return ResNet18(nclass)
    elif backbone == 'resnet32':
        encoder = cifar_resnet32(num_classes=nclass)
        return encoder
    elif backbone == 'resnet20':
        return ResNet20(nclass)
    elif backbone == 'resnet34':
        return ResNet34(nclass)
    elif backbone == 'resnet50':
        return ResNet50(nclass)
    elif backbone == 'resnet56':
        return ResNet56(nclass)
    elif backbone == 'resnet101':
        return ResNet101(nclass)
    elif backbone == 'resnet110':
        return ResNet110(nclass)
    elif backbone == 'resnet152':
        return ResNet152(nclass)
    elif backbone == 'tv_resnet18':
        model = tv_resnet18(num_classes=nclass)
        return model
    elif backbone == 'tv_resnet34':
        model = tv_resnet34(num_classes=nclass)
        return model
    elif backbone == 'tv_resnet50':
        model = tv_resnet50(num_classes=nclass)
        return model
    elif backbone == 'tv_resnet101':
        model = tv_resnet101(num_classes=nclass)
        return model
    elif backbone == 'tv_resnet152':
        model = tv_resnet152(num_classes=nclass)
        return model
    else:
        raise ValueError(f'Unknown backbone: {backbone}')

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=160, head='linear', feat_dim=128, nclass=74, backbone='reduced_resnet18'):
        super(SupConResNet, self).__init__()
        self.encoder = get_encoder(backbone, nclass)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            feat = F.normalize(self.head(feat), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        # print('feature normalized: ', feat.size())
        return feat

    def features(self, x):
        return self.encoder.features(x)

    def logits(self, x):
        # print('logits size: ', self.encoder.forward(x).size())
        return self.encoder.forward(x)

    def fclayer(self, feat):
        return self.encoder.logits(feat)

class ContrastiveLR(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=None, head='mlp', feat_dim=128, nclass=100, backbone='reduced_resnet18', datatype='cifar100'):
        super(ContrastiveLR, self).__init__()
        self.encoder = get_encoder(backbone, nclass)
        # Tự động xác định dim_in nếu không truyền vào
        input_size = (3, 224, 224)  # VFN74
        if datatype == 'cifar100':
            input_size = (3, 32, 32)
        
        if dim_in is None:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_size)
                feat = self.encoder.features(dummy)
                dim_in = feat.shape[1]
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            feat = F.normalize(self.head(feat), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        # print('feature normalized: ', feat.size())
        return feat

    def features(self, x):
        return self.encoder.features(x)

    def logits(self, x):
        # print('logits size: ', self.encoder.forward(x).size())
        return self.encoder.forward(x)

    def fclayer(self, feat):
        return self.encoder.logits(feat)