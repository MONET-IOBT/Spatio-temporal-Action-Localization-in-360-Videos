'''
Modified from https://github.com/pytorch/vision.git
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..layers import *
from ..data import sph_v2
from ..layers.functions.sph_prior_box import SphPriorBox
import math
from spherenet import SphereConv2D, SphereMaxPool2D

class VGG(nn.Module):
    def __init__(self, base, num_classes):
        super(VGG, self).__init__()

        self.vgg = nn.Sequential(*base)

        self.classifier = nn.Sequential(
            nn.Linear(1024*4*7, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):

        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def sph_vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [SphereMaxPool2D(stride=2)]
        else:
            conv2d = SphereConv2D(in_channels, v, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = SphereMaxPool2D(stride=1)
    conv6 = SphereConv2D(512, 1024, stride=1)
    conv7 = SphereConv2D(1024, 1024, stride=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers




base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'C',
            512, 512, 512],
    '512': [],
}

def vgg4ssd():
    return VGG(vgg(base['300'], 3),10)

def sph_vgg4ssd():
    return VGG(sph_vgg(base['300'], 3),10)