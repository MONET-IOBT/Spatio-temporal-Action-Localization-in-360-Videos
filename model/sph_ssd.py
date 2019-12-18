
""" SSD network Classes

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

Updated by Gurkirt Singh for ucf101-24 dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.insert(0, '/home/bo/code/realtime-action-detection')
from layers import *
from data import sph_v2,v3
from layers.functions.sph_prior_box import SphPriorBox
from model.spherenet.sphere_cnn import SphereConv2D, SphereMaxPool2D
from model.KernelTransformer.KTNLayer import KTNConv
import os

# KTN stuff

INPUT_WIDTH = 600
FOV = 120
TIED_WEIGHT = 5

featmap = sph_v2['feature_maps']
imgSize = {'vgg.0':(int(INPUT_WIDTH/2),INPUT_WIDTH), 'vgg.2':(int(INPUT_WIDTH/2),INPUT_WIDTH),\
        'vgg.5':(int(INPUT_WIDTH/4),int(INPUT_WIDTH/2)), 'vgg.7':(int(INPUT_WIDTH/4),int(INPUT_WIDTH/2)),\
        'vgg.10':(int(INPUT_WIDTH/8),int(INPUT_WIDTH/4)), 'vgg.12':(int(INPUT_WIDTH/8),int(INPUT_WIDTH/4)), 'vgg.14':(int(INPUT_WIDTH/8),int(INPUT_WIDTH/4)),\
        'vgg.17':featmap[0], 'vgg.19':featmap[0], 'vgg.21':featmap[0],\
        'vgg.24':featmap[1], 'vgg.26':featmap[1], 'vgg.28':featmap[1],\
        'vgg.31':featmap[1], 'vgg.33':featmap[1],\
        'extras.0':featmap[1], 'extras.1':featmap[2], 'extras.2':featmap[2],\
        'extras.3':featmap[3], 'extras.4':featmap[3], 'extras.5':featmap[4],\
        'extras.6':featmap[4], 'extras.7':featmap[5],\
        'loc.0':featmap[0], 'loc.1':featmap[1], 'loc.2':featmap[2],\
        'loc.3':featmap[3], 'loc.4':featmap[4], 'loc.5':featmap[5],\
        'conf.0':featmap[0], 'conf.1':featmap[1], 'conf.2':featmap[2],\
        'conf.3':featmap[3], 'conf.4':featmap[4], 'conf.5':featmap[5]}

LAYERS = imgSize.keys()
# 


class Sph_SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, base, extras, head, num_classes, ver):
        super(Sph_SSD, self).__init__()

        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = SphPriorBox(ver)
        with torch.no_grad():
            self.priors = self.priorbox.forward().cuda()
            self.num_priors = self.priors.size(0)
            self.size = 512

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=1).cuda()

    def transform(self, net_type):
        for i,layer in enumerate(self.vgg):
            name = 'vgg.' + str(i)
            if name in LAYERS:
                if net_type == 'ktn':
                    new_layer = build_ktnconv(name, layer.weight, layer.bias)
                elif net_type == 'sphnet':
                    stride = layer.stride[0]
                    in_channels = layer.in_channels
                    out_channels = layer.out_channels
                    new_layer = SphereConv2D(in_channels, out_channels, stride=stride)
                self.vgg[i] = new_layer

        for i,layer in enumerate(self.extras):
            name = 'extra.' + str(i)
            if name in LAYERS:
                if net_type == 'ktn':
                    new_layer = build_ktnconv(name, layer.weight, layer.bias)
                elif net_type == 'sphnet':
                    stride = layer.stride[0]
                    in_channels = layer.in_channels
                    out_channels = layer.out_channels
                    new_layer = SphereConv2D(in_channels, out_channels, stride=stride)
                self.extras[i] = new_layer

    def forward(self, x):

        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        annot_len = self.priors.shape[1]
        output = (loc.view(loc.size(0), -1, annot_len),
                  conf.view(conf.size(0), -1, self.num_classes),
                  self.priors
                  )
        assert(output[0].shape[-2] == output[2].shape[-2])
        return output

def build_ktnconv(target, kernel, bias):

    fov = FOV
    ih,iw = imgSize[target]
    tied_weights = 1 if ih%5!=0 or iw%5!=0 else TIED_WEIGHT
    dilation = 1
    arch = 'bilinear' #if target == 'vgg.0' else 'residual'
    kernel_shape_type = "dilated"

    sys.stderr.write("Build layer {0} with arch: {1}, tied_weights: {2}\n".format(target, arch, tied_weights))
    ktnconv = KTNConv(kernel,
                      bias,
                      sphereH=ih,
                      imgW=iw,
                      fov=fov,
                      dilation=dilation,
                      tied_weights=tied_weights,
                      arch=arch,
                      kernel_shape_type=kernel_shape_type)
    return ktnconv


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
            layers += [SphereMaxPool2D(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [SphereMaxPool2D(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = SphereConv2D(in_channels, v, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = SphereMaxPool2D(kernel_size=3, stride=1)
    conv6 = SphereConv2D(512, 1024, stride=1)
    conv7 = SphereConv2D(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers



def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S' and in_channels != 'K4':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            elif v == 'K4':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=4, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes, ver):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    box_len = 4 if ver['no_rotation'] else 5
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * box_len, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * box_len, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'C',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'C',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'K4', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_vgg_ssd(num_classes, cfg):

    size = cfg['min_dim'][0]

    return Sph_SSD(*multibox(vgg(base[str(size)], 3),
                            add_extras(extras[str(size)], 1024),
                            mbox[str(size)], num_classes, cfg), num_classes, cfg)
