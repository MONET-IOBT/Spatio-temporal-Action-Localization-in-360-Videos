#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfg import MODEL_DIR
from .KTNLayer import KTNConv
from .util import enable_gpu

import sys
sys.path.insert(0, '/home/bo/research/realtime-action-detection')
from layers import *
from data import sph_v2
from model.sph_ssd import Sph_SSD


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
class KTNSSD(Sph_SSD):

    # def __init__(self, base, extras, head, num_classes):
    #     super(KTNSSD, self).__init__()

    #     self.num_classes = num_classes
    #     # TODO: implement __call__ in PriorBox
    #     self.priorbox = SphPriorBox(sph_v2)
    #     with torch.no_grad():
    #         self.priors = self.priorbox.forward().cuda()
    #         self.num_priors = self.priors.size(0)
    #         self.size = 300

    #     # SSD network
    #     self.vgg = nn.ModuleList(base)
    #     # Layer learns to scale the l2 normalized features from conv4_3
    #     self.L2Norm = L2Norm(512, 20)
    #     self.extras = nn.ModuleList(extras)

    #     self.loc = nn.ModuleList(head[0])
    #     self.conf = nn.ModuleList(head[1])

    #     self.softmax = nn.Softmax(dim=1).cuda()

    def transform(self):
        for i,layer in enumerate(self.vgg):
            name = 'vgg.' + str(i)
            if name in LAYERS:
                self.vgg[i] = build_ktnconv(name, layer.weight, layer.bias)

        for i,layer in enumerate(self.extras):
            name = 'extra.' + str(i)
            if name in LAYERS:
                self.extras[i] = build_ktnconv(name, layer.weight, layer.bias)

        # for i,layer in enumerate(self.extras):
        #     name = 'loc.' + str(i)
        #     if name in LAYERS:
        #         self.loc[i] = build_ktnconv(name, layer.weight, layer.bias)
        #         print(self.loc[i])
        #         # self.loc[i] = enable_gpu(self.loc[i], gpu=True)

        # for i,layer in enumerate(self.extras):
        #     name = 'conf.' + str(i)
        #     if name in LAYERS:
        #         self.conf[i] = build_ktnconv(name, layer.weight, layer.bias)
        #         # self.conf[i] = enable_gpu(self.conf[i], gpu=True)


#     def forward(self, x):

#         """Applies network layers and ops on input image(s) x.

#         Args:
#             x: input image or batch of images. Shape: [batch,3*batch,300,300].

#         Return:
#             Depending on phase:
#             test:
#                 Variable(tensor) of output class label predictions,
#                 confidence score, and corresponding location predictions for
#                 each object detected. Shape: [batch,topk,7]

#             train:
#                 list of concat outputs from:
#                     1: confidence layers, Shape: [batch*num_priors,num_classes]
#                     2: localization layers, Shape: [batch,num_priors*4]
#                     3: priorbox layers, Shape: [2,num_priors*4]
#         """

#         sources = list()
#         loc = list()
#         conf = list()

#         # apply vgg up to conv4_3 relu
#         for k in range(23):
#             x = self.vgg[k](x)

#         s = self.L2Norm(x)
#         sources.append(s)

#         # apply vgg up to fc7
#         for k in range(23, len(self.vgg)):
#             x = self.vgg[k](x)
#         sources.append(x)

#         # apply extra layers and cache source layer outputs
#         for k, v in enumerate(self.extras):
#             x = F.relu(v(x), inplace=True)
#             if k % 2 == 1:
#                 sources.append(x)

#         # apply multibox head to source layers
#         for (x, l, c) in zip(sources, self.loc, self.conf):
#             loc.append(l(x).permute(0, 2, 3, 1).contiguous())
#             conf.append(c(x).permute(0, 2, 3, 1).contiguous())

#         loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
#         conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
#         annot_len = self.priors.shape[1]
#         output = (loc.view(loc.size(0), -1, annot_len),
#                   conf.view(conf.size(0), -1, self.num_classes),
#                   self.priors
#                   )
#         assert(output[0].shape[-2] == output[2].shape[-2])
#         return output


def build_ktnconv(target, kernel, bias):

    fov = FOV
    ih,iw = imgSize[target]
    tied_weights = 1 if target == 'vgg.0' or ih%5!=0 or iw%5!=0 else TIED_WEIGHT
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

# def ktn_vgg(cfg, i, batch_norm=False):
#     layers = []
#     in_channels = i
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         elif v == 'C':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#     conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
#     conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
#     layers += [pool5, conv6,
#                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
#     return layers


# def add_extras(cfg, i, batch_norm=False):
#     # Extra layers added to VGG for feature scaling
#     layers = []
#     in_channels = i
#     flag = False
#     for k, v in enumerate(cfg):
#         if in_channels != 'S':
#             if v == 'S':
#                 layers += [nn.Conv2d(in_channels, cfg[k + 1],
#                            kernel_size=(1, 3)[flag], stride=2, padding=1)]
#             else:
#                 layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
#             flag = not flag
#         in_channels = v
#     return layers


# def multibox(vgg, extra_layers, cfg, num_classes):
#     loc_layers = []
#     conf_layers = []
#     vgg_source = [24, -2]
#     box_len = 4 if sph_v2['no_rotation'] else 5
#     for k, v in enumerate(vgg_source):
#         loc_layers += [nn.Conv2d(vgg[v].out_channels,
#                                  cfg[k] * box_len, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(vgg[v].out_channels,
#                         cfg[k] * num_classes, kernel_size=3, padding=1)]
#     for k, v in enumerate(extra_layers[1::2], 2):
#         loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
#                                  * box_len, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
#                                   * num_classes, kernel_size=3, padding=1)]
#     return vgg, extra_layers, (loc_layers, conf_layers)


# base = {
#     '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'C',
#             512, 512, 512],
#     '512': [],
# }
# extras = {
#     '300': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 256],
#     '512': [],
# }
# rot = sph_v2['num_rotations']
# no_rot = sph_v2['no_rotation']
# mbox = {
#     '300': [4*(rot,1)[no_rot], 6*(rot,1)[no_rot], 6*(rot,1)[no_rot], 6*(rot,1)[no_rot], 4*(rot,1)[no_rot], 4*(rot,1)[no_rot]],  # number of boxes per feature map location
#     '512': [],
# }


# def build_ktn_ssd(size=300, num_classes=21):

#     if size != 300:
#         print("Error: Sorry only SSD300 is supported currently!")
#         return

#     basenet = ktn_vgg(base[str(size)], 3)
#     return KTNSSD(*multibox(basenet,
#                             add_extras(extras[str(size)], 1024),
#                             mbox[str(size)], num_classes), num_classes)

# if __name__ == '__main__':
#     import time
#     net = build_ktn_ssd(300, 25)
#     net.transform()
#     net = net.cuda()
#     t1 = time.perf_counter()
#     data = torch.randn(4,3,300,600)
#     data = data.cuda()
#     out = net(data)
#     t2 = time.perf_counter()
#     print(out[0].shape)
#     print('Forward Time {:0.3f}'.format(t2-t1))