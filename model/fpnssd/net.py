'''SSD model with VGG16 as feature extractor.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys
sys.path.insert(0, '/home/bo/research/realtime-action-detection')
from model.fpnssd.fpn import FPN50
from layers.functions.sph_prior_box import SphPriorBox
from model.spherenet.sphere_cnn import SphereConv2D, SphereMaxPool2D
from model.KernelTransformer.KTNLayer import KTNConv

class FPNSSD512(nn.Module):

    def __init__(self, num_classes, cfg):
        super(FPNSSD512, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = (4, 6, 6, 6, 6, 4, 4)
        self.in_channels = (256, 256, 256, 256, 256, 256, 256)
        self.cfg = cfg

        self.extractor = FPN50()
        priorbox = SphPriorBox(cfg)
        with torch.no_grad():
            self.priors = priorbox.forward().cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
        	self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4, kernel_size=3, padding=1)]
        	self.cls_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*self.num_classes, kernel_size=3, padding=1)]
        if not cfg['no_rotation']:
            self.rot_layers = nn.ModuleList()
            for i in range(len(self.in_channels)):
                self.rot_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i], kernel_size=3, padding=1)]

    def transform(self):
        def transform1(layer,s):
            for i,l in enumerate(layer):
                stride = l.conv2.stride[0]
                in_channels = l.conv2.in_channels
                out_channels = l.conv2.out_channels
                layer[i].conv2 = SphereConv2D(in_channels, out_channels, stride=stride)
                # layer[i].conv2 = build_ktnconv(s, layer[i].conv2.weight, None)
        transform1(self.extractor.layer1,[128,128])
        transform1(self.extractor.layer2,[128,128])
        transform1(self.extractor.layer3,[64,64])
        transform1(self.extractor.layer4,[32,32])

        def transform2(layer, s):
            stride = layer.stride[0]
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            return SphereConv2D(in_channels, out_channels, stride=stride)
            # return build_ktnconv(s, layer.weight, layer.bias)
        self.extractor.conv6 = transform2(self.extractor.conv6, [16,16])
        self.extractor.conv7 = transform2(self.extractor.conv7, [8,8])
        self.extractor.conv8 = transform2(self.extractor.conv8, [4,4])
        self.extractor.conv9 = transform2(self.extractor.conv9, [2,2])

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0),-1,4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0,2,3,1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0),-1,self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        if not self.cfg['no_rotation']:
            rot_preds = []
            for i, x in enumerate(xs):
                rot_pred = self.rot_layers[i](x)
                rot_pred = rot_pred.permute(0,2,3,1).contiguous()
                rot_preds.append(rot_pred.view(rot_pred.size(0),-1))
            rot_preds = torch.cat(rot_preds, 1)

        if self.cfg['no_rotation']:
            return loc_preds, cls_preds, self.priors
        else:
            return loc_preds, cls_preds, self.priors, rot_preds 

FOV = 120
TIED_WEIGHT = 4

def build_ktnconv(imgSize, kernel, bias):

    fov = FOV
    ih,iw = imgSize
    tied_weights = 1 if ih%TIED_WEIGHT!=0 or iw%TIED_WEIGHT!=0 else TIED_WEIGHT
    dilation = 2
    arch = 'bilinear' 
    kernel_shape_type = "dilated"

    sys.stderr.write("Arch: {0}, tied_weights: {1}\n".format(arch, tied_weights))
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

def test():
    net = FPNSSD512(25)
    loc_preds, cls_preds, priors = net(Variable(torch.randn(1,3,512,1024)))
    print(loc_preds.size(), cls_preds.size(), priors.size())
    print(net.state_dict().keys())

if __name__ == '__main__':
    test()
