'''SSD model with VGG16 as feature extractor.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
import sys
sys.path.insert(0, '/home/bo/research/realtime-action-detection')
from layers.functions.sph_prior_box import SphPriorBox
from data import v1,v2,v3,v4,v5

class MobileNetExtractor512(nn.Module):
    def __init__(self):
        super(MobileNetExtractor512, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1),
            conv_dw(1024, 1024, 1),
        )

        self.extras = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1),
            nn.ReLU()
        )
    ])

    def forward(self, x):
        hs = []
        for i,l in enumerate(self.model):
            x = l(x)
            if i == 9 or i == 14:
                hs.append(x)

        for l in self.extras:
            x = l(x)
            hs.append(x)
        return hs

# def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
#     """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
#     """
#     return Sequential(
#         Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
#                groups=in_channels, stride=stride, padding=padding),
#         ReLU(),
#         Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
#     )

# class MobileNetExtractorLite512(nn.Module):
#     def __init__(self):
#         super(MobileNetExtractor512, self).__init__()

#         def conv_bn(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True)
#             )

#         def conv_dw(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#                 nn.BatchNorm2d(inp),
#                 nn.ReLU(inplace=True),

#                 nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True),
#             )

#         self.model = nn.Sequential(
#             conv_bn(3, 64, 2),
#             conv_dw(64, 64, 1),
#             conv_dw(64, 128, 2),
#             conv_dw(128, 128, 1),
#             conv_dw(128, 256, 2),
#             conv_dw(256, 256, 1),
#             conv_dw(256, 256, 1),
#             conv_dw(256, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 2),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 1024, 1),
#             conv_dw(1024, 1024, 1),
#         )

#         self.extras = nn.ModuleList([
#         nn.Sequential(
#             nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
#             nn.ReLU(),
#             SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
#         ),
#         nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
#             nn.ReLU(),
#             SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
#         ),
#         nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
#             nn.ReLU(),
#             SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
#         ),
#         nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
#             nn.ReLU(),
#             SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
#         ),
#         nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
#             nn.ReLU(),
#             SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
#         )
#     ])

#     def forward(self, x):
#         hs = []
#         for i,l in enumerate(self.model):
#             x = l(x)
#             if i == 9 or i == 14:
#                 hs.append(x)

#         for l in self.extras:
#             x = l(x)
#             hs.append(x)
#         return hs

class MobileSSD512(nn.Module):
    steps = (8, 16, 32, 64, 128, 256, 512)
    box_sizes = (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)  # default bounding box sizes for each feature map.
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))
    fm_sizes = (64, 32, 16, 8, 4, 2, 1)

    def __init__(self, num_classes, cfg):
        super(MobileSSD512, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = (4, 6, 6, 6, 6, 4, 4)
        self.in_channels = (512, 1024, 512, 256, 256, 256, 256)

        self.extractor = MobileNetExtractor512()
        priorbox = SphPriorBox(cfg)
        with torch.no_grad():
            self.priors = priorbox.forward().cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
        	self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4, kernel_size=3, padding=1)]
        	self.cls_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*self.num_classes, kernel_size=3, padding=1)]

        self._initialize_weights()

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
        return loc_preds, cls_preds, self.priors

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# class MobileSSDLite512(nn.Module):
#     steps = (8, 16, 32, 64, 128, 256, 512)
#     box_sizes = (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)  # default bounding box sizes for each feature map.
#     aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))
#     fm_sizes = (64, 32, 16, 8, 4, 2, 1)

#     def __init__(self, num_classes, cfg):
#         super(MobileSSDLite512, self).__init__()
#         self.num_classes = num_classes
#         self.num_anchors = (4, 6, 6, 6, 6, 4, 4)
#         self.in_channels = (512, 1024, 512, 256, 256, 256, 256)

#         self.extractor = MobileNetExtractorLite512()
#         priorbox = SphPriorBox(cfg)
#         with torch.no_grad():
#             self.priors = priorbox.forward().cuda()
#         self.softmax = nn.Softmax(dim=1).cuda()
#         self.loc_layers = nn.ModuleList()
#         self.cls_layers = nn.ModuleList()
#         for i in range(len(self.in_channels)):
#             self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4, kernel_size=3, padding=1)]
#             self.cls_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*self.num_classes, kernel_size=3, padding=1)]

#         self._initialize_weights()

#     def forward(self, x):
#         loc_preds = []
#         cls_preds = []
#         xs = self.extractor(x)
#         for i, x in enumerate(xs):
#             loc_pred = self.loc_layers[i](x)
#             loc_pred = loc_pred.permute(0,2,3,1).contiguous()
#             loc_preds.append(loc_pred.view(loc_pred.size(0),-1,4))

#             cls_pred = self.cls_layers[i](x)
#             cls_pred = cls_pred.permute(0,2,3,1).contiguous()
#             cls_preds.append(cls_pred.view(cls_pred.size(0),-1,self.num_classes))

#         loc_preds = torch.cat(loc_preds, 1)
#         cls_preds = torch.cat(cls_preds, 1)
#         return loc_preds, cls_preds, self.priors

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()

def test():
    net = MobileSSD512(25,v3)
    loc_preds, cls_preds, priors = net(Variable(torch.randn(16,3,512,1024)))
    print(loc_preds.size(), cls_preds.size(), priors.size())

if __name__ == '__main__':
    test()
