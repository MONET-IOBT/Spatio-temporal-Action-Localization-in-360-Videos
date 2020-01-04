'''SSD model with VGG16 as feature extractor.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
import sys
sys.path.insert(0, '/home/bo/research/realtime-action-detection')
from layers.functions.sph_prior_box import SphPriorBox
from data import v1,v2,v3,v4,v5,v6

def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ReLU(inplace=True)
        )

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, padding, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, padding, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])

class MobileNetExtractorLite300V2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(MobileNetExtractorLite300V2, self).__init__()

        block = InvertedResidual
        input_channel = 64
        last_channel = 512
        interverted_residual_setting = [
            # t, c, n, s
            [2, 64, 1, 1],
            [2, 128, 2, 2],
            [2, 256, 3, 2],
            [2, 512, 4, 2],
            [1, 512, 3, 1],
            [2, 512, 3, 2],
            [1, 512, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
            
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        

        self.extras = nn.ModuleList([
            InvertedResidual(512, 256, stride=2, padding=1, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, padding=1, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, padding=0, expand_ratio=0.5)])

    def forward(self, x):
        hs = []
        source_layer_indexes = [
            GraphPath(7, 'conv', 3),
            GraphPath(14, 'conv', 3),
            19,
        ]

        start_layer_index = 0
        for end_layer_index in source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
            else:
                path = None
            for layer in self.features[start_layer_index: end_layer_index]:
                x = layer(x)
            y = x
            if path:
                sub = getattr(self.features[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            hs.append(y)

        for l in self.extras:
            x = l(x)
            hs.append(x)
        return hs

class MobileSSDLite300V2(nn.Module):
    steps = (8, 16, 32, 64, 128, 256, 512)
    box_sizes = (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)  # default bounding box sizes for each feature map.
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))
    fm_sizes = (64, 32, 16, 8, 4, 2, 1)

    def __init__(self, num_classes, cfg):
        super(MobileSSDLite300V2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = (4, 6, 6, 6, 6, 4, 4)
        self.in_channels = (576, 1024, 512, 256, 256, 64)

        self.extractor = MobileNetExtractorLite300V2()
        priorbox = SphPriorBox(cfg)
        with torch.no_grad():
            self.priors = priorbox.forward().cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.loc_layers = nn.ModuleList([
            SeperableConv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            # nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=1),
        ])
        self.cls_layers = nn.ModuleList([
            SeperableConv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=1),
        ])
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

def test():
    net = MobileSSDLite300V2(25,v1)
    loc_preds, cls_preds, priors = net(Variable(torch.randn(16,3,300,300)))
    print(loc_preds.size(), cls_preds.size(), priors.size())

if __name__ == '__main__':
    test()
