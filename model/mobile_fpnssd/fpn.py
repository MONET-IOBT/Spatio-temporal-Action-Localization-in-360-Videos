import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_dw(inp, oup, kernel_size=3, stride=1, padding=0, use_batchnorm=True):
    if use_batchnorm:
        return nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
            nn.ReLU(),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.ReLU(),
        )

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_dw(in_planes, planes, 1, 1, 0)
        self.conv2 = conv_dw(planes, planes, 3, stride, 1)
        self.conv3 = conv_dw(planes, self.expansion*planes, 1, 1, 0)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = conv_dw(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = conv_dw( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = conv_dw( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv9 = conv_dw( 256, 256, kernel_size=3, stride=2, padding=1, use_batchnorm=False)

        # Top-down layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = conv_dw(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = conv_dw(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.relu1(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        p8 = self.conv8(p7)
        p9 = self.conv9(p8)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7, p8, p9


def FPN50():
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    return FPN(Bottleneck, [3,4,23,3])

def FPN152():
    return FPN(Bottleneck, [3,8,36,3])

def test():
    net = FPN50()
    fms = net(torch.randn(1,3,512,512))
    for fm in fms:
        print(fm.size())

if __name__ == "__main__":
    test()