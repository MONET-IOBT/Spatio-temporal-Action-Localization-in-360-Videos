import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/bo/research/realtime-action-detection')
from model.CP.model.cube_pad import CubePad
from model.CP.utils.cube_to_equi import Cube2Equi
from model.CP.utils.equi_to_cube import Equi2Cube

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.pad = CubePad(1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(self.pad(out))))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=0)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=0)
        self.conv8 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=0)
        self.conv9 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

        # cube converter
        self.e2c = Equi2Cube(256, torch.zeros((512,1024)))
        self.c2e32 = Cube2Equi(32)
        self.c2e16 = Cube2Equi(16)
        self.c2e8 = Cube2Equi(8)
        self.c2e4 = Cube2Equi(4)
        self.c2e2 = Cube2Equi(2)
        self.c2e1 = Cube2Equi(1)

        # cube padding
        self.pad1 = CubePad(1)
        self.pad3 = CubePad(3)

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
        batch_size,channels = x.shape[:2]
        # convert to cubes
        cubes = torch.zeros(batch_size*6,channels,256,256).cuda()
        c_i = 0
        x = x.permute(0,2,3,1).cpu().numpy()
        for b_i in range(batch_size):
            cubeset = self.e2c.to_cube(x[b_i])
            for i in cubeset:
                cubes[c_i] = torch.from_numpy(cubeset[i]).permute(2,0,1)
                c_i += 1

        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(self.pad3(cubes))))
        c1 = F.max_pool2d(self.pad1(c1), kernel_size=3, stride=2, padding=0)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(self.pad1(c5))         # 8x16 -> 4x4
        p7 = self.conv7(self.pad1(F.relu(p6))) # 4x8 -> 2x2
        p8 = self.conv8(self.pad1(F.relu(p7))) # 2x4 -> 1x1
        # Top-down
        p5 = self.toplayer(c5) # 16x32 -> 8x8
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(self.pad1(p4)) # 32x64 -> 16x16
        p3 = self.smooth2(self.pad1(p3)) # 64x128 -> 32x32
        # transform to equi
        p3 = self.c2e32.to_equi_nn(p3)
        p4 = self.c2e16.to_equi_nn(p4)
        p5 = self.c2e8.to_equi_nn(p5)
        p6 = self.c2e4.to_equi_nn(p6)
        p7 = self.c2e2.to_equi_nn(p7)
        p8 = self.c2e1.to_equi_nn(p8)
        p9 = self.conv9(F.relu(p8))

        return p3, p4, p5, p6, p7, p8, p9


def FPN50():
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    return FPN(Bottleneck, [3,4,23,3])

def FPN152():
    return FPN(Bottleneck, [3,8,36,3])


def test():
    net = FPN50().cuda()
    fms = net(torch.randn(2,3,512,1024).cuda())
    for fm in fms:
        print(fm.size())

test()
