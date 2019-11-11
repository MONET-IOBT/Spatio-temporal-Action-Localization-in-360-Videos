""" Generates prior boxes for SSD netowrk

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

"""

import torch
from math import sqrt as sqrt
from itertools import product as product

class SphPriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(SphPriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        assert('num_rotations' in cfg)
        self.num_rotations = cfg['num_rotations']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        if self.version == 'sph_v2':
            for k, f in enumerate(self.feature_maps):
                for i, j in product(range(f), repeat=2):
                    boxes = []
                    f_k = self.image_size / self.steps[k]
                    # unit center x,y
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k]/self.image_size
                    boxes.append([cx, cy, s_k, s_k])

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    boxes.append([cx, cy, s_k_prime, s_k_prime])

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        boxes.append([cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)])
                        boxes.append([cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)])

                    # add rotation to all boxes
                    for cx,cy,sx,sy in boxes:
                        for l,m,n in product(range(self.num_rotations), repeat=3):
                            rot_x = 1./self.num_rotations*l
                            rot_y = 1./self.num_rotations*m
                            rot_z = 1./self.num_rotations*n
                            mean += [cx,cy,sx,sy,rot_x,rot_y,rot_z]
                        

        else:
            print('Not implemented')
            exit(-1)
        # back to torch land
        output = torch.Tensor(mean).view(-1, 7)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def pre_compute_iou():
    pass
    # import sys
    # sys.path.insert(0,'/home/bo/research/realtime-action-detection/')
    # from data import sph_v2
    # from layers.sph_box_utils import iou
    # import numpy as np

    # priorbox = SphPriorBox(sph_v2)
    # with torch.no_grad():
    #     priors = priorbox.forward().cuda()
    # data = {}
    # tensor2key = {}
    # cnt = 0
    # for i, j in product(range(2), repeat=2):
    #     if i==j:continue
    #     s1 = str((priors[i].data.cpu(),priors[j].data.cpu()))
    #     s2 = str((priors[j].data.cpu(),priors[i].data.cpu()))
    #     if s1 in tensor2key or s2 in tensor2key:continue
    #     IoU = iou(priors[i],priors[j],np.pi/3,use_precompute=False)
    #     key = s1
        
    #     data[key] = IoU
    # exit(0)
    # import json

    # with open('data.json', 'w') as fp:
    #     json.dump(data, fp)


if __name__ == '__main__':
    pre_compute_iou()