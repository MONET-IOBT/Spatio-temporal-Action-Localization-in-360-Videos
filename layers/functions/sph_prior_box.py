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
        self.no_rotation = cfg['no_rotation']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        if self.version == 'sph_v2' or self.version == 'v4' or self.version == 'v5':
            for k, f in enumerate(self.feature_maps):
                h,w = f
                for i in range(h):
                    for j in range(w):
                        boxes = []
                        f_kx = self.image_size[1] * 1. / self.steps[k][1]
                        f_ky = self.image_size[0] * 1. / self.steps[k][0]
                        # unit center x,y
                        cx = (j + 0.5) / f_kx
                        cy = (i + 0.5) / f_ky

                        # aspect_ratio: 1
                        # rel size: min_size
                        s_kx = self.min_sizes[k] * 1. /self.image_size[1]
                        s_ky = self.min_sizes[k] * 1. /self.image_size[0]
                        boxes.append([cx, cy, s_kx, s_ky])

                        # aspect_ratio: 1
                        # rel size: sqrt(s_k * s_(k+1))
                        s_kx_prime = sqrt(s_kx * (self.max_sizes[k] * 1. /self.image_size[1]))
                        s_ky_prime = sqrt(s_ky * (self.max_sizes[k] * 1. /self.image_size[0]))
                        boxes.append([cx, cy, s_kx_prime, s_ky_prime])

                        # rest of aspect ratios
                        for ar in self.aspect_ratios[k]:
                            boxes.append([cx, cy, s_kx*sqrt(ar), s_ky/sqrt(ar)])
                            boxes.append([cx, cy, s_kx/sqrt(ar), s_ky*sqrt(ar)])

                        for box in boxes:
                            mean += box
                        

        else:
            print('Not implemented')
            exit(-1)
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output