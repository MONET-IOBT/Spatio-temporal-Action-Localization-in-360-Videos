
""" Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Which was adopated by: Ellis Brown, Max deGroot
    https://github.com/amdegroot/ssd.pytorch

    Further:
    Updated by Gurkirt Singh for ucf101-24 dataset
    Licensed under The MIT License [see LICENSE for details]
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse
from data.omni_dataset import OmniUCF24
from data import AnnotationTransform, BaseTransform, UCF24Detection, detection_collate
from data import v1,v2,v3,v4,v5
import torch.utils.data as data
from utils.augmentations import SSDAugmentation
import numpy as np
import cv2
import io
import socket
import struct
import time
import pickle
import zlib

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='5', help='The version of config')
# parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--ssd_dim', default=512, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb options are [rgb,brox,fastOF]')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--max_epoch', default=6, type=int, help='Number of training epochs')
parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--data_root', default='/home/bo/research/dataset/', help='Location of VOC root directory')

## Parse arguments
args = parser.parse_args()
## set random seeds
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)


torch.set_default_tensor_type('torch.FloatTensor')

def main():
    all_versions = [v1,v2,v3,v4,v5]
    args.cfg = all_versions[int(args.version)-1]
    args.outshape = args.cfg['min_dim']
    args.means = (104, 117, 123)

    args.data_root += args.dataset + '/'

    start_streaming()

def start_streaming():
    print('Loading dataset')
    val_dataset = OmniUCF24(args.data_root, 'test', BaseTransform(300, args.means), AnnotationTransform(), 
                                        input_type=args.input_type, outshape=args.outshape, full_test=True)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, 1, num_workers=args.num_workers,
                                 shuffle=False, collate_fn=detection_collate, pin_memory=True)

    num_test_images = 50

    frames = []
    batch_iterator = None
    for _ in range(num_test_images):
        if not batch_iterator:
            batch_iterator = iter(val_data_loader)

        images, _, _ = next(batch_iterator)
        assert(images.shape[0] == 1)

        frame = images.squeeze(0).permute(1,2,0).numpy()

        frames.append(frame)

    print('Dataset loaded.')

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8485))
    connection = client_socket.makefile('wb')
    print('Connected.')

    img_counter = 0
    
    for frame in frames:

        result, frame = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(frame, 0)
        size = len(data)

        print("{}: {}".format(img_counter, size))
        client_socket.sendall(struct.pack(">L", size) + data)
        img_counter += 1

if __name__ == '__main__':
    main()
