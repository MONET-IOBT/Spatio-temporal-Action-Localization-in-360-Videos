"""
    Copyright (c) 2017, Gurkirt Singh

    This code and is available
    under the terms of MIT License provided in LICENSE.
    Please retain this notice and LICENSE if you use
    this file (or any portion of it) in your project.
    ---------------------------------------------------------
"""

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import AnnotationTransform, UCF24Detection, BaseTransform, CLASSES, detection_collate
from data import v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13
from model.sph_ssd import build_vgg_ssd
from model.fpnssd.net import FPNSSD512
from model.vggssd.net import SSD512
from model.mobile_ssd_v1.net import MobileSSD512
from model.mobile_ssd_v2.net import MobileSSDLite300V2
from model.mobile_fpnssd.net import MobileFPNSSD512
import torch.utils.data as data
from layers.sph_box_utils import decode, nms
from utils.evaluation import evaluate_detections
import os, time
import argparse
import numpy as np
import pickle
import scipy.io as sio # to save detection as mat files
import socket
import sys
import cv2
import struct ## new
import zlib

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v5', help='The version of config')
parser.add_argument('--basenet', default='fpn_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--ssd_dim', default=512, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb can take flow as well')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--eval_iter', default='150000,', type=str, help='Number of training iterations')
parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--save_root', default='/home/bo/research/dataset/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')
parser.add_argument('--net_type', default='conv2d', help='conv2d or sphnet or ktn')


args = parser.parse_args()
all_versions = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13]
args.cfg = all_versions[int(args.version[-1])-1]
args.outshape = args.cfg['min_dim']
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)


if args.input_type != 'rgb':
    args.conf_thresh = 0.05

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test_net(net, save_root, exp_name, input_type, iteration, num_classes, thresh=0.5 ):
    HOST=''
    PORT=8485

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn,addr=s.accept()

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))

    while True:
        t1 = time.perf_counter()
        while len(data) < payload_size:
            # print("Recv: {}".format(len(data)))
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        t2 = time.perf_counter()

        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        images = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0)

        with torch.no_grad():

            if args.cuda:
                images = images.cuda()
            t3 = time.perf_counter()
            
            output = net(images)
            loc_data = output[0]
            conf_preds = output[1]
            prior_data = output[2]

        tf = time.perf_counter()
        fps = 1/(tf - t1) 
        print("Input shape:",images.shape,
            ', receive time {:0.3f}'.format(t2 - t1),
            ', decode time {:0.3f}'.format(t3 - t2),
            ', process time {:0.3f}'.format(tf - t3),
            ', total time {:0.3f}'.format(tf - t1),
            ', speed {:0.3f}'.format(fps))
            
    return 


def main():

    means = (104, 117, 123)  # only support voc now

    exp_name = '{}-SSD-{}-{}-bs-{}-{}-lr-{:05d}-{}'.format(args.net_type, args.dataset,
                args.input_type, args.batch_size, args.cfg['base'], int(args.lr*100000), args.cfg['name'])

    args.save_root += args.dataset+'/'
    args.listid = '01' ## would be usefull in JHMDB-21
    print('Exp name', exp_name, args.listid)
    for iteration in [int(itr) for itr in args.eval_iter.split(',') if len(itr)>0]:
        log_file = open(args.save_root + 'cache/' + exp_name + "/testing-{:d}.log".format(iteration), "w", 1)
        log_file.write(exp_name + '\n')
        trained_model_path = args.save_root + 'cache/' + exp_name + '/ssd300_ucf24_' + repr(iteration) + '.pth'
        log_file.write(trained_model_path+'\n')
        num_classes = len(CLASSES) + 1  #7 +1 background
        if args.cfg['base'] == 'fpn':
            assert(args.ssd_dim == 512)
            net = FPNSSD512(num_classes, args.cfg)
        elif args.cfg['base'] == 'vgg16':
            if args.cfg['min_dim'][0] == 512:
                net = SSD512(num_classes, args.cfg)
                net.loc_layers.apply(weights_init)
                net.cls_layers.apply(weights_init)
            else:
                net = build_vgg_ssd(num_classes, args.cfg)
                net.loc.apply(weights_init)
                net.conf.apply(weights_init)
        elif args.cfg['base'] == 'mobile_v1_512':
            net = MobileSSD512(num_classes, args.cfg)
        elif args.cfg['base'] == 'mobile_v2_300_lite':
            net = MobileSSDLite300V2(num_classes, args.cfg)
        elif args.cfg['base'] == 'fpn_mobile_512':
            net = MobileFPNSSD512(num_classes, args.cfg)
        elif args.cfg['base'] == 'yolov3':
            net = Darknet('model/yolov3/cfg/yolov3-spp.cfg', arc='CE')
        else:
            return 

        net.load_state_dict(torch.load(trained_model_path))
        net.eval()
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        print('Finished loading model %d !' % iteration)
        
        # evaluation
        torch.cuda.synchronize()
        log_file.write('Testing net \n')
        test_net(net, args.save_root, exp_name, args.input_type, iteration, num_classes)

if __name__ == '__main__':
    main()
