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
from data.omni_dataset import OmniUCF24
from data import AnnotationTransform, UCF24Detection, BaseTransform, CLASSES, detection_collate, v1,v2,v3,v4,v5,v6
from model.fpnssd.net import FPNSSD512
from model.sph_ssd import build_vgg_ssd
from model.vggssd.net import SSD512
import torch.utils.data as data
from layers.sph_box_utils import decode, nms
from utils.evaluation import evaluate_detections
import os, time
import argparse
import numpy as np
import pickle
import scipy.io as sio # to save detection as mat files

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v1', help='The version of config')
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
parser.add_argument('--data_root', default='/home/bo/research/dataset/', help='Location of VOC root directory')
parser.add_argument('--save_root', default='/home/bo/research/dataset/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')
parser.add_argument('--net_type', default='conv2d', help='conv2d or sphnet or ktn')


args = parser.parse_args()
all_versions = [v1,v2,v3,v4,v5,v6]
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
    """ Test a SSD network on an Action image database. """

    torch.cuda.synchronize()
    with torch.no_grad():
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        images = torch.randn(args.test_batch_size,3,*args.outshape)

        if args.cuda:
            images = images.cuda()
        output = net(images)
        loc_data = output[0]
        conf_preds = output[1]
        prior_data = output[2]

        print("Input:",images.shape)
        print("Output:",output[0].shape,output[1].shape)

        torch.cuda.synchronize()
        tf = time.perf_counter()
        print('Forward Time {:0.3f}'.format(tf - t1))
            
    return 


def main():

    means = (104, 117, 123)  # only support voc now

    exp_name = '{}-SSD-{}-{}-bs-{}-{}-lr-{:05d}-{}'.format(args.net_type, args.dataset,
                args.input_type, args.batch_size, args.cfg['base'], int(args.lr*100000), args.cfg['name'])

    args.save_root += args.dataset+'/'
    args.data_root += args.dataset+'/'
    args.listid = '01' ## would be usefull in JHMDB-21
    print('Exp name', exp_name, args.listid)
    for iteration in [int(itr) for itr in args.eval_iter.split(',') if len(itr)>0]:
        log_file = open(args.save_root + 'cache/' + exp_name + "/testing-{:d}.log".format(iteration), "w", 1)
        log_file.write(exp_name + '\n')
        trained_model_path = args.save_root + 'cache/' + exp_name + '/ssd300_ucf24_' + repr(iteration) + '.pth'
        log_file.write(trained_model_path+'\n')
        num_classes = len(CLASSES) + 1  #7 +1 background
        if args.cfg['base'] == 'fpn':
            net = FPNSSD512(num_classes, args.cfg)
        else:
            if args.cfg['min_dim'][0] == 512:
                net = SSD512(num_classes, args.cfg)
            else:
                net = build_vgg_ssd(num_classes, args.cfg)
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
