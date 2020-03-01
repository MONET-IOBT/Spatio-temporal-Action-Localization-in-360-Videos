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
from data.omni_dataset import OmniUCF24,OmniJHMDB
from data import AnnotationTransform, UCF24Detection, JHMDB, BaseTransform, UCF24_CLASSES, JHMDB_CLASSES, detection_collate, v1,v2,v3,v4,v5
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
from PIL import ImageDraw,Image,ImageFont
import cv2
import socket
import struct
import zlib

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='2', help='The version of config')
parser.add_argument('--basenet', default='fpn_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--ssd_dim', default=512, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb can take flow as well')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--eval_iter', default='150000,', type=str, help='Number of training iterations')
parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--data_root', default='/home/picocluster/research/dataset/', help='Location of VOC root directory')
parser.add_argument('--save_root', default='/home/picocluster/research/dataset/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')
parser.add_argument('--net_type', default='conv2d', help='conv2d or sphnet or ktn')
parser.add_argument('--lossy', default=False, type=str2bool, help='Lossy image transmission')
parser.add_argument('--cache_size', default=150, type=int, help='cache size')

args = parser.parse_args()
all_versions = [v1,v2,v3,v4,v5]
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

CLASSES = UCF24_CLASSES if args.dataset == 'ucf24' else JHMDB_CLASSES

# use the score of a box if the overlaping > T
def score_of_edge(v1,v2,iouth):
    N2 = len(v2['boxes'])
    score = torch.zeros(1,N2)

    x1,y1,x2,y2 = v1['boxes'][-1,:]

    xx1 = torch.clamp(v2['boxes'][:,0],min=x1)
    yy1 = torch.clamp(v2['boxes'][:,1],min=y1)
    xx2 = torch.clamp(v2['boxes'][:,2],max=x2)
    yy2 = torch.clamp(v2['boxes'][:,3],max=y2)
    w = xx2-xx1
    h = yy2-yy1
    w = torch.clamp(w, min=0.0)
    h = torch.clamp(h, min=0.0)
    inter = w*h
    area = torch.mul(v2['boxes'][:,2]-v2['boxes'][:,0],v2['boxes'][:,3]-v2['boxes'][:,1])
    union = (x2-x1)*(y2-y1) + area - inter
    iou = inter/union
    assert(len(iou) == N2)

    for i in range(N2):
        if iou[i] >= iouth:
            score[0,i] = v2['scores'][i]
    return score


def getPathCount(live_paths):
    return len(live_paths)

def sort_live_paths(live_paths,path_order_score,dead_paths,dp_count,gap):
    sorted_live_paths = {}
    ind = np.argsort(-path_order_score)
    lpc = 0
    for lp in range(getPathCount(live_paths)):
        olp = ind[lp]
        if live_paths[olp]['lastfound'] < gap:
            sorted_live_paths[lpc] = {}
            sorted_live_paths[lpc]['boxes'] = live_paths[olp]['boxes']
            sorted_live_paths[lpc]['scores'] = live_paths[olp]['scores']
            sorted_live_paths[lpc]['allscores'] = live_paths[olp]['allscores']
            sorted_live_paths[lpc]['pathscore'] = live_paths[olp]['pathscore']
            sorted_live_paths[lpc]['foundat'] = live_paths[olp]['foundat']
            sorted_live_paths[lpc]['count'] = live_paths[olp]['count']
            sorted_live_paths[lpc]['lastfound'] = live_paths[olp]['lastfound']
            lpc += 1
        else:
            dead_paths[dp_count] ={}
            dead_paths[dp_count]['boxes'] = live_paths[olp]['boxes']
            dead_paths[dp_count]['scores'] = live_paths[olp]['scores']
            dead_paths[dp_count]['allscores'] = live_paths[olp]['allscores']
            dead_paths[dp_count]['pathscore'] = live_paths[olp]['pathscore']
            dead_paths[dp_count]['foundat'] = live_paths[olp]['foundat']
            dead_paths[dp_count]['count'] = live_paths[olp]['count']
            dead_paths[dp_count]['lastfound'] = live_paths[olp]['lastfound']
            dp_count += 1
    return sorted_live_paths,dead_paths,dp_count

def fill_gaps(path,gap):
    # fix this found at?
    gap_filled_paths = {}
    if len(path)>0:
        g_count = 0

        for lp in range(getPathCount(path)):
            if len(path[lp]['foundat']) > gap:
                gap_filled_paths[g_count] = {}
                gap_filled_paths[g_count]['start'] = path[lp]['foundat'][0]
                gap_filled_paths[g_count]['end'] = path[lp]['foundat'][-1]
                gap_filled_paths[g_count]['pathscore'] = path[lp]['pathscore']
                gap_filled_paths[g_count]['foundat'] = path[lp]['foundat']
                gap_filled_paths[g_count]['count'] = path[lp]['count']
                gap_filled_paths[g_count]['lastfound'] = path[lp]['lastfound']
                count = 0
                i = 0
                while i <= len(path[lp]['scores'])-1:
                    diff_found = path[lp]['foundat'][i] - path[lp]['foundat'][max(0,i-1)]
                    if count == 0:
                        gap_filled_paths[g_count]['boxes'] = path[lp]['boxes'][i,:].unsqueeze(0)
                        gap_filled_paths[g_count]['scores'] = path[lp]['scores'][i].unsqueeze(0)
                        gap_filled_paths[g_count]['allscores'] = path[lp]['allscores'][i,:].unsqueeze(0)
                        i += 1
                        count += 1
                    else:
                        for d in range(diff_found):
                            gap_filled_paths[g_count]['boxes'] = torch.cat((gap_filled_paths[g_count]['boxes'],
                                                                path[lp]['boxes'][i,:].unsqueeze(0)),0)
                            gap_filled_paths[g_count]['scores'] = torch.cat((gap_filled_paths[g_count]['scores'],
                                                                path[lp]['scores'][i].unsqueeze(0)),0)
                            gap_filled_paths[g_count]['allscores'] = torch.cat((gap_filled_paths[g_count]['allscores'],
                                                                path[lp]['allscores'][i,:].unsqueeze(0)),0)
                            count += 1
                        i += 1
                g_count += 1
    return gap_filled_paths

def sort_paths(live_paths):
    sorted_live_paths ={}

    lp_count = getPathCount(live_paths)
    if lp_count > 0:
        path_order_score = np.zeros(lp_count)

        for lp in range(len(live_paths)):
            scores,_ = torch.sort(live_paths[lp]['scores'],descending=True)
            num_sc = len(scores)
            path_order_score[lp] = torch.mean(scores[:min(20,num_sc)])

        ind = np.argsort(-path_order_score)

        for lpc in range(len(live_paths)):
            olp = ind[lpc]
            sorted_live_paths[lpc] = {}
            sorted_live_paths[lpc]['start'] = live_paths[olp]['start']
            sorted_live_paths[lpc]['end'] = live_paths[olp]['end']
            sorted_live_paths[lpc]['boxes'] = live_paths[olp]['boxes']
            sorted_live_paths[lpc]['scores'] = live_paths[olp]['scores']
            sorted_live_paths[lpc]['allscores'] = live_paths[olp]['allscores']
            sorted_live_paths[lpc]['pathscore'] = live_paths[olp]['pathscore']
            sorted_live_paths[lpc]['foundat'] = live_paths[olp]['foundat']
            sorted_live_paths[lpc]['count'] = live_paths[olp]['count']
            sorted_live_paths[lpc]['lastfound'] = live_paths[olp]['lastfound']
    return sorted_live_paths


def incremental_linking(frames,iouth, gap):
    num_frames = len(frames)

    live_paths = {}
    dead_paths = {}
    dp_count = 0

    for t in range(num_frames):
        num_box = len(frames[t]['boxes'])
        if t == 0:
            for b in range(num_box):
                live_paths[b] = {}
                live_paths[b]['boxes'] = frames[t]['boxes'][b,:].unsqueeze(0)
                live_paths[b]['scores'] = frames[t]['scores'][b].unsqueeze(0)
                live_paths[b]['allscores'] = frames[t]['allscores'][b,:].unsqueeze(0)
                live_paths[b]['pathscore'] = frames[t]['scores'][b].unsqueeze(0)
                live_paths[b]['foundat'] = [t]
                live_paths[b]['count'] = 1
                live_paths[b]['lastfound'] = 0
        else:
            lp_count = getPathCount(live_paths)

            if num_box > 0:
                edge_scores = torch.zeros(lp_count,num_box)

                for lp in range(lp_count):
                    edge_scores[lp,:] = score_of_edge(live_paths[lp],frames[t],iouth)

            dead_count = 0
            covered_boxes = np.zeros(num_box)
            path_order_score = np.zeros(lp_count)
            # maybe compare all live path-new box pairs
            for lp in range(lp_count):
                if live_paths[lp]['lastfound'] < gap:
                    # all box scores for a live path
                    if num_box > 0 and sum(edge_scores[lp,:]) > 0:
                        box_to_lp_score = edge_scores[lp,:]
                        maxInd = torch.argmax(box_to_lp_score)
                        m_score = box_to_lp_score[maxInd]
                        live_paths[lp]['count'] += 1
                        lpc = live_paths[lp]['count']
                        live_paths[lp]['boxes'] = torch.cat((live_paths[lp]['boxes'],
                                                    frames[t]['boxes'][maxInd,:].unsqueeze(0)),0)
                        live_paths[lp]['scores'] = torch.cat((live_paths[lp]['scores'],
                                                    frames[t]['scores'][maxInd].unsqueeze(0)),0)
                        live_paths[lp]['allscores'] = torch.cat((live_paths[lp]['allscores'],
                                                    frames[t]['allscores'][maxInd,:].unsqueeze(0)),0)
                        live_paths[lp]['pathscore'] += m_score
                        live_paths[lp]['foundat'] += [t]
                        live_paths[lp]['lastfound'] = 0
                        edge_scores[:,maxInd] = 0
                        covered_boxes[maxInd] = 1
                    else:
                        live_paths[lp]['lastfound'] += 1

                    scores,_ = torch.sort(live_paths[lp]['scores'])
                    num_sc = len(scores)
                    path_order_score[lp] = torch.mean(scores[max(0,num_sc-gap):num_sc])
                else:
                    dead_count += 1
            
            # sort paths after appending boxes
            live_paths,dead_paths,dp_count = sort_live_paths(live_paths,path_order_score,
                                                    dead_paths,dp_count,gap)
            lp_count = getPathCount(live_paths)

            if sum(covered_boxes) < num_box:
                for b in range(num_box):
                    if covered_boxes[b] == 0:
                        live_paths[lp_count] = {}
                        live_paths[lp_count]['boxes'] = frames[t]['boxes'][b,:].unsqueeze(0)
                        live_paths[lp_count]['scores'] = frames[t]['scores'][b].unsqueeze(0)
                        live_paths[lp_count]['allscores'] = frames[t]['allscores'][b,:].unsqueeze(0)
                        live_paths[lp_count]['pathscore'] = frames[t]['scores'][b].unsqueeze(0)
                        live_paths[lp_count]['foundat'] = [t]
                        live_paths[lp_count]['count'] = 1
                        live_paths[lp_count]['lastfound'] = 0
                        lp_count += 1
        
    live_paths = fill_gaps(live_paths,gap)
    dead_paths = fill_gaps(dead_paths,gap)
    lp_count = getPathCount(live_paths)
    if 'boxes' in dead_paths:
        for dp in range(len(dead_paths)):
            live_paths[lp_count] = {}
            live_paths[lp_count]['start'] = dead_paths[dp]['start']
            live_paths[lp_count]['end'] = dead_paths[dp]['end']
            live_paths[lp_count]['boxes'] = dead_paths[dp]['boxes']
            live_paths[lp_count]['scores'] = dead_paths[dp]['scores']
            live_paths[lp_count]['allscores'] = dead_paths[dp]['allscores']
            live_paths[lp_count]['pathscore'] = dead_paths[dp]['pathscore']
            live_paths[lp_count]['foundat'] = dead_paths[dp]['foundat']
            live_paths[lp_count]['count'] = dead_paths[dp]['count']
            live_paths[lp_count]['lastfound'] = dead_paths[dp]['lastfound']
            lp_count += 1

    live_paths = sort_paths(live_paths)

    return live_paths



def doFilter(video_result,a,f,nms_thresh):
    scores = video_result[f]['scores'][:,a].squeeze()
    c_mask = scores.gt(args.conf_thresh)
    # c_mask = scores.gt(0.001)
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.array([]),np.array([]),np.array([])
    boxes = video_result[f]['boxes'].clone()
    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
    boxes = boxes[l_mask].view(-1, 4)
    a_mask = c_mask.unsqueeze(1).expand_as(video_result[f]['scores'])
    allscores = video_result[f]['scores'][a_mask].view(-1,video_result[f]['scores'].shape[-1])
    ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
    scores = scores[ids[:counts]]
    boxes = boxes[ids[:counts]]
    allscores = allscores[ids[:counts]]
    height,width = args.cfg['min_dim']
    boxes[:, 0] *= width
    boxes[:, 2] *= width
    boxes[:, 1] *= height
    boxes[:, 3] *= height
    return boxes,scores,allscores


def genActionPaths(video_result, a, nms_thresh, iouth,gap):
    action_frames = {}
    t1 = time.perf_counter()
    for f in range(len(video_result)):
        # get decoded boxes of actual size
        boxes,scores,allscores = doFilter(video_result,a,f,nms_thresh)
        # if len(boxes.shape)>1 and a==1:
        #     print('filter',f,a,scores,boxes)
        action_frames[f] = {}
        action_frames[f]['boxes'] = boxes
        action_frames[f]['scores'] = scores
        action_frames[f]['allscores'] = allscores
    t2 = time.perf_counter()

    paths = incremental_linking(action_frames,iouth, gap)
    t3 = time.perf_counter()
    # print('Filter:{:0.3f},'.format(t2 - t1),
    #     'linking:{:0.3f}'.format(t3 - t2))
    return paths


# generate path from from frame-level detections
def actionPath(video_result):
    gap = 3
    iouth = 0.1
    numActions = len(CLASSES)
    nmsThresh = 0.45
    allPath = {}
    for a in range(1,numActions+1):
        allPath[a] = genActionPaths(video_result, a, nmsThresh, iouth,gap)
    return allPath

# construct a path that maximizes action socres
def dpEM_max(M,alpha):
    # action score of all frames
    r,c = M.shape # r numactions,c num frames

    D = np.zeros((r,c+1))
    D[:,0] = 0
    D[:,1:] = M.cpu().numpy()

    v = np.array([range(r)])

    phi = np.zeros((r,c))

    for j in range(1,c+1):
        for i in range(r):
            tmp = D[:,j-1]-alpha*(v!=i)
            tb = np.argmax(tmp)
            dmax = np.max(tmp)
            D[i,j] += dmax
            phi[i,j-1] = tb

    D = D[:,1:]

    q = [c-1]
    p = [np.argmax(D[:,-1])]

    i = p[-1]
    j = q[-1]
    
    while j>0:
        tb = phi[i,j]
        p = [tb] + p
        q = [j-1] + q
        j -= 1
        i = int(tb)
    return p,np.array(q),D

def extract_action(p,q,D,action):
    indexs = np.where(np.array(p) == action)[0]
    len_idx = len(indexs)

    if len(indexs) == 0:
        # no point on path is the specified action
        ts, te, scores, label, total_score = [],[],[],[],[]
    else:
        indexs_diff = np.concatenate((indexs,[indexs[-1]+1])) - np.concatenate(([indexs[0]-2],indexs))
        # find where the adjacent action is different
        ts = np.where(indexs_diff > 1)[0]

        if len(ts) > 1:
            te = ts[1:]-1
            te = np.concatenate((te,[len(indexs)-1]))
        else:
            te = [len(indexs)-1]

        ts = indexs[ts]
        te = indexs[te]
        # score of each tube
        scores = (D[action,q[te]] - D[action,q[ts]]) / (te - ts)
        label = [action for _ in range(len(ts))]
        total_score = np.ones(len(ts)) * D[p[-1],q[-1]] / len(p)
    return ts,te,scores,label,total_score


def actionPathSmoother4oneVideo(video_paths,alpha,num_action):
    final_tubes = {}
    final_tubes['starts'] = {}
    final_tubes['ts'] = {}
    final_tubes['te'] = {}
    final_tubes['label'] = {}
    final_tubes['path_total_score'] = {}
    final_tubes['dpActionScore'] = {}
    final_tubes['dpPathScore'] = {}
    final_tubes['path_boxes'] = {}
    final_tubes['path_scores'] = {}
    action_count = 0

    if len(video_paths) > 0:
        for a in range(1,num_action+1):
            action_paths = video_paths[a]
            num_act_paths = getPathCount(action_paths)
            for p in range(num_act_paths):
                M = action_paths[p]['allscores'].transpose(1,0)
                assert(len(M.shape) == 2)
                M += 20

                # refine the path 
                pred_path,time,D = dpEM_max(M,alpha)
                Ts, Te, Scores, Label, DpPathScore = extract_action(pred_path,time,D,a)

                # print("Num tubes for action",a,len(Ts))
                for k in range(len(Ts)):
                    final_tubes['starts'][action_count] = action_paths[p]['start']
                    final_tubes['ts'][action_count] = Ts[k]
                    final_tubes['te'][action_count] = Te[k]
                    final_tubes['dpActionScore'][action_count] = Scores[k]
                    final_tubes['label'][action_count] = Label[k]
                    final_tubes['dpPathScore'][action_count] = DpPathScore[k]
                    final_tubes['path_total_score'][action_count] = torch.mean(action_paths[p]['scores'])
                    final_tubes['path_boxes'][action_count] = action_paths[p]['boxes']
                    final_tubes['path_scores'][action_count] = action_paths[p]['scores']
                    action_count += 1
    return final_tubes


def actionPathSmoother(allPath,alpha,num_action):
    final_tubes = actionPathSmoother4oneVideo(allPath,alpha,num_action)
    return final_tubes

def convert2eval(final_tubes,min_num_frames,topk):
    xmld = {}
    xmld['score'] = {}
    xmld['nr'] = {}
    xmld['class'] = {}
    xmld['framenr'] = {}
    xmld['boxes'] = {}

    action_score = final_tubes['dpActionScore']
    path_score = final_tubes['path_scores']

    ts = final_tubes['ts']
    starts = final_tubes['starts']
    te = final_tubes['te']

    act_nr = 0

    for a in range(len(ts)):
        act_ts = ts[a]
        act_te = te[a]
        act_path_scores = path_score[a]

        act_scores,_ = torch.sort(act_path_scores[act_ts:act_te+1],descending=True)

        topk_mean = torch.mean(act_scores[:min(topk,len(act_scores))])

        bxs = final_tubes['path_boxes'][a][act_ts:act_te+1,:]

        label = final_tubes['label'][a]

        if topk_mean > 0 and (act_te-act_ts) > min_num_frames:
            xmld['score'][act_nr] = topk_mean
            xmld['nr'][act_nr] = act_nr
            xmld['class'][act_nr] = label
            xmld['framenr'][act_nr] = {'fnr':np.array(range(act_ts,act_te+1)) + starts[a]}
            xmld['boxes'][act_nr] = {'bxs':bxs}
            act_nr += 1
    return xmld

def sort_detection(dt_tubes):
    sorted_tubes = {}
    sorted_tubes['score'] = {}
    sorted_tubes['nr'] = {}
    sorted_tubes['class'] = {}
    sorted_tubes['framenr'] = {}
    sorted_tubes['boxes'] = {}

    num_detection = len(dt_tubes['class'])
    if num_detection > 0:
        scores = dt_tubes['score']
        indexes = [k for k, _ in sorted(scores.items(), key=lambda item: -item[1])]
        for dt in range(num_detection):
            dtind = indexes[dt]
            sorted_tubes['framenr'][dt] = {'fnr':dt_tubes['framenr'][dtind]['fnr']}
            sorted_tubes['boxes'][dt] = {'bxs':dt_tubes['boxes'][dtind]['bxs']}
            sorted_tubes['class'][dt] = dt_tubes['class'][dtind]
            sorted_tubes['score'][dt] = dt_tubes['score'][dtind]
            sorted_tubes['nr'][dt] = dt
    return sorted_tubes

def inters_union(bounds1,bounds2):
    box_a = torch.Tensor(bounds1)
    box_b = torch.Tensor(bounds2)
    max_xy = torch.min(box_a[2:],box_b[2:])
    min_xy = torch.max(box_a[:2],box_b[:2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[0] * inter[1]

    area_a = ((box_a[2]-box_a[0])*(box_a[3]-box_a[1]))
    area_b = ((box_b[2]-box_b[0])*(box_b[3]-box_b[1]))
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def compute_spatial_temporal_iou(gt_fnr,gt_bb,dt_fnr,dt_bb):
    tgb = gt_fnr[0]
    tge = gt_fnr[-1]
    tdb = dt_fnr[0]
    tde = dt_fnr[-1]

    T_i = max(0,min(tge,tde)-max(tgb,tdb))

    if T_i > 0:
        T_i += 1
        T_u = max(tge,tde) - min(tgb,tdb) + 1
        T_iou = T_i/T_u
        int_fnr = range(max(tgb,tdb),min(tge,tde)+1)
        int_find_dt = []
        for i in range(len(dt_fnr)):
            if dt_fnr[i] in int_fnr:
                int_find_dt.append(i)
        int_find_gt = []
        for i in range(len(gt_fnr)):
            if gt_fnr[i] in int_fnr:
                int_find_gt.append(i)
        assert(len(int_find_gt) == len(int_find_dt))

        iou = np.zeros(len(int_find_dt))
        for i in range(len(int_find_dt)):
            gt_bound = gt_bb[int_find_gt[i],:]
            dt_bound = dt_bb[int_find_dt[i],:]
            
            iou[i] = inters_union(gt_bound,dt_bound)
        st_iou = T_iou*np.mean(iou)
    else:
        st_iou = 0
    return st_iou

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # print('voc_ap() - use_07_metric:=' + str(use_07_metric))
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# count of detected tubes per class
cc = [0 for _ in range(len(CLASSES))]
# result for each detected tube per class
allscore = {}
for a in range(len(CLASSES)):
    allscore[a] = np.zeros((10000,2))
# num of gt tubes per class
total_num_gt_tubes = [0 for _ in range(len(CLASSES))]
# avg iou per class
averageIoU = np.zeros(len(CLASSES))
preds = []
gts = []

def get_PR_curve(annot, xmldata, iouth):
    numActions = len(CLASSES)
    maxscore = -10000
    # annotName = annot[1][0]
    action_id = annot[2][0][0][2] - 1

    gt_tubes = annot[2][0]
    dt_tubes = sort_detection(xmldata)

    num_detection = len(dt_tubes['class'])
    num_gt_tubes = len(gt_tubes)

    for gtind in range(num_gt_tubes):
        action_id = gt_tubes[gtind][2][0][0]
        assert(action_id > 0)
        action_id -= 1
        total_num_gt_tubes[action_id] += 1

    pred = -1
    gt = action_id
    dt_labels = dt_tubes['class']
    covered_gt_tubes = np.zeros(num_gt_tubes)
    for dtind in range(num_detection):
        # frame number range
        dt_fnr = dt_tubes['framenr'][dtind]['fnr']
        # bounding boxes
        dt_bb = dt_tubes['boxes'][dtind]['bxs']
        # class label
        dt_label = dt_labels[dtind] - 1
        # the tube having the max score decides
        # the label of the video
        if dt_tubes['score'][dtind] > maxscore:
            pred = dt_label
            maxscore = dt_tubes['score'][dtind]
        # cc counts the number of detections per class
        cc[dt_label] += 1
        assert(cc[dt_label]<10000)

        ioumax = -10000
        maxgtind = 0
        for gtind in range(num_gt_tubes):
            action_id = gt_tubes[gtind][2] - 1#class
            # if this gt tube is not covered and has the same label as this detected tube
            if (not covered_gt_tubes[gtind]) and dt_label == action_id:
                gt_fnr = range(gt_tubes[gtind][0][0][0] - gt_tubes[gtind][1][0][0] + 1)
                gt_bb = gt_tubes[gtind][3]
                iou = compute_spatial_temporal_iou(gt_fnr,gt_bb,dt_fnr,dt_bb)
                if iou > ioumax:
                    # find the best possible gttube based on stiou
                    ioumax = iou
                    maxgtind = gtind

        if ioumax > iouth:
            covered_gt_tubes[gtind] = 1
            # records the score,T/F of each dt tube at every step for every class
            allscore[dt_label][cc[dt_label],:] = [dt_tubes['score'][dtind],1]
            # max iou with rest gt tubes
            averageIoU[dt_label] += ioumax
        else:
            allscore[dt_label][cc[dt_label],:] = [dt_tubes['score'][dtind],0]
    preds.append(pred)
    gts.append(gt)
    return pred == gt,gt

def evaluate_tubes(outfile):
    actions = CLASSES
    numActions = len(CLASSES)
    AP = np.zeros(numActions)
    AIoU = np.zeros(numActions)
    # todo: need to store all detection info of all videos into allscore
    tmpscore = {}
    for a in range(numActions):
        tmpscore[a] = allscore[a][:cc[a],:].copy()
        scores = tmpscore[a][:,0]
        result = tmpscore[a][:,1]
        si = np.argsort(-scores)
        result = result[si]
        fp = np.cumsum(result == 0)
        tp = np.cumsum(result == 1)
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        # need to calculate AUC
        cdet = 0
        if len(tp) > 0:
            cdet = int(tp[-1])
            AIoU[a] = (averageIoU[a]+0.000001)/(cdet+0.000001) if cdet > 1 else averageIoU[a]

        recall = tp/float(total_num_gt_tubes[a]+1)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        AP[a] = voc_ap(recall,precision)
        ptr_str = 'Action {:02d} AP = {:0.5f} and AIOU {:0.5f}\
             GT {:03d} total det {:02d} correct det {:02d} {:s}\n'\
             .format(a, AP[a],AIoU[a],total_num_gt_tubes[a],cc[a],cdet,actions[a])
        print(ptr_str)
        outfile.write(ptr_str)

    acc = np.mean(np.array(preds)==np.array(gts))
    mAP = np.mean(AP)
    mAIoU = np.mean(AIoU)

    ptr_str = 'Mean AP {:0.2f} meanAIoU {:0.3f} accuracy {:0.3f}\n'.format(mAP,mAIoU,acc)
    print(ptr_str)
    outfile.write(ptr_str)

    return mAP,mAIoU,acc,AP


# smooth tubes and evaluate them
def getTubes(allPath,video_id,annot_map):
    # read all groundtruth actions
    final_annot_location = args.data_root + 'splitfiles/finalAnnots.mat'
    annot = sio.loadmat(final_annot_location)
    annot = annot['annot'][0][video_id]

    if args.dataset == 'ucf24':
        # transform the annotation
        for tid,tube in enumerate(annot[2][0]):
            new_boxes = None
            for i,old_box in enumerate(tube[3]):
                key = (old_box[0],old_box[1],old_box[2],old_box[3])
                if key not in annot_map:
                    exit(0)
                new_box = torch.Tensor(annot_map[key]).unsqueeze(0)
                if new_boxes is None:
                    new_boxes = new_box
                else:
                    new_boxes = torch.cat((new_boxes,new_box),0)
            annot[2][0][tid][3] = new_boxes

    # smooth action path
    alpha = 3
    numActions = len(CLASSES)
    smoothedtubes = actionPathSmoother(allPath,alpha,numActions)

    min_num_frames = 8
    topk = 40

    xmldata = convert2eval(smoothedtubes, min_num_frames, topk)

    # evaluate
    # iouths = [0.2] + [0.5 + 0.05*i for i in range(10)]
    iouth = args.iou_thresh
    return get_PR_curve(annot, xmldata, iouth),xmldata

def drawTubes(xmldata,output_dir,frames,gt_label):
    dt_tubes = sort_detection(xmldata)
    num_detection = len(dt_tubes['class'])
    dt_labels = dt_tubes['class']

    for frame in frames:
        for ch in range(0,3):
            frame[ch,:,:] += args.means[2-ch]

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    for dtind in range(num_detection):
        # frame number range
        dt_fnr = dt_tubes['framenr'][dtind]['fnr']
        # bounding boxes
        dt_bb = dt_tubes['boxes'][dtind]['bxs']
        # class label
        dt_label = dt_labels[dtind] - 1
        assert(dt_label>=0)

        for fn,bb in zip(dt_fnr,dt_bb):
            x = frames[fn]

            img = Image.fromarray(x.cpu().permute(1, 2, 0).numpy().astype(np.uint8))
            draw = ImageDraw.Draw(img)
            x1,y1,x2,y2 = bb
            draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline ="red")
            ft = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 26)
            draw.text((x1, y1),CLASSES[dt_label],(255,255,255),font=ft)

            npimg = np.array(img)

            result, frame = cv2.imencode('.jpg', npimg, encode_param)
            data = pickle.dumps(frame, 0)
            size = len(data)

            args.client_socket.sendall(struct.pack(">L", size) + 
                                        struct.pack(">L", dt_label) + 
                                        struct.pack(">L", gt_label) + data)
            break

def process_video_result(video_result,outfile,iteration,annot_map):
    frame_det_res = video_result['data']
    videoname = video_result['videoname']
    video_id = video_result['video_id']
    frames = video_result['frame']

    frame_save_dir = args.save_root+'detections/CONV-rgb-'+args.listid+'-'+str(iteration).zfill(6)+'/'
    output_dir = frame_save_dir+videoname
    print("Processing:",videoname,'id=',video_id,"total frames:",len(frame_det_res))

    t1 = time.perf_counter()
    allPath = actionPath(frame_det_res)

    t2 = time.perf_counter()
    tmp,xmldata = getTubes(allPath,video_id,annot_map)
    res,gt_label = tmp

    t3 = time.perf_counter()
    drawTubes(xmldata,output_dir,frames,gt_label)

    tf = time.perf_counter()

    print('Gen path {:0.3f}'.format(t2 - t1),
        ', gen tubes {:0.3f}'.format(t3 - t2),
        ', draw tubes {:0.3f}'.format(tf - t3),
        ', total time {:0.3f}'.format(tf - t1))
    # print("Detecting event:",videoname,
    #     'total time {:0.3f}, time per frame {:0.3f}'.format(tf - t1,(tf-t1)/len(frame_det_res)),
    #     'Success' if res else 'Failure')

def update_annot_map(annot_map,old_labels,new_labels):
    # record transform
    for old,new in zip(old_labels,new_labels):
        old2 = (int(old[0]),int(old[1]),int(old[2]-old[0]),int(old[3]-old[1]))
        if sum(old2) == 0:continue
        annot_map[old2] = [int(new[0]),int(new[1]),int(new[2]),int(new[3])]

def test_net(net, save_root, exp_name, input_type, dataset, iteration, num_classes, outfile, thresh=0.5 ):
    """ Test a SSD network on an Action image database. """

    # val_data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
    #                         shuffle=False, collate_fn=detection_collate, pin_memory=True)
    image_ids = dataset.ids
    save_ids = []
    val_step = 250
    num_images = len(dataset)
    video_list = dataset.video_list
    det_boxes = [[] for _ in range(len(CLASSES))]
    gt_boxes = []
    print_time = True
    batch_iterator = None
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    # num_batches = len(val_data_loader)
    num_batches = len(dataset)
    det_file = save_root + 'cache/' + exp_name + '/detection-'+str(iteration).zfill(6)+'.pkl'
    frame_save_dir = save_root+'detections/CONV-'+input_type+'-'+args.listid+'-'+str(iteration).zfill(6)+'/'

    print('Caching dataset...')
    cache_size = args.cache_size
    cached_data = []
    for i in range(cache_size):
        cached_data.append(dataset[i])
        if i>0 and i%(cache_size/10)==0:
            print(i/cache_size*100,'%')
    print('Data cached')

    # connect to server
    args.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    args.client_socket.connect(('127.0.0.1', 8485))
    connection = args.client_socket.makefile('wb')
    print('Connected.')

    # id identify diffferent video segments
    pre_video_id = -1
    # list to store all results of a single video
    video_result = {}
    video_result['data'] = []
    video_result['frame'] = []
    annot_map = {}
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    tube_t_s = time.perf_counter()
    with torch.no_grad():
        val_itr = 0
        while True:

            torch.cuda.synchronize()

            image, target, img_index = cached_data[val_itr]
            print(val_itr)

            images = torch.stack([image], 0)
            targets = [torch.FloatTensor(target)]
            img_indexs = [img_index]

            batch_size = images.size(0)
            height, width = images.size(2), images.size(3)

            if args.cuda:
                images = images.cuda()

            t1 = time.perf_counter()
            output = net(images)

            loc_data = output[0]
            conf_preds = output[1]
            prior_data = output[2]
            tf = time.perf_counter()
            print('Forward Time {:0.3f}'.format(tf - t1))
            continue

            for b in range(batch_size):
                gt = targets[b].numpy()
                gt[:, 0] *= width
                gt[:, 2] *= width
                gt[:, 1] *= height
                gt[:, 3] *= height
                # print(gt)
                gt_boxes.append(gt)
                decoded_boxes = decode(loc_data[b].data, prior_data.data, args.cfg['variance']).clone()
                conf_scores = net.softmax(conf_preds[b]).data.clone()
                index = img_indexs[b]
                annot_info = image_ids[index]

                frame_num = annot_info[1]; video_id = annot_info[0]; videoname = video_list[video_id]
                # check if this id is different from the previous one
                if (video_id != pre_video_id) and (len(video_result['data']) > 0):
                    # process this video
                    video_result['videoname'] = video_list[pre_video_id]
                    video_result['video_id'] = pre_video_id
                    tube_t_e = time.perf_counter()
                    fps = len(video_result['frame'])/(tube_t_e - tube_t_s)
                    print("Frame-level detection FPS:",fps)
                    process_video_result(video_result,outfile,iteration,annot_map)
                    annot_map = {}
                    tube_t_s = time.perf_counter()
                    video_result['data'] = []
                    video_result['frame'] = []
                if args.dataset == 'ucf24':
                    update_annot_map(annot_map,image_ids[index][3],gt)
                pre_video_id = video_id

                res = {}
                res['scores'] = conf_scores
                res['boxes'] = decoded_boxes
                video_result['data'].append(res)
                video_result['frame'].append(images[b])

                count += 1

            val_itr = (val_itr+1)%len(cached_data)
            if val_itr == 0:
                pre_video_id = -1
                video_result = {}
                video_result['data'] = []
                video_result['frame'] = []
                annot_map = {}
                tube_t_s = time.perf_counter()

    return


def main():

    args.means = (104, 117, 123)  # only support voc now

    exp_name = '{}-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.net_type, args.dataset,
                args.input_type, args.batch_size, args.cfg['base'], int(args.lr*100000))

    args.save_root += args.dataset+'/'
    args.data_root += args.dataset+'/'
    args.listid = '01' ## would be usefull in JHMDB-21
    print('Exp name', exp_name, args.listid)

    # 
    for iteration in [int(itr) for itr in args.eval_iter.split(',') if len(itr)>0]:
        log_file = open(args.save_root + 'cache/' + exp_name + "/testing-{:d}-{:0.2f}.log".format(iteration,args.iou_thresh), "w", 1)
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
        # Load dataset
        if args.dataset == 'ucf24':
            dataset = OmniUCF24(args.data_root, 'test', BaseTransform(300, args.means), AnnotationTransform(), 
                                    input_type=args.input_type, outshape=args.outshape, full_test=True)
        else:
            dataset = OmniJHMDB(args.data_root, 'test', BaseTransform(300, None), AnnotationTransform(), 
                                outshape=args.outshape)

        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        test_net(net, args.save_root, exp_name, args.input_type, dataset, iteration, num_classes, log_file)

        log_file.close()

if __name__ == '__main__':
    main()
