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
parser.add_argument('--version', default='v6', help='The version of config')
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
    if 'boxes' in live_paths:
        return len(live_paths['boxes'])
    return 0

def sort_live_path(live_paths,path_order_score,dead_paths,dp_count,gap):
    sorted_live_paths = {}
    ind = torch.argsort(-path_order_score)
    lpc = 0
    for lp in range(getPathCount(live_paths)):
        olp = ind[0,lp]
        if live_paths[0,olp]['lastfound'] < gap:
            sorted_live_paths[lpc]['boxes'] = live_paths[olp]['boxes']
            sorted_live_paths[lpc]['scores'] = live_paths[olp]['scores']
            sorted_live_paths[lpc]['allScores'] = live_paths[olp]['allScores']
            sorted_live_paths[lpc]['pathScore'] = live_paths[olp]['pathScore']
            sorted_live_paths[lpc]['foundAT'] = live_paths[olp]['foundAT']
            sorted_live_paths[lpc]['count'] = live_paths[olp]['count']
            sorted_live_paths[lpc]['lastfound'] = live_paths[olp]['lastfound']
            lpc += 1
        else:
            dead_paths[dp_count]['boxes'] = live_paths[olp]['boxes']
            dead_paths[dp_count]['scores'] = live_paths[olp]['scores']
            dead_paths[dp_count]['allScores'] = live_paths[olp]['allScores']
            dead_paths[dp_count]['pathScore'] = live_paths[olp]['pathScore']
            dead_paths[dp_count]['foundAT'] = live_paths[olp]['foundAT']
            dead_paths[dp_count]['count'] = live_paths[olp]['count']
            dead_paths[dp_count]['lastfound'] = live_paths[olp]['lastfound']
            dp_count += 1
    return sorted_live_paths,dead_paths,dp_count

def fill_gaps(paths.gap):
    gap_filled_paths = {}
    if 'boxes' in paths:
        g_count = 0

        for lp in range(getPathCount(paths)):
            if len(paths[lp]['foundAt']) > gap:
                gap_filled_paths[g_count]['start'] = path[lp]['foundAt'][0]
                gap_filled_paths[g_count]['end'] = path[lp]['foundAt'][-1]
                gap_filled_paths[g_count]['pathScore'] = path[lp]['pathScore']
                gap_filled_paths[g_count]['foundAt'] = path[lp]['foundAt']
                gap_filled_paths[g_count]['count'] = path[lp]['count']
                gap_filled_paths[g_count]['lastfound'] = path[lp]['lastfound']
                count = 0
                i = 0
                while i <= len(path[lp]['scores'])-1:
                    diff_found = path[lp]['foundAt'][i] - path[lp]['foundAt'][max(0,i-1)]
                    if count == 0 or diff_found == 1:
                        gap_filled_paths[g_count]['boxes'][count,:] = path[lp]['boxes'][i,:]
                        gap_filled_paths[g_count]['scores'][count] = path[lp]['scores'][i]
                        gap_filled_paths[g_count]['allscores'][count,:] = path[lp]['allscores'][i,:]
                        i += 1
                        count += 1
                    else:
                        for d in range(diff_found):
                            gap_filled_paths[g_count]['boxes'][count,:] = path[lp]['boxes'][i,:]
                            gap_filled_paths[g_count]['scores'][count] = path[lp]['scores'][i]
                            gap_filled_paths[g_count]['allscores'][count,:] = path[lp]['allscores'][i,:]
                            count += 1
                        i += 1
                g_count += 1
    return gap_filled_paths

def sort_paths(live_paths):
    sorted_live_paths ={}

    lp_count = getPathCount(live_paths)
    if lp_count > 0:
        path_order_score = torch.zeros(1,lp_count)

        for lp in range(len(live_paths)):
            scores = -sorted(-live_paths[lp]['scores'])
            num_sc = len(scores)
            path_order_score[lp] = torch.mean(scores[:min(20,num_sc)])

        ind = torch.argsort(-path_order_score)

        for lpc in range(len(live_paths)):
            olp = ind[0,lpc]
            sorted_live_paths[lpc]['start'] = live_paths[olp]['start']
            sorted_live_paths[lpc]['end'] = live_paths[olp]['end']
            sorted_live_paths[lpc]['boxes'] = live_paths[olp]['boxes']
            sorted_live_paths[lpc]['scores'] = live_paths[olp]['scores']
            sorted_live_paths[lpc]['allscores'] = live_paths[olp]['allscores']
            sorted_live_paths[lpc]['pathscore'] = live_paths[olp]['pathscore']
            sorted_live_paths[lpc]['foundAt'] = live_paths[olp]['foundAt']
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
                live_paths[b]['boxes'] = frames[t]['boxes'][b,:]
                live_paths[b]['scores'] = frames[t]['scores'][b]
                live_paths[b]['allscores'] = frames[t]['allscores'][b,:]
                live_paths[b]['pathscore'] = frames[t]['score'][b]
                live_paths[b]['foundAt'] = [1]
                live_paths[b]['count'] = 1
                live_paths[b]['lastfound'] = 0
        else:
            lp_count = getPathCount(live_paths)

            edge_scores = torch.zeros(lp_count,num_box)

            for lp in range(lp_count):
                edge_scores[lp,:] = score_of_edge(live_paths[lp],frames[t],iouth)

            dead_count = 0
            covered_boxes = torch.zeros(1,num_box)
            path_order_score = torch.zeros(1,lp_count)
            for lp in range(lp_count):
                if live_paths[lp]['lastfound'] < gap:
                    box_to_lp_score = edge_scores[lp,:]
                    if sum(box_to_lp_score) > 0:
                        maxInd = torch.argmax(box_to_lp_score)
                        m_score = box_to_lp_score[maxInd]
                        live_paths[lp]['count'] += 1
                        lpc = live_paths[lp]['count']

                        live_paths[lp]['scores'] = torch.cat((live_paths[lp]['scores'],
                                                    frames[t]['boxes'][maxInd,:]),0)
                        live_paths[lp]['allscores'] = torch.cat((live_paths[lp]['allscores'],
                                                    frames[t]['allscores'][maxInd,:]),0)
                        live_paths[lp]['pathscore'] += m_score
                        live_paths[lp]['foundAt'] += [t]
                        live_paths[lp]['lastfound'] = 0
                        edge_scores[:,maxInd] = 0
                        covered_boxes[0,maxInd] = 1
                    else:
                        live_paths[b]['lastfound'] += 1

                    scores = sorted(live_paths[lp]['scores'])
                    num_sc = len(scores)
                    path_order_score[0,lp] = mean(scores[max(0,num_sc-gap):num_sc])

                else:
                    dead_count += 1
            live_paths,dead_paths,dp_count = sort_live_paths(live_paths,path_order_score,
                                                    dead_paths,dp_count,gap)
            lp_count = getPathCount(live_paths)

            if torch.sum(covered_boxes) < num_box:
                for b in range(num_box):
                    if covered_boxes[0,b] == 0:
                        live_paths[b]['boxes'] = frames[t]['boxes'][b,:]
                        live_paths[b]['scores'] = frames[t]['scores'][b]
                        live_paths[b]['allscores'] = frames[t]['allscores'][b,:]
                        live_paths[b]['pathscore'] = frames[t]['score'][b]
                        live_paths[b]['foundAt'] = [t]
                        live_paths[b]['count'] = 1
                        live_paths[b]['lastfound'] = 0
                        lp_count += 1
    live_paths = fill_gaps(live_paths,gap)
    dead_paths = fill_gaps(dead_paths,gap)
    lp_count = getPathCount(live_paths)
    lp += 1
    if 'boxes' in dead_paths:
        for dp in range(len(dead_paths)):
            live_paths[lp]['start'] = dead_paths[dp]['start']
            live_paths[lp]['end'] = dead_paths[dp]['end']
            live_paths[lp]['boxes'] = dead_paths[dp]['boxes']
            live_paths[lp]['scores'] = dead_paths[dp]['scores']
            live_paths[lp]['allscores'] = dead_paths[dp]['allscores']
            live_paths[lp]['pathscore'] = dead_paths[dp]['pathscore']
            live_paths[lp]['foundAt'] = dead_paths[dp]['foundAt']
            live_paths[lp]['count'] = dead_paths[dp]['count']
            live_paths[lp]['lastfound'] = dead_paths[dp]['lastfound']
            lp += 1

    live_paths = sort_paths(live_paths)

    return live_paths



def doFilter(video_result,a,f,nms_thresh):
    scores = video_result(f)['scores'][:,a]
    pick = scores>0.001
    scores = scores[pick]
    boxes = video_result(f)['boxes'][pick,:]
    allscores = video_result(f)['scores'][pick,:]
    pick = torch.argsort(-scores)
    topick = min(50,len(pick))
    pick = pick[:topick]
    scores = scores[pick]
    boxes = boxes[pick,:]
    allscores = allscores[pick,:]
    pick,_ = nms(boxes, scores, args.nms_thresh, args.topk)
    pick = pick[:min(10,len(pick))]
    boxes = boxes[pick,:]
    scores = scores[pick]
    allscores = allscores[pick,:]
    return boxes,scores,allscores


def genActionPaths(video_result, a, nms_thresh, iouth,gap):
    action_frames = {}
    for f in range(len(video_result)):
        boxes,scores,allscores = doFilter(video_result,a,f,nms_thresh)
        action_frames[f] = {}
        action_frames[f]['boxes'] = boxes
        action_frames[f]['scores'] = scores
        action_frames[f]['allscores'] = allscores

    paths = incremental_linking(action_frames,iouth, gap)
    return paths


# generate path from from frame-level detections
def actionPath(video_result):
    opt = {}
    opt['actions'] = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
        'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
        'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
        'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
        'VolleyballSpiking','WalkingWithDog']
    gap = 3
    iouth = 0.1
    numActions = len(opt['actions'])
    nmsThresh = 0.45
    allPath = []
    for a in range(numActions):
        allPath.append(genActionPaths(video_result, a, nms_thresh, iouth,gap))
    return allPath

def dpEM_max(M,alpha):
    r,c = M.shape

    D = np.zeros(r,c+1)
    D[:,0] = 0
    D[:,1:] = M

    v = np.array([range(r)]).transpose(1,0)

    phi = np.zeros(r,c)

    for j in range(1,c+1):
        for i in range(r):
            tb = np.argmax(D[:,j-1]-alpha*(v!=i))
            dmax = np.max(D[:,j-1]-alpha*(v!=i))
            D[i,j] += dmax
            phi[i,j-1] = tb

    D = D(:,1:)

    q = c
    p = np.argmax(D[:,-1])

    i = p
    j = q-1

    while j>0:
        tb = phi[i,j]
        p = [tb] + p
        q = [j-1] + q
        j -= 1
        i = tb
    return p,q,D

def extract_action(p,q,D,action):
    indexs = np.where(np.array(p) == action)

    if len(indexs) == 0:
        ts, te, scores, label, total_score = [],[],[],[],[]
    else:
        indexs_diff = np.array(indexs+[indexs[-1]+1]) - np.array([indexs[0]-2]+indexs)
        ts = np.where(indexs_diff > 1)

        if len(ts) > 1:
            te = [ts[1:]-1,len(indexs)]
        else:
            te = len(indexs)

        ts = indexs[ts]
        te = indexs[te]
        scores = (D[action,q[te]] - D[action,q[ts]]) / (te - ts)
        label = np.ones(len(ts),1) * action
        total_score = np.ones(len(ts),1) * D(p[-1],q[-1]) / len(p)
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
        for a in range(num_action):
            action_paths = video_paths[a]
            num_act_paths = getPathCount(action_paths)
            for p in range(num_act_paths):
                M = action_paths[p]['allscores'][:,:num_action]
                M += 20

                pred_path,time,D = dpEM_max(M,alpha[a])
                Ts, Te, Scores, Label, DpPathScore = extract_action(pred_path,time,D,a)

                for k in range(len(Ts)):
                    final_tubes['starts'][action_count] = action_paths[p]['start']
                    final_tubes['ts'][action_count] = Ts[k]
                    final_tubes['te'][action_count] = Te[k]
                    final_tubes['dpActionScore'][action_count] = Scores[k]
                    final_tubes['label'][action_count] = Label[k]
                    final_tubes['dpPathScore'][action_count] = DpPathScore[k]
                    final_tubes['path_total_score'][action_count] = mean(action_paths[p]['scores'])
                    final_tubes['path_boxes'][action_count] = action_paths[p]['boxes']
                    final_tubes['path_scores'][action_count] = action_paths[p]['scores']
                    action_count += 1
    return final_tubes


def actionPathSmoother(allPath,alpha,num_action):
    # final_tubes = {}
    # final_tubes['starts'] = {}
    # final_tubes['ts'] = {}
    # final_tubes['te'] = {}
    # final_tubes['label'] = {}
    # final_tubes['path_total_score'] = {}
    # final_tubes['dpActionScore'] = {}
    # final_tubes['dpPathScore'] = {}
    # final_tubes['path_boxes'] = {}
    # final_tubes['path_scores'] = {}

    final_tubes = actionPathSmoother4oneVideo(allPath,alpha,num_action)

    # action_count = 0
    # for k in range(len(alltubes.ts)):
    #     final_tubes['starts'][action_count] = vid_tubes['starts'][k]
    #     final_tubes['ts'][action_count] = vid_tubes['ts'][k]
    #     final_tubes['te'][action_count] = vid_tubes['te'][k]
    #     final_tubes['dpActionScore'][action_count] = vid_tubes['dpActionScore'][k]
    #     final_tubes['label'][action_count] = vid_tubes['label'][k]
    #     final_tubes['dpPathScore'][action_count] = vid_tubes['dpPathScore'][k]
    #     final_tubes['path_total_score'][action_count] = vid_tubes['path_total_score'][k]
    #     final_tubes['path_boxes'][action_count] = vid_tubes['path_boxes'][k]
    #     final_tubes['path_scores'][action_count] = vid_tubes['path_score'][k]
    #     action_count += 1

    return final_tubes

def convert2eval(final_tubes,min_num_frames,kthresh,topk):
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
        act_path_scores = path_scores[a]

        act_scores = -sorted(-act_path_scores[act_ts:act_te+1])

        topk_mean = mean(act_scores[:min(topk,len(act_scores))])

        bxs = final_tubes['path_boxes'][a][act_ts:act_te+1,:]

        bxs = [bxs[:,:2],bxs[:,2:]-bxs[:,:2]]

        label = final_tubes['label'][a]

        if topk_mean > 0 and (act_te-act_ts) > min_num_frames:
            xmld['score'][act_nr] = topk_mean
            xmld['nr'][act_nr] = act_nr
            xmld['class'][act_nr] = label
            xmld['framenr'][act_nr] = {'fnr':np.array(range(act_ts,act_te+1)) + starts[a] - 1}
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
        indexes = np.argsort(-scores)
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
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
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


def xVOCap(rec,prec):
    mprec = np.array([0]+rec+[1])
    mpre = np.array([0]+rec+[0])
    for i in range(len(mprec)-2,-1,-1):
        mpre[i] = max(mpre[i],mpre[i+1])

    i = np.where(mrec[1:]!=mrec(:-1))+1
    ap = np.sum((mrec[i]-mrec[i-1])*mpre[i])
    return ap

def get_PR_curve(annot, xmldata, iouth):
    actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
        'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
        'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
        'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
        'VolleyballSpiking','WalkingWithDog']
    numActions = len(actions)
    AP = np.zeros(numActions)
    averageIoU = np.zeros(numActions)

    cc = zeros(num_actions)
    allscore = {}
    for a in range(numActions):
        allscore[a] = np.zeros(10000,2)

    total_num_gt_tubes = np.zeros(numActions)

    preds = -1
    gts = 0
    maxscore = -10000
    annotName = annot[1][0]
    print("Check video name again:",annotName)
    action_id = int(annot[2][0][0][2])
    print("Check action id=",action_id)

    gt_tubes = annot[2][0]
    dt_tubes = sort_detection(xmldata)

    num_detection = len(dt_tubes['class'])
    num_gt_tubes = len(gt_tubes)
    print("num gt tubes=",num_gt_tubes)

    for gtind in range(num_gt_tubes):
        action_id = gt_tubes[gtind][2]
        assert(action_id == int(action_id))
        total_num_gt_tubes[action_id] += 1

    gts = action_id
    dt_labels = dt_tubes['class']
    covered_gt_tubes = np.zeros(num_gt_tubes)
    for dtind in range(num_detection):
        # frame number range
        dt_fnr = dt_tubes['framenr'][dtind]['fnr']
        # bounding boxes
        dt_bb = dt_tubes['boxes'][dtind]['bxs']
        # class label
        dt_label = dt_labels[dtind]
        # the tube having the max score decides
        # the label of the video
        if dt_tubes['score'][dtind] > maxscore:
            preds = dt_label
            maxscore = dt_tubes['score'][dtind]
        # cc counts the number of detections per class
        cc[dt_label] += 1

        ioumax = -10000
        maxgtind = 0
        for gtind in range(num_gt_tubes):
            action_id = gt_tubes[gtind][2]#class
            # if this gt tube is not covered and has the same label as this detected tube
            if (not covered_gt_tubes[gtind]) and dt_label == action_id:
                gt_fnr = range(gt_tubes[gtind][1],gt_tubes[gtind][0]+1)
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

    # todo: need to store all detection info of all videos into allscore
    for a in range(numActions):
        allscore[a] = allscore[a][:cc[a],:]
        scores = allscore[a][:,0]
        labels = allscore[a][:,1]
        si = np.argsort(-scores)
        labels = labels[si]
        fp = np.cumsum(labels == 0)
        tp = np.cumsum(labels == 1)
        cdet = 0
        if len(tp) > 0:
            cdet = tp[-1]
            averageIoU[a] = (averageIoU[a]+0.000001)/(tp[-1]+0.000001)

        recall = tp/total_num_gt_tubes[a]
        precision = tp/(fp+tp)
        AP[a] = xVOCap(recall,precision)
        print('Action %02d AP = %0.5f and AIOU %0.5f\
             GT %03d total det %02d correct det %02d %s\n'\
             .format(a, AP[a],averageIoU[a],
            total_num_gt_tubes[a],len(tp),cdet,actions[a]))

    acc = (preds==gts)
    mAP = np.max(AP)
    mAIoU = np.max(averageIoU)

    return mAP,mIoU,acc,AP


# smooth tubes and evaluate them
def getTubes(allPath,video_id):
    # read all groundtruth actions
    final_annot_location = args.data_root + 'splitfiles/correctedAnnots_test.mat'
    annot = sio.loadmat(final_annot_location)[0][video_id]
    # smooth action path
    alphas = [3,5]
    numActions = 24
    for alpha in alphas:
        smoothedtubes = actionPathSmoother(allPath,alpha*torch.ones(numActions,1),numActions)

        min_num_frames = 8
        topk = 40

        xmldata = convert2eval(smoothedtubes, min_num_frames, topk)

        # evaluate
        iouths = [0.2] + [0.5 + 0.05*i for i in range(10)]
        for iouth in iouths:
            tmAP,tmIoU,tacc,AP = get_PR_curve(annot, xmldata, iouth)
            print('%.2f %0.3f %0.3f %.2f\n'.format(iou_th,tmAP,tacc,AP))

def process_video_result(video_result):
    frame_det_res = video_result['data']
    videoname = video_result['videoname']
    video_id = video_result['video_id']
    print("Processing:",videoname,'id=',video_id,"total frames:",len(video_result))
    allPath = actionPath(frame_det_res)
    getTubes(allPath,video_id)


def test_net(net, save_root, exp_name, input_type, dataset, iteration, num_classes, thresh=0.5 ):
    """ Test a SSD network on an Action image database. """

    val_data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                            shuffle=False, collate_fn=detection_collate, pin_memory=True)
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
    num_batches = len(val_data_loader)
    det_file = save_root + 'cache/' + exp_name + '/detection-'+str(iteration).zfill(6)+'.pkl'
    print('Number of images ', len(dataset),' number of batchs', num_batches)
    frame_save_dir = save_root+'detections/CONV-'+input_type+'-'+args.listid+'-'+str(iteration).zfill(6)+'/'
    print('\n\n\nDetections will be store in ',frame_save_dir,'\n\n')

    # id identify diffferent video segments
    pre_video_id = -1
    # list to store all results of a single video
    video_result = {}
    video_result['data'] = []
    with torch.no_grad():
        for val_itr in range(len(val_data_loader)):
            if not batch_iterator:
                batch_iterator = iter(val_data_loader)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            images, targets, img_indexs = next(batch_iterator)
            batch_size = images.size(0)
            height, width = images.size(2), images.size(3)

            if args.cuda:
                images = images.cuda()
            output = net(images)

            loc_data = output[0]
            conf_preds = output[1]
            prior_data = output[2]

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf - t1))
            for b in range(batch_size):
                gt = targets[b].numpy()
                gt[:, 0] *= width
                gt[:, 2] *= width
                gt[:, 1] *= height
                gt[:, 3] *= height
                gt_boxes.append(gt)
                decoded_boxes = decode(loc_data[b].data, prior_data.data, args.cfg['variance']).clone()
                conf_scores = net.softmax(conf_preds[b]).data.clone()
                index = img_indexs[b]
                annot_info = image_ids[index]

                frame_num = annot_info[1]; video_id = annot_info[0]; videoname = video_list[video_id]
                # check if this id is different from the previous one
                if (video_id != pre_video_id) and (len(video_result['data']) > 0):
                    # process this video
                    video_result['videoname'] = videoname
                    video_result['video_id'] = video_id
                    process_video_result(video_result)
                    video_result['data'] = []
                pre_video_id = video_id

                output_dir = frame_save_dir+videoname
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                output_file_name = output_dir+'/{:05d}.mat'.format(int(frame_num))
                res = {}
                res['scores'] = conf_scores
                res['decoded_boxes'] = decoded_boxes
                video_result['data'].append(res)
                # save_ids.append(output_file_name)
                # sio.savemat(output_file_name, mdict=res)

                for cl_ind in range(1, num_classes):
                    scores = conf_scores[:, cl_ind].squeeze()
                    c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                    scores = scores[c_mask].squeeze()
                    # print('scores size',scores.size())
                    if scores.dim() == 0 or scores.shape[0] == 0:
                        # print(len(''), ' dim ==0 ')
                        det_boxes[cl_ind - 1].append(np.asarray([]))
                        continue
                    boxes = decoded_boxes.clone()
                    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    boxes = boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                    scores = scores[ids[:counts]].cpu().numpy()
                    boxes = boxes[ids[:counts]].cpu().numpy()
                    # print('boxes sahpe',boxes.shape)
                    boxes[:, 0] *= width
                    boxes[:, 2] *= width
                    boxes[:, 1] *= height
                    boxes[:, 3] *= height

                    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                    det_boxes[cl_ind - 1].append(cls_dets)

                count += 1
            if val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te - ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('NMS stuff Time {:0.3f}'.format(te - tf))
        if (len(video_result['data']) > 0):
            # process this video
            process_video_result(video_result)
    print('Evaluating detections for itration number ', iteration)

    #Save detection after NMS along with GT
    with open(det_file, 'wb') as f:
        pickle.dump([gt_boxes, det_boxes, save_ids], f, pickle.HIGHEST_PROTOCOL)

    return evaluate_detections(gt_boxes, det_boxes, CLASSES, iou_thresh=thresh)


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
        # Load dataset
        dataset = OmniUCF24(args.data_root, 'test', BaseTransform(300, means), AnnotationTransform(), 
                            cfg=args.cfg, input_type=args.input_type, 
                            outshape=args.outshape, full_test=True)
        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        mAP, ap_all, ap_strs = test_net(net, args.save_root, exp_name, args.input_type, dataset, iteration, num_classes)
        for ap_str in ap_strs:
            print(ap_str)
            log_file.write(ap_str + '\n')
        ptr_str = '\nMEANAP:::=>' + str(mAP) + '\n'
        print(ptr_str)
        log_file.write(ptr_str)

        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()

if __name__ == '__main__':
    main()
