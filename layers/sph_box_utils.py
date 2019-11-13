""" Bounding box utilities

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

"""

import torch
import numpy as np


def genuv(h, w):
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w))
    u = u.type(torch.FloatTensor)
    v = v.type(torch.FloatTensor)
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    return torch.stack([u, v], dim=-1)

def uv2xyz(uv):
    sin_u = torch.sin(uv[..., 0])
    cos_u = torch.cos(uv[..., 0])
    sin_v = torch.sin(uv[..., 1])
    cos_v = torch.cos(uv[..., 1])
    return torch.stack([
        cos_v * cos_u,
        cos_v * sin_u,
        sin_v
    ], dim=-1)

def xyz2uv(xyz):
    c = torch.sqrt((xyz[..., :2] ** 2).sum(-1))
    u = torch.atan2(xyz[..., 1], xyz[..., 0])
    v = torch.atan2(xyz[..., 2], c)
    return torch.stack([u, v], dim=-1)

def get_rotated_mat(bbox,fov,outshape=(300,300)):
    xmin,ymin,xmax,ymax,rot_x,rot_y,rot_z = bbox
    uv = genuv(*outshape)
    xyz = uv2xyz(uv)

    # convert rotation to rad
    rot_x = np.pi*2*rot_x - np.pi
    rot_y = np.pi*2*rot_y - np.pi
    rot_z = np.pi*2*rot_z - np.pi
    # rotate along x-axis
    xyz_rot = xyz.clone().detach()
    xyz_rot[..., 0] = xyz[..., 0]
    xyz_rot[..., 1] = torch.cos(rot_x) * xyz[..., 1] - torch.sin(rot_x) * xyz[..., 2]
    xyz_rot[..., 2] = torch.sin(rot_x) * xyz[..., 1] + torch.cos(rot_x) * xyz[..., 2]
    xyz = xyz_rot.clone().detach()
    # rotate along y-axis
    xyz_rot = xyz.clone().detach()
    xyz_rot[..., 0] = torch.cos(rot_y) * xyz[..., 0] - torch.sin(rot_y) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = torch.sin(rot_y) * xyz[..., 0] + torch.cos(rot_y) * xyz[..., 2]
    xyz = xyz_rot.clone().detach()
    # rotate along z-axis
    xyz_rot = xyz.clone().detach()
    xyz_rot[..., 0] = torch.cos(rot_z) * xyz[..., 0] - torch.sin(rot_z) * xyz[..., 1]
    xyz_rot[..., 1] = torch.sin(rot_z) * xyz[..., 0] + torch.cos(rot_z) * xyz[..., 1]
    xyz_rot[..., 2] = xyz[..., 2]
    # get rotated uv matrix
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]

    x = torch.tan(u)
    y = torch.tan(v) / torch.cos(u)
    x = x / (2 * torch.tan(torch.FloatTensor([fov / 2]))) + 0.5
    y = y / (2 * torch.tan(torch.FloatTensor([fov / 2]))) + 0.5

    return u,v,x,y

def get_region(u,v,x,y,fov,bbox):
    xmin,ymin,xmax,ymax,rot_x,rot_y,rot_z = bbox
    invalid = (u < -fov / 2) | (u > fov / 2) |\
              (v < -fov / 2) | (v > fov / 2)
    x[invalid] = -1
    y[invalid] = -1

    x = x.cuda()
    y = y.cuda()

    valid = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return valid


def iou(bbox1,bbox2,fov=np.pi/3,use_precompute=False):
    # get region of the first box
    u1,v1,x1,y1 = get_rotated_mat(bbox1,fov)
    valid1 = get_region(u1,v1,x1,y1,fov,bbox1)

    # get region of the second box
    u2,v2,x2,y2 = get_rotated_mat(bbox2,fov)
    valid2 = get_region(u2,v2,x2,y2,fov,bbox2)

    intersec = sum(sum(valid1&valid2))
    union = sum(sum(valid1|valid2))

    return intersec/union

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:4]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:4]/2,      # xmax, ymax
                     boxes[:,4:]), 1)  

def jaccard(box_a, box_b, use_gpu):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    A = box_a.size(0)
    B = box_b.size(0)
    if use_gpu:
        IoU = torch.cuda.FloatTensor(A, B)
    else:
        IoU = torch.Tensor(A, B)

    for i in range(A):
        for j in range(B):
            print(i,j)
            IoU[i][j] = iou(box_a[i],box_b[j])
    
    return IoU


def sph_match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, use_gpu):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    print(truths)
    overlaps = jaccard(
        truths,
        point_form(priors),
        use_gpu
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,7]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,7] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:4])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:4])
    # match wh / prior wh
    g_wh = (matched[:, 2:4] - matched[:, :2]) / priors[:, 2:4]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh, matched[:,4:]], 1)  # [num_priors,7]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    print("decode not implemented in sph_box_utils.py")
    exit(0)

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    # 1. find a box with the highest score
    # 2. keep k top boxes with the highest score
    # 3. calculate IoU of remaining box and the top-1 box
    # 4. keep the remaining box with IoU >= threshold
    # 5. go to step 1 until no remaining box
    print('nms not implemented in sph_box_utils.py')
    exit(0)

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
