# Mathematical
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import cv2

# Pytorch
import torch
from torch.utils import data
from torchvision import datasets

# Misc
from functools import lru_cache

import os

import sys
sys.path.insert(0, '/home/monet/research/realtime-action-detection')
from utils.augmentations import SSDAugmentation
from data import v6

def genuv(h, w):
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    return np.stack([u, v], axis=-1)


def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * cos_u,
        cos_v * sin_u,
        sin_v
    ], axis=-1)


def xyz2uv(xyz):
    c = np.sqrt((xyz[..., :2] ** 2).sum(-1))
    u = np.arctan2(xyz[..., 1], xyz[..., 0])
    v = np.arctan2(xyz[..., 2], c)
    return np.stack([u, v], axis=-1)

def get_rotated_mat(outshape,inshape,rot_x,rot_y,rot_z,fov):
    uv = genuv(*outshape)
    xyz = uv2xyz(uv.astype(np.float64))

    # rotate along x-axis
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = xyz[..., 0]
    xyz_rot[..., 1] = np.cos(rot_x) * xyz[..., 1] - np.sin(rot_x) * xyz[..., 2]
    xyz_rot[..., 2] = np.sin(rot_x) * xyz[..., 1] + np.cos(rot_x) * xyz[..., 2]
    # rotate along y-axis
    xyz = xyz_rot.copy()
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(rot_y) * xyz[..., 0] - np.sin(rot_y) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = np.sin(rot_y) * xyz[..., 0] + np.cos(rot_y) * xyz[..., 2]
    # rotate along z-axis
    xyz = xyz_rot.copy()
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(rot_z) * xyz[..., 0] - np.sin(rot_z) * xyz[..., 1]
    xyz_rot[..., 1] = np.sin(rot_z) * xyz[..., 0] + np.cos(rot_z) * xyz[..., 1]
    xyz_rot[..., 2] = xyz[..., 2]

    # get rotated uv matrix
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]

    h,w = inshape
    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(fov / 2)) + w / 2
    y = y * h / (2 * np.tan(fov / 2)) + h / 2

    return u,v,x,y

def get_region(u,v,x,y,fov,xmin,ymin,xmax,ymax):
    invalid = (u < -fov / 2) | (u > fov / 2) |\
              (v < -fov / 2) | (v > fov / 2)
    x[invalid] = -1
    y[invalid] = -1

    valid = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return valid


def IoU(bbox1,bbox2,outshape=(300,600),inshape=(300,300),fov=np.pi/3):
    h,w = inshape
    # get region of the first box
    xmin1,ymin1,xmax1,ymax1,rot_x1,rot_y1,rot_z1 = bbox1
    u1,v1,x1,y1 = get_rotated_mat(outshape,inshape,rot_x1,rot_y1,rot_z1,fov)
    valid1 = get_region(u1,v1,x1,y1,fov,xmin1*w,ymin1*h,xmax1*w,ymax1*h)

    # get region of the second box
    xmin2,ymin2,xmax2,ymax2,rot_x2,rot_y2,rot_z2 = bbox2
    u2,v2,x2,y2 = get_rotated_mat(outshape,inshape,rot_x2,rot_y2,rot_z2,fov)
    valid2 = get_region(u2,v2,x2,y2,fov,xmin2*w,ymin2*h,xmax2*w,ymax2*h)

    intersec = sum(sum(valid1&valid2))
    union = sum(sum(valid1|valid2))
    iou = intersec/union

    return iou



def uv2img_idx(uv, h, w, u_fov, v_fov, rot_x=0, rot_y=0, rot_z=0):
    # the coord on sphere of each pixel in the output image
    xyz = uv2xyz(uv.astype(np.float64)) # out_h, out_w, (x,y,z)

    # rotate along z-axis
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(rot_z) * xyz[..., 0] - np.sin(rot_z) * xyz[..., 1]
    xyz_rot[..., 1] = np.sin(rot_z) * xyz[..., 0] + np.cos(rot_z) * xyz[..., 1]
    xyz_rot[..., 2] = xyz[..., 2]
    xyz = xyz_rot.copy()
    # rotate along y-axis
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(rot_y) * xyz[..., 0] - np.sin(rot_y) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = np.sin(rot_y) * xyz[..., 0] + np.cos(rot_y) * xyz[..., 2]
    xyz = xyz_rot.copy()
    # rotate along x-axis
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = xyz[..., 0]
    xyz_rot[..., 1] = np.cos(rot_x) * xyz[..., 1] - np.sin(rot_x) * xyz[..., 2]
    xyz_rot[..., 2] = np.sin(rot_x) * xyz[..., 1] + np.cos(rot_x) * xyz[..., 2]
    # get rotated uv matrix
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]

    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(u_fov / 2)) + w / 2
    y = y * h / (2 * np.tan(v_fov / 2)) + h / 2

    invalid = (u < -u_fov / 2) | (u > u_fov / 2) |\
              (v < -v_fov / 2) | (v > v_fov / 2) 
    x[invalid] = -1
    y[invalid] = -1

    invalid = (x < 0) | (x > w) | (y < 0) | (y > h)

    return np.stack([y, x], axis=0),invalid


class OmniDataset(data.Dataset):
    def __init__(self, dataset, fov=120, outshape=(512, 2*512),
                 z_rotate=True, y_rotate=True, x_rotate=False,
                 fix_aug=False, use_background=True, num_bgs=22, save_final_annot=True):
        '''
        Convert classification dataset to omnidirectional version
        @dataset  dataset with same interface as torch.utils.data.Dataset
                  yield (PIL image, label) if indexing
        '''
        self.dataset = dataset
        self.ids = dataset.ids
        self.video_list = dataset.video_list
        self.fov = fov
        self.outshape = outshape
        self.z_rotate = z_rotate
        self.y_rotate = y_rotate
        self.x_rotate = x_rotate
        self.name = dataset.name
        self.use_background = use_background
        self.video_list = dataset.video_list
        self.ids = dataset.ids
        self.root = dataset.root

        self.aug = None
        if fix_aug:
            self.aug = [
                {
                    'z_rotate': np.random.uniform(-np.pi, np.pi),
                    'y_rotate': np.random.uniform(-np.pi/2, np.pi/2),
                    'x_rotate': np.random.uniform(-np.pi, np.pi),
                }
                for _ in range(len(self.dataset))
            ]

        # load backgorounds
        self.bg_imgs = []
        img_root = '/home/monet/research/realtime-action-detection/data/background/'
        for bg_idx in range(1,23):
            img_name = img_root + str(bg_idx) + '.jpg'
            bg_img = cv2.imread(img_name)
            bg_img = cv2.resize(bg_img, (self.outshape[1], self.outshape[0]))
            self.bg_imgs += [bg_img]

        # map video to background
        self.vid2bgidx = {}
        for vid in range(len(self.video_list)):
            self.vid2bgidx[vid] = np.random.randint(0,22)

        # map video to rotation
        self.vid2rot = {}
        for vid in range(len(self.video_list)):
            if self.y_rotate:
                if self.aug is not None:
                    rot_y = self.aug[idx]['y_rotate']
                else:
                    rot_y = np.random.uniform(-np.pi/2, np.pi/2)
            else:
                rot_y = 0

            if self.z_rotate:
                if self.aug is not None:
                    rot_z = self.aug[idx]['z_rotate']
                else:
                    rot_z = np.random.uniform(-np.pi, np.pi)
            else:
                rot_z = 0

            if self.x_rotate:
                if self.aug is not None:
                    rot_x = self.aug[idx]['x_rotate']
                else:
                    rot_x = np.random.uniform(-np.pi, np.pi)
            else:
                rot_x = 0

            self.vid2rot[vid] = (rot_x,rot_y,rot_z)

        # save jhmdb convereted data in cache
        
        # self.final_dataset_location = self.root + 'cache/final_dataset_' + self.dataset.image_set + '.npy'
        # self.original_annot_location = self.root +'splitfiles/finalAnnots.mat'
        # self.final_annot_location = self.root + 'splitfiles/correctedAnnots_' + self.dataset.image_set + '.mat'

        # if self.dataset.image_set == 'test':
            # data_type = '2d'
        #     print('transforming annotation')
        #     assert(os.path.exists(self.original_annot_location))
        #     import collections
        #     self.annot_map = collections.defaultdict(dict)

        #     # transform the images
        #     for idx in range(len(self.dataset)):
        #         annot_info = self.ids[idx]
        #         video_id = annot_info[0]
        #         videoname = self.video_list[video_id]

        #         label = self._get_label(self.dataset[idx][1], *self.vid2rot[video_id])
        #         old_label = self.ids[idx][3]
        #         for old,new in zip(old_label,label):
                    # old2 = (int(old[0]),int(old[1]),int(old[2]-old[0]),int(old[3]-old[1]))
                    # if sum(old2) == 0:continue
                    # if data_type == '2d':
                    #     self.annot_map[videoname][old2] = old
                    # else:
                    #     self.annot_map[videoname][old2] = [int(new[0]*1024),
                    #                                         int(new[1]*512),
                    #                                         int(new[2]*1024),
                    #                                         int(new[3]*512)]
        #         if idx%100 == 0:
        #             print('Transforming %6d/%6d'%(idx,len(dataset)))

        #     # transform the annotation
        #     import scipy.io as sio
        #     old_annots = sio.loadmat(self.original_annot_location)
        #     for annot in old_annots['annot'][0]:
        #         filename = annot[1][0]
        #         if filename in self.annot_map:
        #             for tube in annot[2][0]:
        #                 new_boxes = []
        #                 for i,old_box in enumerate(tube[3]):
        #                     key = (old_box[0],old_box[1],old_box[2],old_box[3])
        #                     assert(key in self.annot_map[filename])
        #                     new_boxes.append(self.annot_map[filename][key])
        #                 tube[3] = new_boxes
        #         else:
        #             print(filename)
        #     sio.savemat(self.final_annot_location,{'annot':old_annots['annot'][0]})
        #     print('transform finishes')
        #     exit(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, index = self._transform_item(idx)
        return img, label, index

    def _transform_item(self, idx):
        annot_info = self.ids[idx]
        video_id = annot_info[0]

        rot_x,rot_y,rot_z = self.vid2rot[video_id]

        bg_img = None
        if self.use_background:
            bg_idx = self.vid2bgidx[video_id]
            bg_img = self.bg_imgs[bg_idx]

        img, label, index = self.dataset[idx]

        if len(img.shape)==2:
            img = self._get_img(img, rot_x, rot_y, rot_z)
            label = self._get_label(label, rot_x, rot_y, rot_z)
        else:
            img_stack = []
            for ch in range(img.shape[0]):
                img_ch = self._get_img(img, rot_x, rot_y, rot_z, bg_img, ch=ch)
                img_stack.append(img_ch.unsqueeze(0))
            img = torch.cat(img_stack, dim=0)
            assert(img.shape[0]==3)
            label = self._get_label(label, rot_x, rot_y, rot_z)

        return img, label, index

    def _get_label(self, bboxes, rot_x, rot_y, rot_z):

        new_bboxes = []
        for xmin,ymin,xmax,ymax,ac_type in bboxes:
            umin = np.arctan((xmin-0.5)*2*1.7321)
            umax = np.arctan((xmax-0.5)*2*1.7321)

            all_v = []
            all_v += [np.arctan2(ymin-0.5,np.sqrt(1./12+(xmin-0.5)**2))]
            all_v += [np.arctan2(ymin-0.5,np.sqrt(1./12+(xmax-0.5)**2))]
            all_v += [np.arctan2(ymax-0.5,np.sqrt(1./12+(xmin-0.5)**2))]
            all_v += [np.arctan2(ymax-0.5,np.sqrt(1./12+(xmax-0.5)**2))]
            if xmin <= 0.5 and xmax >= 0.5:
                all_v += [np.arctan2(ymin-0.5,np.sqrt(1./12))]
                all_v += [np.arctan2(ymax-0.5,np.sqrt(1./12))]

            # [-0.5,0.5] left->right;top->bottom
            umin,umax = umin/2/np.pi,umax/2/np.pi
            vmin,vmax = min(all_v)/np.pi,max(all_v)/np.pi

            h,w = vmax-vmin,umax-umin
            cu,cv = (umax+umin)/2,(vmax+vmin)/2 
            
            # rotate around origin before translation
            cu2 = cu * np.cos(rot_x) + cv * np.sin(rot_x)
            cv2 = -cu * np.sin(rot_x) + cv * np.cos(rot_x)
            cu,cv = cu2,cv2

            cu += 0.5
            cv += 0.5

            cu -= rot_z/2/np.pi
            cv -= rot_y/np.pi
            cu = cu%1
            if cv > 1:
                cv = 2 - cv
                cu = cu-0.5 if cu>=0.5 else cu+0.5
            elif cv < 0:
                cv = -cv
                cu = cu-0.5 if cu>=0.5 else cu+0.5
            new_bboxes.append([cu-w/2,cv-h/2,cu+w/2,cv+h/2,ac_type])
        bboxes = new_bboxes

        return bboxes


    def _get_img(self, img, rot_x, rot_y, rot_z, bg_img, ch = None):
        # get image content from one channel

        if ch is not None:
            img = img[ch,:,:]
        h, w = img.shape[:2]
        uv = genuv(*self.outshape) # out_h, out_w, (out_phi, out_theta)
        fov = self.fov * np.pi / 180

        img_idx, invalid = uv2img_idx(uv, h, w, fov, fov, rot_x, rot_y, rot_z)
        x = map_coordinates(img, img_idx, order=1)

        if bg_img is not None and ch is not None:
            means = (104, 117, 123)
            bg_img_ch = bg_img[:,:,2-ch]
            x[invalid] = bg_img_ch[invalid] - means[ch]

        return torch.FloatTensor(x.copy())


from data import UCF24Detection, AnnotationTransform, BaseTransform, JHMDB

class OmniUCF24(OmniDataset):
    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='ucf24', input_type='rgb', full_test=False, *args, **kwargs):
        self.UCF24 = UCF24Detection(root, image_set, transform, target_transform,
                                    dataset_name, input_type, full_test)
        super(OmniUCF24, self).__init__(self.UCF24, *args, **kwargs)

class OmniJHMDB(OmniDataset):
    def __init__(self, root, image_set, transform=None, target_transform=None, *args, **kwargs):
        self.JHMDB = JHMDB(root, image_set, transform, target_transform, split=1)
        super(OmniJHMDB, self).__init__(self.JHMDB, *args, **kwargs)

if __name__ == '__main__':

    import os
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--idx', nargs='+', required=True,
                        help='image indices to demo')
    parser.add_argument('--out_dir', default='output/demo',
                        help='directory to output demo image')
    parser.add_argument('--dataset', default='OmniJHMDB',
                        choices=['OmniUCF24','OmniJHMDB'],
                        help='which dataset to use')

    parser.add_argument('--fov', type=int, default=120,
                        help='fov of the tangent plane')
    parser.add_argument('--flip', action='store_true',
                        help='whether to apply random flip')
    parser.add_argument('--z_rotate', action='store_true',
                        help='whether to apply random panorama horizontal rotation')
    parser.add_argument('--y_rotate', action='store_true',
                        help='whether to apply random panorama vertical rotation')
    parser.add_argument('--fix_aug', action='store_true',
                        help='whether to apply random panorama vertical rotation')

    parser.add_argument('--ssd_dim', default=300, type=int,
                        help='Input Size for SSD') # only support 300 now
    parser.add_argument('--input_type', default='rgb', type=str,
                        help='INput tyep default rgb options are [rgb,brox,fastOF]')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    args.train_sets = 'train'
    args.means = (104, 117, 123)
    np.random.seed(111)

    if args.dataset == 'OmniUCF24':
        args.data_root = '/home/bo/research/dataset/ucf24/'
        dataset = OmniUCF24(args.data_root, 'test', BaseTransform(300, args.means),
                           AnnotationTransform(), input_type=args.input_type, full_test=True)
    elif args.dataset == 'OmniJHMDB':
        args.data_root = '/home/monet/research/dataset/jhmdb/'
        dataset = OmniJHMDB(args.data_root, 'test', BaseTransform(300, None),
                           AnnotationTransform())

    else:
        exit(0)

    for idx in args.idx:
        idx = int(idx)
        path = os.path.join(args.out_dir, '%d.png' % idx)
        x, label, _ = dataset[idx]
        # for ch in range(0,3):
        #     x[ch,:,:] -= args.means[ch]

        print(path, label)
        img = Image.fromarray(x.permute(1, 2, 0).numpy().astype(np.uint8))
        img.save(path)
