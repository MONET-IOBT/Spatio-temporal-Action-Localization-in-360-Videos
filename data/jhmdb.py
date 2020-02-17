from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
import cv2
import numpy as np
import os.path
import scipy.io as sio
import glob
import torch
# import torch.utils.data as data
import sys
sys.path.insert(0, '/home/bo/research/realtime-action-detection')
from data import UCF24Detection, AnnotationTransform

JHMDB_CLASSES = ('brush_hair', 'catch', 'clap', 'climb_stairs', 'golf',
             'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
             'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
             'stand', 'swing_baseball', 'throw', 'walk', 'wave')

class JHMDB(torch.utils.data.Dataset):
  def __init__(self, root, image_set, transform=None, target_transform=None, split=1):
    self.image_set = image_set
    self.root = root #'/home/bo/research/dataset/jhmdb'
    self._vddb = []
    self._height = 240
    self._width = 320
    self._split = split - 1
    self.transform = transform
    self.target_transform = target_transform
    self.name = 'jhmdb'

    self._num_classes = 21
    self._classes = JHMDB_CLASSES
    self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
    cache_file = os.path.join(self.root, 'cache',
        'jhmdb_%d_%d_db.pkl' % (self._height, self._width))
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        self._vddb = joblib.load(fid)
      print ('{} gt vddb loaded from {}'.format(self.image_set, cache_file))
    else:
      self._vddb = self._read_video_list()

      [self._load_annotations(v) for v in self._vddb]

      with open(cache_file, 'wb') as fid:
        joblib.dump(self._vddb, fid)

    self._curr_idx = 0

    mean_file = os.path.join(self.root, 'cache',
                             'mean_frame_{}_{}.npy'.format(self._height,
                                                           self._width))
    if os.path.exists(mean_file):
      self._mean_frame = np.load(mean_file)
    else:
      self._mean_frame = self.compute_mean_frame()

    if image_set == 'train':
      self._vddb = self.keeps(1)
    else:
      if image_set == 'test':
        self._vddb = self.keeps(2)

    # create video list and ids
    self.video_list = []
    self.ids = []
    for vid, video in enumerate(self._vddb):
      self.video_list.append(video['video_name'])
      for fid in range(len(video['gt_bboxes'])):
        self.ids.append([vid,fid,np.asarray([video['gt_label']]),np.asarray([video['gt_bboxes'][fid]])])

  def __getitem__(self, index):
    im, gt, img_index = self.pull_item(index)

    return im, gt, img_index

  def __len__(self):
    return len(self.ids)

  def pull_item(self, index):
    annot_info = self.ids[index]
    video_id = annot_info[0]
    frame_num = annot_info[1]
    videoname = self.video_list[video_id]
    img = self._vddb[video_id]['video'][frame_num] - self._mean_frame
    target = self.target_transform(annot_info[3], annot_info[2], 320, 240)
    if self.transform is not None:
      target = np.array(target)
      img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
      img = img[:, :, (2, 1, 0)]
      target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

    return torch.from_numpy(img).permute(2, 0, 1), target, index

  @property
  def vddb(self):
    return self._vddb
  @property
  def size(self):
    return len(self._vddb)

  def keeps(self, num):
    result = []
    for i in range(len(self.vddb)):
      if self.vddb[i]['split'][self._split] == num:
        result.append(self.vddb[i])
    return result

  def _load_annotations(self, video):
    """Read video annotations from text files.
    """
    gt_file = os.path.join(self.root, 'puppet_mask',
                           video['video_name'], 'puppet_mask.mat')
    if not os.path.isfile(gt_file):
      raise Exception(gt_file + 'does not exist.')
    masks = sio.loadmat(gt_file)['part_mask']
    print(gt_file)
    gt_label = self._class_to_ind[video['video_name'][: video['video_name'].find("/")]]
    depth = masks.shape[2]

    pixels = self.clip_reader(video['video_name'])

    gt_bboxes = np.zeros((depth, 4), dtype=np.float32)
    
    for j in range(depth):
      mask = masks[:, :, j]
      (a, b) = np.where(mask > 0)
      y1 = a.min()
      y2 = a.max()
      x1 = b.min()
      x2 = b.max()

      gt_bboxes[j] = np.array([x1, y1, x2, y2])
    video['video'] = pixels
    video['gt_bboxes'] = gt_bboxes
    video['gt_label'] = gt_label

  def _read_video_list(self):
    """Read JHMDB video list from a text file."""

    vddb = []
    tmp = []
    for i in range(self._num_classes):
      file_name = os.path.join(self.root+'/splits',
                               '{}_test_split1.txt'.format(self._classes[i]))
      if not os.path.isfile(file_name):
        raise NameError('The video list file does not exists: ' + file_name)
      with open(file_name) as f:
        lines = f.readlines()

      for line in lines:
        split = np.zeros(3, dtype=np.uint8)
        p1 = line.find(' ')
        video_name = self._classes[i] + '/' + line[: p1 - 4]
        split[0] = int((line[p1 + 1 :].strip()))
        vddb.append({'video_name': video_name,
                     'split': split})
        tmp.append(video_name)

      file_name = os.path.join(self.root+'/splits',
                               '{}_test_split2.txt'.format(self._classes[i]))
      if not os.path.isfile(file_name):
        raise NameError('The video list file does not exists: ' + file_name)
      with open(file_name) as f:
        lines = f.readlines()

      for line in lines:
        p1 = line.find(' ')
        video_name = self._classes[i] + '/' + line[: p1 - 4]
        try:
          index = tmp.index(video_name)
          vddb[index]['split'][1] = int((line[p1 + 1:].strip()))
        except ValueError:
          tmp.append(video_name)
          split = np.zeros(3, dtype=np.uint8)
          split[1] = int((line[p1 + 1:].strip()))
          vddb.append({'video_name': video_name,
                       'split': split})

      file_name = os.path.join(self.root+'/splits',
                               '{}_test_split3.txt'.format(self._classes[i]))
      if not os.path.isfile(file_name):
        raise NameError('The video list file does not exists: ' + file_name)
      with open(file_name) as f:
        lines = f.readlines()

      for line in lines:
        p1 = line.find(' ')
        video_name = self._classes[i] + '/' + line[: p1 - 4]
        try:
          index = tmp.index(video_name)
          vddb[index]['split'][2] = int((line[p1 + 1:].strip()))
        except ValueError:
          tmp.append(video_name)
          split = np.zeros(3, dtype=np.uint8)
          split[2] = int((line[p1 + 1:].strip()))
          vddb.append({'video_name': video_name,
                       'split': split})

    return vddb

  def clip_reader(self, video_prefix):
    """Load frames in the clip.

    Using openCV to load the clip frame by frame.
    If specify the cropped size (crop_size > 0), randomly crop the clip.

      Args:
        index: Index of a video in the dataset.

      Returns:
        clip: A matrix (channel x depth x height x width) saves the pixels.
      """
    clip = []
    r1 = 0
    framepath = os.path.join(self.root, 'Rename_Images', video_prefix)
    num_frames = len(glob.glob(framepath + '/*.png'))
    for i in range(num_frames):
      filename = os.path.join(
          self.root, 'Rename_Images', video_prefix,
          '%05d.png' % (i + 1))

      im = cv2.imread(filename)
      if r1 == 0:
        r1 = self._height / im.shape[0]
        r2 = self._width / im.shape[1]
      im = cv2.resize(im, None, None, fx=r2, fy=r1,
                      interpolation=cv2.INTER_LINEAR)
      clip.append(im)
    return np.asarray(clip, dtype=np.uint8)

  def compute_mean_frame(self):
    sum_frame = np.zeros((self._height, self._width, 3), dtype=np.float32)
    num_frames = 0
    for db in self._vddb:
      curr_frame = np.sum(db['video'], dtype=np.float32, axis=0)
      sum_frame += curr_frame
      num_frames += db['video'].shape[0]
    sum_frame = sum_frame / num_frames
    np.save(os.path.join(self.root, 'cache',
                         'mean_frame_{}_{}.npy'.format(self._height,
                                                       self._width)),
            sum_frame)
    return sum_frame

if __name__ == '__main__':
  from data import UCF24Detection, AnnotationTransform, BaseTransform, JHMDB
  from PIL import Image
  data_root = '/home/bo/research/dataset/jhmdb/'
  dataset = JHMDB(data_root, 'train', BaseTransform(300, None),AnnotationTransform())
  path = os.path.join('output/demo', '%d.png' % 0)
  x, label, _ = dataset[1000]

  print(path, label)
  img = Image.fromarray(x.permute(1, 2, 0).numpy().astype(np.uint8))
  img.save(path)