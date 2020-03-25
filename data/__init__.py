#from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .ucf24 import UCF24Detection, AnnotationTransform, detection_collate, UCF24_CLASSES
from .config import *
# from .jhmdb import JHMDB, JHMDB_CLASSES
import cv2
import numpy as np


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        if mean is not None:
            self.mean = np.array(mean, dtype=np.float32)
        else:
            self.mean = None

    def __call__(self, image, boxes=None, labels=None):
        if self.mean is not None:
            return base_transform(image, self.size, self.mean), boxes, labels
        else:
            x = cv2.resize(image, (self.size, self.size)).astype(np.float32)
            return x, boxes, labels
        
