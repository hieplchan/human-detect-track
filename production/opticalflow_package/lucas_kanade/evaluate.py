import cv2
import numpy as np
from matplotlib.path import Path
from itertools import compress
import time

from lucas_kanade.utils import *
from lucas_kanade.params import lk_params

class Lucas_Kanade:
    ''' Point track with Lucas Kanade '''

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.old_frame = np.zeros((self.width, self.height, 3), dtype = np.uint8)
        self.new_frame = np.zeros((self.width, self.height, 3), dtype = np.uint8)
        self.old_points = np.zeros((1, 2), dtype = np.float32)
        self.new_points = np.zeros((1, 2), dtype = np.float32)
        self.bodies_points = []

    def pointTrackCal(self, frame):
        ''' Calculate new list of all tracked point '''
        self.new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.new_points, status, error = cv2.calcOpticalFlowPyrLK(self.old_frame, self.new_frame, self.old_points, None, **lk_params)
        self.old_frame = self.new_frame
        self.old_points = self.new_points
        return array2keypointlist(self.new_points)

    def detectorUpdate(self, frame, keypoints, boxs):
        ''' Update point to track - split point per each body in frame'''
        self.old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.old_points = keypointlist2array(keypoints)

        #region PointInPolygon
        del self.bodies_points[:]
        tmp_new_point_list = keypointlist2tuplelist(keypoints)
        for box in boxs:
            polygon = Path(box)
            grid = polygon.contains_points(tmp_new_point_list)
            grid = list(compress(range(len(grid)), grid))
            self.bodies_points.append([tmp_new_point_list[idx] for idx in grid])
        #endregion
