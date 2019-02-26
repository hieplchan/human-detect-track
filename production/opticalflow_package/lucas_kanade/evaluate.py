import cv2
import numpy as np
from lucas_kanade.utils import *
from lucas_kanade.params import lk_params

class Lucas_Kanade:
    ''' Point track with Lucas Kanade '''

    def __init__(self, width, height):
        self.old_frame = np.zeros((width, height, 3), dtype = np.uint8)
        self.new_frame = np.zeros((width, height, 3), dtype = np.uint8)
        self.old_points = np.zeros((1, 2), dtype = np.float32)
        self.new_points = np.zeros((1, 2), dtype = np.float32)

    def pointUpdate(self, keypoint):
        ''' Update point to track '''
        self.old_points = keypointlist2array(keypoint)

    def frameUpdate(self, frame):
        self.old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def pointTrackCal(self, frame):
        self.new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.new_points, status, error = cv2.calcOpticalFlowPyrLK(self.old_frame, self.new_frame, self.old_points, None, **lk_params)
        self.old_frame = self.new_frame
        self.old_points = self.new_points

        return array2keypointlist(self.new_points)
