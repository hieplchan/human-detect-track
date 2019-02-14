import numpy as np
import time
import cv2
import os

from posenet_minimum import ROOT_DIR

""" IMAGE VIDEO TEST PARAMS """
CAM_WIDTH = 1920 #960
CAM_HEIGHT = 1080 #540
INPUT_IMG_TEST_DIR = '/media/hiep/DATA/Work_space/Tracking_CCTV/CCTV_Data/Image'
VIDEO_PATH = '/media/hiep/DATA/Work_space/Tracking_CCTV/CCTV_Data/Video/'
OUTPUT_VIDEO_PATH = ROOT_DIR + '/output/'
VIDEO_NAME = '8.mp4'

""" LUCAS KANADE PARAMS """
# Lucas kanade params
lk_params = dict(winSize = (15, 15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

TRIANGLE_WIDE = 10

""" POSENET PARAMS """
POSENET_MODEL_NUM = 50
SCALE_FACTOR = 1
OUTPUT_STRIDE = 16
THRESHOLD = 0.05
