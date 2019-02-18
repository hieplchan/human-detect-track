import numpy as np
import time
import cv2
import os
import torch

from posenet_minimum import ROOT_DIR

def valid_resolution(width, height, output_stride):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height

""" IMAGE VIDEO TEST PARAMS """
CAM_WIDTH = 1920
CAM_HEIGHT = 1080

# INPUT_IMG_TEST_DIR = '/home/hiep/Desktop/tracking_cctv/CCTV_Data/Image/'
# VIDEO_PATH = '/home/hiep/Desktop/tracking_cctv/CCTV_Data/Video/'
INPUT_IMG_TEST_DIR = '/media/hiep/DATA/Work_space/Tracking_CCTV/CCTV_Data/Image/'
VIDEO_PATH = '/media/hiep/DATA/Work_space/Tracking_CCTV/CCTV_Data/Video/'
OUTPUT_VIDEO_PATH = ROOT_DIR + '/output/'
VIDEO_NAME = '3.mp4'

""" LUCAS KANADE PARAMS """
# Lucas kanade params
lk_params = dict(winSize = (15, 15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

TRIANGLE_WIDE = 10

""" POSENET PARAMS """
device = torch.device('cpu')
POSENET_MODEL_NUM = 50
SCALE_FACTOR = 0.5
OUTPUT_STRIDE = 16
THRESHOLD = 0.05

TARGET_WIDTH, TARGET_HEIGHT = valid_resolution(CAM_WIDTH*SCALE_FACTOR, CAM_HEIGHT*SCALE_FACTOR, OUTPUT_STRIDE)
SCALE = np.array([CAM_HEIGHT/TARGET_HEIGHT, CAM_WIDTH/TARGET_WIDTH])
