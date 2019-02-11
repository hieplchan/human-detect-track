import numpy as np
import logging
import time
import cv2
import os

from posenet_lucas_kanade import ROOT_DIR

""" LOGGING PARAMS """
LOG_DIR = ROOT_DIR + '/log/'
if os.path.exists(LOG_DIR + 'process_time_log.csv'):
  os.remove(LOG_DIR + 'process_time_log.csv')
logger = logging.getLogger('process_time')
logger.setLevel(logging.DEBUG)
# Process tiem logger
process_time_log_handle = logging.FileHandler(LOG_DIR + 'process_time_log.csv')
process_time_log_handle.setLevel(logging.DEBUG)
process_time_log_handle.setFormatter(logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s', '%Y-%m-%d %H:%M:%S'))
logger.addHandler(process_time_log_handle)

""" VIDEO PARAMS """
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
VIDEO_NAME = '2.mp4'
VIDEO_PATH = '/media/hiep/DATA/Working/Tracking_CCTV/CCTV_Data/Video/'
OUTPUT_VIDEO_PATH = ROOT_DIR + '/output/'

""" LUCAS KANADE PARAMS """
# Lucas kanade params
lk_params = dict(winSize = (15, 15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

TRIANGLE_WIDE = 10

""" POSENET PARAMS """
POSENET_MODEL_NUM = 50
SCALE_FACTOR = 0.5
OUTPUT_STRIDE = 16
INPUT_IMG_TEST_DIR = '/media/hiep/DATA/Work_space/Tracking_CCTV/CCTV_Data/Image'
