import numpy as np
import time
import cv2

""" LOGGING PARAMS """
from optical_flow_tracking import LOG_DIR
import logging
logger = logging.getLogger('process_time')
logger.setLevel(logging.DEBUG)
# Process tiem logger
process_time_log_handle = logging.FileHandler(LOG_DIR + 'process_time_log.csv')
process_time_log_handle.setLevel(logging.DEBUG)
process_time_log_handle.setFormatter(logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s', '%Y-%m-%d %H:%M:%S'))
logger.addHandler(process_time_log_handle)

""" VIDEO PARAMS """
VIDEO_NAME = "1.mp4"
VIDEO_PATH = "/media/hiep/DATA/Working/Tracking_CCTV/CCTV_Data/Video/"

""" LUCAS KANADE PARAMS """
# Lucas kanade params
lk_params = dict(winSize = (15, 15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
