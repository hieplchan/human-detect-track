import time

from optical_flow_tracking import LOG_DIR
import logging
logging.basicConfig(filename = LOG_DIR + 'point_tracking.log',level=logging.DEBUG)

""" VIDEO PARAMETER """

VIDEO_NAME = "1.mp4"
VIDEO_PATH = "/media/hiep/DATA/Working/Tracking_CCTV/CCTV_Data/Video/"
