import torch
import numpy as np

CAM_WIDTH = 1920
CAM_HEIGHT = 1080
TARGET_WIDTH = (int(CAM_WIDTH) // OUTPUT_STRIDE) * OUTPUT_STRIDE + 1
TARGET_HEIGHT = (int(CAM_HEIGHT) // OUTPUT_STRIDE) * OUTPUT_STRIDE + 1

DEVICE = torch.device('cuda')
MODEL_PATH = ''

POSENET_MODEL_NUM = 50
SCALE_FACTOR = 0.5
OUTPUT_STRIDE = 16
THRESHOLD = 0.2
SCALE = np.array([CAM_HEIGHT/TARGET_HEIGHT, CAM_WIDTH/TARGET_WIDTH])

def params_reconfig():
    ''' Calculate valid frame resolution for Posenet stride'''
    TARGET_WIDTH = (int(CAM_WIDTH) // OUTPUT_STRIDE) * OUTPUT_STRIDE + 1
    TARGET_HEIGHT = (int(CAM_HEIGHT) // OUTPUT_STRIDE) * OUTPUT_STRIDE + 1
    np.array([CAM_HEIGHT/TARGET_HEIGHT, CAM_WIDTH/TARGET_WIDTH])
