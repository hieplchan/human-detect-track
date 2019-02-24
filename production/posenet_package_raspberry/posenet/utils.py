import cv2
import torch
import numpy as np

from posenet.models import MobileNetV1
from posenet import params

def params_reconfig():
    ''' Calculate valid frame resolution for Posenet stride'''
    TARGET_WIDTH = (int(CAM_WIDTH) // OUTPUT_STRIDE) * OUTPUT_STRIDE + 1
    TARGET_HEIGHT = (int(CAM_HEIGHT) // OUTPUT_STRIDE) * OUTPUT_STRIDE + 1
    SCALE = np.array([CAM_HEIGHT/TARGET_HEIGHT, CAM_WIDTH/TARGET_WIDTH])

def load(model_path, output_stride, device):
    ''' Load posenet model to device (CPU or GPU) '''
    model = MobileNetV1(output_stride = output_stride)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.to(device)
    model.half()
    return model

def process_input(source_img, targer_width, target_height, device):
    source_img = cv2.normalize(source_img, None, 0, 255, cv2.NORM_MINMAX)
    input_img = cv2.resize(source_img, (targer_width, target_height), interpolation = cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, targer_width)
    input_img = torch.as_tensor(input_img, device = device)
    input_img = input_img.half()
    return input_img
