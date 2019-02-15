import cv2
import torch
import numpy as np

from posenet_minimum.utils import params

def _process_input(source_img, scale_factor, output_stride):
    input_img = cv2.resize(source_img, (params.TARGET_WIDTH, params.TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
#
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, params.TARGET_HEIGHT, params.TARGET_WIDTH)
    input_img = torch.from_numpy(input_img).float().to(params.device)
    return input_img, source_img, params.SCALE


def read_cap(cap, scale_factor, output_stride):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor, output_stride):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)
