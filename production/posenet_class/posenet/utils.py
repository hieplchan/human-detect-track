import cv2
import torch
import numpy as np

def process_input(source_img, targer_width, target_height, device):
    source_img = cv2.normalize(source_img, None, 0, 255, cv2.NORM_MINMAX)
    input_img = cv2.resize(source_img, (targer_width, target_height), interpolation = cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, targer_width)
    input_img = torch.as_tensor(input_img, device = device)
    return input_img

def drawResultBox(draw_image, boxs):
    img = draw_image
    for box in boxs:
        img = cv2.rectangle(img, box[0], box[3], (0,255,0), 3)
    return img

def drawResultPoint(draw_image, cv_keypoints):
    img = draw_image
    if cv_keypoints:
        img = cv2.drawKeypoints(img, cv_keypoints, outImage = np.array([]), color = (255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img
