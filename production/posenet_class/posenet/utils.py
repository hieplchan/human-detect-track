import cv2
import torch
import numpy as np

from posenet.models import MobileNetV1
from posenet import params
from posenet.decode_multi import decode_multiple_poses
from posenet.params import SCALE_FACTOR, OUTPUT_STRIDE, THRESHOLD, TARGET_WIDTH, TARGET_HEIGHT, DEVICE

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
    return model

def process_input(source_img, targer_width, target_height, device):
    source_img = cv2.normalize(source_img, None, 0, 255, cv2.NORM_MINMAX)
    input_img = cv2.resize(source_img, (targer_width, target_height), interpolation = cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, targer_width)
    input_img = torch.as_tensor(input_img, device = device)
    return input_img

def getResultPointBox(model, draw_image):
    ''' Return good key point of multiple person '''
    input_image = process_input(draw_image, TARGET_WIDTH, TARGET_HEIGHT, DEVICE)
    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

    pose_scores, keypoint_scores, keypoint_coords, boxs = decode_multiple_poses(
                                                        heatmaps_result.squeeze(0),
                                                        offsets_result.squeeze(0),
                                                        displacement_fwd_result.squeeze(0),
                                                        displacement_bwd_result.squeeze(0),
                                                        OUTPUT_STRIDE,
                                                        max_pose_detections = 50,
                                                        score_threshold = THRESHOLD,
                                                        nms_radius = 50,
                                                        min_pose_score = THRESHOLD)
    cv_keypoints = []
    keypoint_coords[:, :, :] = keypoint_coords[:, :, :]/SCALE_FACTOR

    for ii, score in enumerate(pose_scores):
        if score < THRESHOLD:
            continue

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < THRESHOLD:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10.))

    return cv_keypoints, boxs

def drawResultBox(draw_image, boxs):
    for box in boxs:
        draw_image = cv2.rectangle(draw_image, box[0], box[3], (0,255,0), 3)
    return draw_image

def drawResultPoint(draw_image, cv_keypoints):
    if cv_keypoints:
        draw_image = cv2.drawKeypoints(draw_image, cv_keypoints, outImage = np.array([]), color = (255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return draw_image
