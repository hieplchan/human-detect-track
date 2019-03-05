import cv2
import torch
import numpy as np

from posenet.models import MobileNetV1
from posenet.decode_multi import decode_multiple_poses
from posenet.utils import process_input

def getResultPointBox(model, draw_image, targer_width, target_height, scale_factor, output_stride, threshold, device):
    ''' Return good key point of multiple person '''
    input_image = process_input(draw_image, targer_width, target_height, device)
    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

    pose_scores, keypoint_scores, keypoint_coords, boxs = decode_multiple_poses(
                                                        heatmaps_result.squeeze(0),
                                                        offsets_result.squeeze(0),
                                                        displacement_fwd_result.squeeze(0),
                                                        displacement_bwd_result.squeeze(0),
                                                        scale_factor,
                                                        output_stride,
                                                        max_pose_detections = 50,
                                                        score_threshold = threshold,
                                                        nms_radius = 50,
                                                        min_pose_score = threshold)
    cv_keypoints = []
    keypoint_coords[:, :, :] = keypoint_coords[:, :, :]/scale_factor

    for ii, score in enumerate(pose_scores):
        if score < threshold:
            continue

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < threshold:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10.))

    return cv_keypoints, boxs

class Detector:
    ''' Posenet person detector worker '''

    def __init__(self, width, height, scale_factor = 0.5, output_stride = 16, threshold = 0.4):
        self.cam_width = width
        self.cam_height = height
        self.scale_factor = scale_factor
        self.output_stride = output_stride
        self.threshold = threshold
        self.device = torch.device('cpu')

        # Calculate compatible size for mobilenetv1
        self.target_width = (int(self.cam_width * self.scale_factor) // self.output_stride) * self.output_stride + 1
        self.target_height = (int(self.cam_height * self.scale_factor) // self.output_stride) * self.output_stride + 1
        self.scale = np.array([self.cam_height/self.target_height, self.cam_width/self.target_width])

    def load_model(self, model_path):
        self.model_path = model_path
        self.model = MobileNetV1(output_stride = self.output_stride)
        self.model.load_state_dict(torch.load(self.model_path, map_location = self.device))
        self.model.to(self.device)

    def reload_model(model_path):
        ''' Reload model if params change, dispose old model memory '''
        pass

    def getFrameResult(self, draw_image):
        ''' Process one frame and return detected points, boxs '''
        points, boxs = getResultPointBox(self.model, draw_image, self.target_width, self.target_height, self.scale_factor, self.output_stride, self.threshold, self.device)
        return points, boxs
