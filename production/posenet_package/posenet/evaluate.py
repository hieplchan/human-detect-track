import cv2
import numpy as np

from posenet.decode_multi import decode_multiple_poses
from posenet.params import SCALE_FACTOR, OUTPUT_STRIDE, THRESHOLD, TARGET_WIDTH, TARGET_HEIGHT, DEVICE
from posenet.utils import process_input

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
            cv_keypoints.append(cv2.KeyPoint(kc[1]/2, kc[0]/2, 10.))

    return cv_keypoints, boxs

def drawResultBox(draw_image, boxs):
    for box in boxs:
        boxs_image = cv2.rectangle(draw_image, (box[1], box[3]), (box[0], box[2]), (0,255,0), 3)
    return boxs_image

def drawResultPoint(draw_image, cv_keypoints):
    if cv_keypoints:
        points_image = cv2.drawKeypoints(draw_image, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return points_image
