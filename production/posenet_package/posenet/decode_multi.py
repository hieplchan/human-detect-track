import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from posenet import params
from posenet import pose_constants
from posenet.decode import decode_pose

def within_nms_radius_fast(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis=1) <= squared_nms_radius)

def get_instance_score_fast(
        exist_pose_coords,
        squared_nms_radius,
        keypoint_scores, keypoint_coords):

    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > squared_nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)

def build_part_with_score_torch(score_threshold, local_max_radius, heatmaps_result):
    lmd = 2 * local_max_radius + 1
    max_vals = F.max_pool2d(heatmaps_result, lmd, stride=1, padding=1)
    max_loc = (heatmaps_result == max_vals) & (heatmaps_result >= score_threshold)
    max_loc_idx = max_loc.nonzero()
    scores_vec = heatmaps_result[max_loc]
    sort_idx = torch.argsort(scores_vec, descending=True)
    return scores_vec[sort_idx], max_loc_idx[sort_idx]

def decode_multiple_poses(heatmaps_result, offsets, displacements_fwd, displacements_bwd, draw_image,
                            max_pose_detections, score_threshold, nms_radius, min_pose_score):

    part_scores, part_idx = build_part_with_score_torch(score_threshold, pose_constants.LOCAL_MAXIMUM_RADIUS, heatmaps_result)

    part_scores = part_scores.cpu().numpy()
    part_idx = part_idx.cpu().numpy()

    heatmaps_result = heatmaps_result.cpu().numpy()
    height = heatmaps_result.shape[1] #68
    width = heatmaps_result.shape[2] #121

    # change dimensions from (x, h, w) to (x//2, h, w, 2) to allow return of complete coord array
    offsets = offsets.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    displacements_fwd = displacements_fwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    displacements_bwd = displacements_bwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))

    squared_nms_radius = nms_radius ** 2
    pose_count = 0
    boxs = []
    pose_scores = np.zeros(max_pose_detections)
    pose_keypoint_scores = np.zeros((max_pose_detections, pose_constants.NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((max_pose_detections, pose_constants.NUM_KEYPOINTS, 2))

    for root_score, (root_id, root_coord_y, root_coord_x) in zip(part_scores, part_idx):
        root_coord = np.array([root_coord_y, root_coord_x])
        root_image_coords = root_coord * params.OUTPUT_STRIDE + offsets[root_id, root_coord_y, root_coord_x]

        if within_nms_radius_fast(
                pose_keypoint_coords[:pose_count, root_id, :], squared_nms_radius, root_image_coords):
            continue

        keypoint_scores, keypoint_coords = decode_pose(root_score,
                                                root_id, root_image_coords,
                                                heatmaps_result, offsets,
                                                displacements_fwd, displacements_bwd)

        pose_score = get_instance_score_fast(
            pose_keypoint_coords[:pose_count, :, :], squared_nms_radius, keypoint_scores, keypoint_coords)

        # NOTE this isn't in the original implementation, but it appears that by initially ordering by
        # part scores, and having a max # of detections, we can end up populating the returned poses with
        # lower scored poses than if we discard 'bad' ones and continue (higher pose scores can still come later).
        # Set min_pose_score to 0. to revert to original behaviour
        if min_pose_score == 0. or pose_score >= min_pose_score:
            pose_scores[pose_count] = pose_score
            pose_keypoint_scores[pose_count, :] = keypoint_scores
            pose_keypoint_coords[pose_count, :, :] = keypoint_coords
            pose_count += 1
            # print(pose_score)
            boxs.append(getBoundingBoxPoints(keypoint_coords))

        # if pose_count >= max_pose_detections:
        #     break

    # print(pose_count)

    return pose_scores, pose_keypoint_scores, pose_keypoint_coords, boxs

def getBoundingBoxPoints(keypoint_coords):

    keypoint_coords = keypoint_coords/params.SCALE_FACTOR
    keypoint_coords = keypoint_coords.astype(np.int32)
    maxY = keypoint_coords[:,0].max()
    minY = keypoint_coords[:,0].min()
    maxX = keypoint_coords[:,1].max()
    minX = keypoint_coords[:,1].min()

    return [maxX, minX, maxY, minY]

def draw_skel_and_kp(draw_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score, min_part_score):

    out_img = draw_image
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(pose_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :]/params.SCALE_FACTOR, min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1]/params.SCALE_FACTOR, kc[0]/params.SCALE_FACTOR, 10. * ks))

    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img, cv_keypoints

def draw_keypoint(draw_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score, min_part_score):
    out_img = draw_image
    cv_keypoints = []

    keypoint_coords[:, :, :] = keypoint_coords[:, :, :]/params.SCALE_FACTOR

    for ii, score in enumerate(pose_scores):
        if score < min_pose_score:
            continue

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10.))

    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out_img, cv_keypoints

def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results

def lines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0)):
    return out_img
