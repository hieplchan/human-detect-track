from .decode import *
from .constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


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

def build_part_with_score_torch(score_threshold, local_max_radius, scores):
    lmd = 2 * local_max_radius + 1
    max_vals = F.max_pool2d(scores, lmd, stride=1, padding=1)
    max_loc = (scores == max_vals) & (scores >= score_threshold)
    max_loc_idx = max_loc.nonzero()
    scores_vec = scores[max_loc]
    sort_idx = torch.argsort(scores_vec, descending=True)
    return scores_vec[sort_idx], max_loc_idx[sort_idx]

def decode_multiple_poses(
        heatmaps_result, offsets, displacements_fwd, displacements_bwd, output_stride, draw_image,
        max_pose_detections=10, score_threshold=0.5, nms_radius=20, min_pose_score=0.5):
    '''
    scores: heatmap
    '''

    # print('----- Decode multi pose -----')

    part_scores, part_idx = build_part_with_score_torch(score_threshold, LOCAL_MAXIMUM_RADIUS, heatmaps_result)
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
    pose_scores = np.zeros(max_pose_detections)
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))

    for root_score, (root_id, root_coord_y, root_coord_x) in zip(part_scores, part_idx):
        root_coord = np.array([root_coord_y, root_coord_x])
        root_image_coords = root_coord * output_stride + offsets[root_id, root_coord_y, root_coord_x]

        if within_nms_radius_fast(
                pose_keypoint_coords[:pose_count, root_id, :], squared_nms_radius, root_image_coords):
            continue

        keypoint_scores, keypoint_coords = decode_pose(
            root_score, root_id, root_image_coords,
            heatmaps_result, offsets, output_stride,
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

        if pose_count >= max_pose_detections:
            break

    return pose_scores, pose_keypoint_scores, pose_keypoint_coords
