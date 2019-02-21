import numpy as np

from posenet import pose_constants
from posenet import params

def traverse_to_targ_keypoint(edge_id, source_keypoint, target_keypoint_id, scores, offsets, displacements):
    height = scores.shape[1]
    width = scores.shape[2]
    source_keypoint_indices = np.clip(np.round(source_keypoint / params.OUTPUT_STRIDE), a_min = 0, a_max = [height - 1, width - 1]).astype(np.int32)
    displaced_point = source_keypoint + displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]]
    displaced_point_indices = np.clip(np.round(displaced_point / params.OUTPUT_STRIDE), a_min = 0, a_max = [height - 1, width - 1]).astype(np.int32)
    score = scores[target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]]
    image_coord = displaced_point_indices * params.OUTPUT_STRIDE + offsets[target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]]
    return score, image_coord


def decode_pose(root_score, root_id, root_image_coord,
                scores, offsets,
                displacements_fwd,
                displacements_bwd):

    num_parts = scores.shape[0]
    num_edges = len(pose_constants.PARENT_CHILD_TUPLES)

    instance_keypoint_scores = np.zeros(num_parts)
    instance_keypoint_coords = np.zeros((num_parts, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coord

    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = pose_constants.PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
            instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(edge,
                                            instance_keypoint_coords[source_keypoint_id],
                                            target_keypoint_id,
                                            scores, offsets, displacements_bwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = pose_constants.PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
            instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(edge,
                                            instance_keypoint_coords[source_keypoint_id],
                                            target_keypoint_id,
                                            scores, offsets, displacements_fwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    return instance_keypoint_scores, instance_keypoint_coords
