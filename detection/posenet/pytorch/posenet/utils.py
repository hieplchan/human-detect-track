import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import posenet.constants

def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    print(target_width)
    print(target_height)
    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0

    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, source_img, scale

def _process_input_pytorch(source_img, scale_factor=1.0, output_stride=16):
    print('Origin Width: {}, Height: {}'.format(source_img.width, source_img.height))
    # Size Calculate
    target_width, target_height = valid_resolution(
        source_img.width * scale_factor, source_img.height * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.height / target_height, source_img.width / target_width])

    print('Targer Width: {}, Height: {}'.format(target_width, target_height))

    """
    Transform Block
    """
    # Normalize image = (image - mean) / std
    r_mean, g_mean, b_mean = (0.5,  0.5, 0.5)
    r_std, g_std, b_std = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=(r_mean, g_mean, b_mean),
                                     std=(r_std, g_std, b_std))
    transform = transforms.Compose([
                transforms.Resize((target_height, target_width), interpolation= Image.BILINEAR), #Image.LANCZOS, Image.NEAREST, Image.BICUBIC, Image.BILINEAR
                transforms.ToTensor(),
                normalize])
    input_img = transform(source_img)
    input_tensor = torch.stack([input_img], 0)
    print('Tensor Image Shape: {}'.format(input_img.shape))
    source_img = np.array(source_img)
    source_img = source_img[:, :, ::-1].copy()
    """
    End Transform Block
    """

    return input_tensor, source_img, scale


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)

def read_cap_pytorch(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input_pytorch(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)

def read_imgfile_pytorch(path, scale_factor=1.0, output_stride=16):
    img = Image.open(path)
    img = img.resize((1920, 1080), Image.BILINEAR)
    return _process_input_pytorch(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):

    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def show_image(name, img):
    cv2.imshow(name, img)
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def heatmap_inspection(heatmap):
        print("heatmap: " + str(heatmap[0][0].shape))
        np_heatmap = heatmap[0].cpu().numpy() * 255
        np_heatmap_mask = np.zeros(np_heatmap[0].shape, np.float)
        # print(np_heatmap_mask.shape)
        # print(np_heatmap[1].shape)
        for i in range(len(np_heatmap)):
            # print(i)
            np_heatmap_mask += np_heatmap[i]
            # show_image(PART_NAMES[i], np_heatmap[i])
        np_heatmap_mask = np_heatmap_mask/np.max(np_heatmap_mask)*255
        # show_image('heatmap scale factor', np_heatmap_mask.astype(np.uint8))
        # print(np.max(np_heatmap_mask))
        # np_heatmap_mask = cv2.resize(np_heatmap_mask, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        # show_image('heatmap', np_heatmap_mask)
        return np_heatmap_mask
