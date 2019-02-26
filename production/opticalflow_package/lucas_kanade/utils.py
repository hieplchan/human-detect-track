import cv2
import numpy as np

def keypointlist2array(keypoint):
    if (keypoint != []):
        return np.array([keypoint[idx].pt for idx in range(0, len(keypoint))], dtype=np.float32).reshape(len(keypoint), 1, 2)
    else:
        return np.empty([1, 1, 2], dtype=np.float32)

def array2keypointlist(point_array):
    if (type(point_array) is np.ndarray):
        return [cv2.KeyPoint(point[0][0], point[0][1], 10.) for point in point_array]
    else:
        return []

def drawResultPoint(draw_image, cv_keypoints):
    if cv_keypoints:
        draw_image = cv2.drawKeypoints(draw_image, cv_keypoints, outImage = np.array([]), color = (255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return draw_image
