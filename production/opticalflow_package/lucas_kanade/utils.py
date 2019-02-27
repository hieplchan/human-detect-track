import cv2
import numpy as np
from random import randint

def keypointlist2array(keypoints):
    if (keypoints != []):
        return np.array([keypoints[idx].pt for idx in range(0, len(keypoints))], dtype = np.float32).reshape(len(keypoints), 1, 2)
    else:
        return np.empty([1, 1, 2], dtype = np.float32)

def keypointlist2tuplelist(keypoints):
    return [keypoints[idx].pt for idx in range(0, len(keypoints))]

def array2keypointlist(point_array):
    if (type(point_array) is np.ndarray):
        return [cv2.KeyPoint(point[0][0], point[0][1], 10.) for point in point_array]
    else:
        return []

def tuplelist2keypointlist(point_tuple):
    return [cv2.KeyPoint(point[0], point[1], 10.) for point in point_tuple]

def drawResultPoint(draw_image, cv_keypoints, color = (255, 255, 0)):
    if cv_keypoints:
        draw_image = cv2.drawKeypoints(draw_image, cv_keypoints, outImage = np.array([]), color = color, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return draw_image

def drawResultBodiesPoint(draw_image, bodies_points):
    for idx in range(0, len(bodies_points)):
        draw_image = drawResultPoint(draw_image, tuplelist2keypointlist(bodies_points[idx]), (randint(0, 255), randint(0, 255), randint(0, 255)))
    return draw_image
