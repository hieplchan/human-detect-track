import os
import cv2
import time
import torch

import posenet
from posenet.params import SCALE_FACTOR, OUTPUT_STRIDE, THRESHOLD, TARGET_WIDTH, TARGET_HEIGHT, DEVICE
from posenet.decode_multi import decode_multiple_poses
import lucas_kanade

import torch.multiprocessing as mp

#Video load for test
cap = cv2.VideoCapture('/home/hiep/Desktop/Tracking_CCTV/CCTV_Data/Video/4.mp4')

# Posenet model setting and load
posenet.MODEL_PATH = '/home/hiep/Desktop/Tracking_CCTV/production/opticalflow_package/posenet/_models/mobilenet_v1_050_gpu.pth'

def detector_worker(queue):
    ''' Return good key point of multiple person '''
    model = posenet.load(posenet.MODEL_PATH, posenet.OUTPUT_STRIDE, posenet.DEVICE)

    while(True):
        time_mark = time.time()
        print('Item in queue get: ' + str(queue.qsize()))
        if (queue.empty()):
            print('Detector Free')
            continue
        input = queue.get(block = True)

        #region Posenet Decode
        input_image = posenet.process_input(input[0], TARGET_WIDTH, TARGET_HEIGHT, DEVICE)
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
        #endregion

        # return [cv_keypoints, boxs]

        print('Pose process: ' + str(os.getpid()) + ' - ' + str(input[1]) + ' - ' + str((time.time() - time_mark)*1000))

if __name__ == "__main__":
    mp.set_start_method('spawn', force = True)
    detector_input_queue = mp.Queue(maxsize = 10)
    frame_id = 0

    with torch.no_grad():
        the_pool = mp.Pool(1, detector_worker, (detector_input_queue,))
        while (cap.isOpened()):
            res, draw_image = cap.read()
            if (res != 1):
                print('Video ended')
                time.sleep(100)
                break

            res, draw_image = cap.read()
            print('Main process: ' + str(os.getpid()))
            detector_input_queue.put(obj = [draw_image, frame_id], block = True)
            print('detector_input_queue put size: ' + str(detector_input_queue.qsize()))
            frame_id += 1
