import time
import cv2
import torch
import torch.multiprocessing as mp

import posenet
from posenet.params import SCALE_FACTOR, OUTPUT_STRIDE, THRESHOLD, TARGET_WIDTH, TARGET_HEIGHT, DEVICE
from posenet.decode_multi import decode_multiple_poses
import lucas_kanade

#Video load for test
cap = cv2.VideoCapture('/media/hiep/DATA/Work_space/Tracking_CCTV/CCTV_Data/Video/1.mp4')

# Posenet model setting and load
posenet.MODEL_PATH = '/media/hiep/DATA/Work_space/Tracking_CCTV/production/opticalflow_package/posenet/_models/mobilenet_v1_050_gpu.pth'

global model
model = posenet.load(posenet.MODEL_PATH, posenet.OUTPUT_STRIDE, posenet.DEVICE)
model.share_memory()

def getResultPointBox(input):
    ''' Return good key point of multiple person '''
    time_mark = time.time()
    # print(input[2])
    global model
    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input[0])
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
    print((time.time() - time_mark)*1000)

    # return [cv_keypoints, boxs]

tracktor = lucas_kanade.Lucas_Kanade(posenet.CAM_WIDTH, posenet.CAM_HEIGHT)

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    with torch.no_grad():
        # frame_num = 0
        array_of_input_image = []
        for i in range(0, 500):
            res, draw_image = cap.read()
            input_image = posenet.process_input(draw_image, TARGET_WIDTH, TARGET_HEIGHT, DEVICE)
            array_of_input_image.append([input_image, i])

        print('Load done')
        print(len(array_of_input_image))
        start = time.time()

        with mp.Pool(processes=1) as p:
            p.map(getResultPointBox, array_of_input_image)

        # with mp.Pool(processes=1) as p:
        #     p.apply_async(getResultPointBox, array_of_input_image)
        #
        # p.close()
        # p.join()

        # for input in array_of_input_image:
        #     getResultPointBox(input)

        stop = time.time()
        print((stop - start)*1000)

        # while (True):
        #
        #     start = time.time()
        #     processes = []
        #
        #     for i in range(3): # No. of processes
        #         p = mp.Process(target=getResultPointBox, args=(model, input_image))
        #         p.start()
        #         processes.append(p)
        #
        #     for p in processes: p.join()
        #
        #     # posenet.getResultPointBox(model, draw_image)
        #     # posenet.getResultPointBox(model, draw_image)
        #
        #     # if ((frame_num % 20) == 0):
        #     #     resultPoints, resultBoxs = posenet.getResultPointBox(model, draw_image)
        #     # elif ((frame_num % 20) == 1):
        #     #     tracktor.detectorUpdate(draw_image, resultPoints, resultBoxs)
        #     #     # decoded_image = posenet.drawResultBox(draw_image, resultBoxs)
        #     #     # decoded_image = lucas_kanade.drawResultBodiesPoint(draw_image, tracktor.bodies_points)
        #     # else:
        #     #     resultPoints = tracktor.pointTrackCal(draw_image)
        #
        #     # decoded_image = lucas_kanade.drawResultPoint(draw_image, resultPoints)
        #     stop = time.time()
        #     print((stop - start)*1000)
        #     frame_num = frame_num + 1
        #     # cv2.imwrite('output/' + str(frame_num) + '.jpg', decoded_image)
        #
        #     # cv2.imshow('decoded_image', decoded_image)
        #     # cv2.waitKey(1)
        #
        # time.sleep(60)
