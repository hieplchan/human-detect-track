import timeit
import torch
import time
import posenet
import cv2
import numpy as np

img_path = '/home/hiep/Desktop/Tracking_CCTV/CCTV_Data/Image/1.jpg'
cap = cv2.VideoCapture('/home/hiep/Desktop/Tracking_CCTV/CCTV_Data/Video/2.mp4')

# Posenet model setting and load
posenet.params.MODEL_PATH = '/home/hiep/Desktop/Tracking_CCTV/production/posenet_package/posenet/_models/mobilenet_v1_050_gpu.pth'
posenet.params.OUTPUT_STRIDE = 16
posenet.params.SCALE_FACTOR = 0.5
posenet.params.params_reconfig()
model = posenet.load_model(posenet.params.MODEL_PATH, posenet.params.OUTPUT_STRIDE, posenet.params.DEVICE)

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    with torch.no_grad():
        count = 0
        time_measure = []
        # input_image, draw_image = posenet.read_imgfile(img_path)

        # while (count == 0):
        while (True):

            input_image, draw_image = posenet.read_cap(cap, posenet.params.TARGET_WIDTH, posenet.params.TARGET_HEIGHT, posenet.params.DEVICE)
            start = time.time()
            points, boxs = posenet.getResultPointBox(model, input_image)
            # heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
            #
            # pose_scores, keypoint_scores, keypoint_coords, boxs = posenet.decode_multiple_poses(
            #                                                     heatmaps_result.squeeze(0),
            #                                                     offsets_result.squeeze(0),
            #                                                     displacement_fwd_result.squeeze(0),
            #                                                     displacement_bwd_result.squeeze(0),
            #                                                     posenet.params.OUTPUT_STRIDE,
            #                                                     draw_image,
            #                                                     max_pose_detections = 50,
            #                                                     score_threshold = posenet.params.THRESHOLD,
            #                                                     nms_radius = 50,
            #                                                     min_pose_score = posenet.params.THRESHOLD)

            stop = time.time()
            # decoded_image, keypoint = posenet.draw_keypoint(draw_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=params.THRESHOLD, min_part_score=params.THRESHOLD)
            #
            # for box in boxs:
            #     decoded_image = cv2.rectangle(decoded_image, (box[1], box[3]), (box[0], box[2]), (0,255,0), 3)
            #     # print(box)
            #
            # point_to_track = np.array([keypoint[idx].pt for idx in range(0, len(keypoint))])
            # # print(point_to_track.shape)
            #
            #
            # print((stop - start)*1000)
            time_measure.append((stop - start)*1000)
            #
            # # posenet.utils.show_image('draw_image', draw_image)
            # # posenet.utils.show_image('Pose estimation', decoded_image)
            # count = count + 1
            # # cv2.imwrite('output/' + str(count) + '.png', decoded_image)
            # video.write(decoded_image)
            # if (count == 700):
            #     video.release()
            #     exit()
            # print(count)
            print(sum(time_measure)/len(time_measure))

    time.sleep(60)
