import timeit
import torch
import time
from posenet_minimum import posenet
from posenet_minimum.utils import params
import cv2

img_path = params.INPUT_IMG_TEST_DIR + '1.jpg'
cap = cv2.VideoCapture(params.VIDEO_PATH + params.VIDEO_NAME)
import numpy as np

model = posenet.load_model(params.OUTPUT_STRIDE)

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    with torch.no_grad():
        count = 0
        time_measure = []
        # input_image, draw_image, output_scale = posenet.read_imgfile(img_path, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)
        # while (count == 0):
        while (True):
            input_image, draw_image, output_scale = posenet.read_cap(cap, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)

            start = time.time()
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            # print(timeit.timeit("model = posenet.load_model(params.OUTPUT_STRIDE)", setup="from __main__ import posenet, params", number=1)*1000)
            # print(timeit.timeit("input_image, draw_image, output_scale = posenet.read_imgfile(img_path, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)", setup="from __main__ import posenet, params, img_path", number=1)*1000)
            # print(timeit.timeit("heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)", setup="from __main__ import model, input_image", number=1)*1000)

            pose_scores, keypoint_scores, keypoint_coords, boxs = posenet.decode_multiple_poses(
                                                                heatmaps_result.squeeze(0),
                                                                offsets_result.squeeze(0),
                                                                displacement_fwd_result.squeeze(0),
                                                                displacement_bwd_result.squeeze(0),
                                                                params.OUTPUT_STRIDE,
                                                                draw_image,
                                                                max_pose_detections = 50,
                                                                score_threshold = params.THRESHOLD,
                                                                nms_radius = 50,
                                                                min_pose_score=0.5)

            stop = time.time()
            # decoded_image, keypoint = posenet.draw_skel_and_kp(draw_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.05, min_part_score=0.05)
            #
            # for box in boxs:
            #     decoded_image = cv2.rectangle(decoded_image, (box[1], box[3]), (box[0], box[2]), (0,255,0), 3)
            #     print(box)
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
            # cv2.imwrite('output/' + str(count) + '.png', decoded_image)
            # # print(count)
            print(sum(time_measure)/len(time_measure))

    time.sleep(60)
