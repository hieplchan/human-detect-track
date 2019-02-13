import torch
import math
import cv2
import numpy as np

from posenet_lucas_kanade import posenet
from posenet_lucas_kanade.posenet.constants import *
from posenet_lucas_kanade.posenet.utils import *

from posenet_lucas_kanade.utils.params import *

# Load posenet model - can be cpu() or cuda()
model = posenet.load_model(POSENET_MODEL_NUM, OUTPUT_STRIDE)
model = model.cpu()

def show_image(name, img):
    cv2.imshow(name, img)
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():

    # Output stride - default 16
    output_stride = model.output_stride

    ''' Image folder test '''
    filenames = [f.path for f in os.scandir(INPUT_IMG_TEST_DIR) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    for f in filenames:
        # Convert draw_image (HEIGHT, WIDTH, 3) to input_image (1, 3, ~HEIGHT*SCALE_FACTOR, ~WIDTH*SCALE_FACTOR) - scale [~1/SCALE_FACTOR ~1/SCALE_FACTOR]
        input_image, draw_image, output_scale = posenet.read_imgfile(f, scale_factor = SCALE_FACTOR, output_stride = OUTPUT_STRIDE)
        main_processing(input_image, draw_image, output_scale)

    ''' Video file test '''
    # print(VIDEO_PATH + VIDEO_NAME)
    # cap = cv2.VideoCapture(VIDEO_PATH + VIDEO_NAME)
    # cap.set(3, CAM_WIDTH)
    # cap.set(4, CAM_HEIGHT)
    # while cap.isOpened():
    #     input_image, draw_image, output_scale = posenet.read_cap(cap, scale_factor = SCALE_FACTOR, output_stride = OUTPUT_STRIDE)
    #     main_processing(input_image, draw_image, output_scale)
    # video.release()

def feature_inspection(features, draw_image):
    print('----- Features inspection -----')
    print(features.shape)
    print(features[0][1].shape)

    for i in range(0, 256):
        print('Feature num: ' + str(i))
        np_features_all_mask = np.zeros([features[0][1].shape[0], features[0][1].shape[1], 1])
        np_features_all_mask[:,:,0] = features[0][i]
        np_features_all_mask = np_features_all_mask/np.max(np_features_all_mask)*255
        np_features_all_mask = np_features_all_mask.astype(np.uint8)
        np_features_all_mask = cv2.resize(np_features_all_mask, (draw_image.shape[1], draw_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        np_features_all_mask = cv2.cvtColor(np_features_all_mask, cv2.COLOR_GRAY2BGR)
        overlay_img = cv2.addWeighted(draw_image, 0.3, np_features_all_mask, 0.8, 0)
        overlay_img = cv2.resize(overlay_img, (640, 480), interpolation=cv2.INTER_NEAREST)
        show_image('Features inspection', overlay_img)


def heatmap_inspection(heatmaps_result, draw_image, scale_factor, output_stride):
    # print('----- heatmap_inspection -----')
    # Get heatmap above threshole
    np_heatmap = heatmaps_result
    np_heatmap = np_heatmap[0].cpu().numpy()
    below_threshold_indices = np_heatmap < THRESHOLD
    np_heatmap[below_threshold_indices] = 0

    ''' Face mask '''
    np_heatmap_face_mask = np.zeros(np_heatmap[0].shape, np.float)
    np_heatmap_face_mask = np_heatmap[0] + np_heatmap[1] + np_heatmap[2] + np_heatmap[3] + np_heatmap[4]

    ''' All mask'''
    np_heatmap_all_mask = np_heatmap_face_mask
    for i in range(4, len(np_heatmap)):
        np_heatmap_all_mask += np_heatmap[i]

    # # Equal mask
    # ispeople_indices = np_heatmap_all_mask > 0
    # np_heatmap_all_mask[ispeople_indices] = 255
    # Not equal mask
    np_heatmap_all_mask = np_heatmap_all_mask/np.max(np_heatmap_all_mask)*255
    np_heatmap_all_mask = np_heatmap_all_mask.astype(np.uint8)
    np_heatmap_all_mask = cv2.resize(np_heatmap_all_mask, (draw_image.shape[1], draw_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    np_heatmap_all_mask = cv2.cvtColor(np_heatmap_all_mask, cv2.COLOR_GRAY2BGR)
    overlay_img = cv2.addWeighted(draw_image, 0.3, np_heatmap_all_mask, 0.8, 0)
    # show_image('Overlay picture', overlay_img)
    return overlay_img

def decode_inspection(heatmaps_result, draw_image, scale_factor, output_stride, offsets_result, displacement_fwd_result, displacement_bwd_result, overlay_img):
    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                                                        heatmaps_result.squeeze(0),
                                                        offsets_result.squeeze(0),
                                                        displacement_fwd_result.squeeze(0),
                                                        displacement_bwd_result.squeeze(0),
                                                        output_stride,
                                                        draw_image,
                                                        max_pose_detections = 50,
                                                        score_threshold = THRESHOLD,
                                                        nms_radius = 50,
                                                        min_pose_score=0.05)

    decoded_image = posenet.draw_skel_and_kp(overlay_img, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.25, min_part_score=0.25)
    show_image('Pose estimation', decoded_image)

def main_processing(input_image, draw_image, output_scale):

    # Pytorch in reduce memory mode
    with torch.no_grad():
        # Convert to torch data type torch.Size([1, 3, ~HEIGHT*SCALE_FACTOR, ~WIDTH*SCALE_FACTOR])
        input_image = torch.Tensor(input_image).cpu()

        start_time = time.time()
        # Posenet compute - return heatmap torch.Size([1, 17, int(HEIGHT*SCALE_FACTOR/OUTPUT_STRIDE + 1), int(WIDTH*SCALE_FACTOR/OUTPUT_STRIDE + 1)])
        features, heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

        heatmaps_result1 = heatmaps_result
        stop_time = time.time()
        print('Compute time: ' + str((stop_time - start_time)*1000))

        # Features inspection
        feature_inspection(features, draw_image)

        # Heatmap result inspection
        # overlay_img = heatmap_inspection(heatmaps_result, draw_image, SCALE_FACTOR, OUTPUT_STRIDE)

        # Decode human body inspection
        # decode_inspection(heatmaps_result1, draw_image, SCALE_FACTOR, OUTPUT_STRIDE, offsets_result, displacement_fwd_result, displacement_bwd_result, overlay_img)

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    main()
