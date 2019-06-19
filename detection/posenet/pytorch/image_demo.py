import sys
sys.path.append('/home/hiep/pytorch-vision')

import cv2
import time
import argparse
import os
import torch

import posenet

import datetime
from posenet.constants import *
from posenet.utils import *

from utils.model_utils import load_checkpoint, model_report, model_save

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=100)
parser.add_argument('--scale_factor', type=float, default=0.5)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./images')
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model_report(model, input=(3,529,961))

    model = model.cuda()
    # model = model.cpu()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]


    start = time.time()
    # for f in filenames:
    input_image, draw_image, output_scale = posenet.read_imgfile_pytorch(
        args.image_dir + '/retail1.jpg', scale_factor=args.scale_factor, output_stride=output_stride)

    print(input_image.shape)
    input_image = torch.Tensor(input_image).cuda()
    # input_image = torch.Tensor(input_image).cpu()

    with torch.no_grad():
        for idx in range(0):
            start_time = datetime.datetime.utcnow().timestamp()
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
            # print('image_demo headmap' + str(heatmaps_result.shape))

            # heatmap_mask = heatmap_inspection(heatmaps_result) * 2
            # heatmap_mask[heatmap_mask > 255] = 255
            # gray_heatmap_img = cv2.cvtColor(draw_image, cv2.COLOR_BGR2GRAY)
            # # heatmap_mask = cv2.cvtColor(heatmap_mask, cv2.COLOR_GRAY2BGR)
            # heatmap_mask = heatmap_mask.astype(np.uint8)
            # print('******')
            # print(type(gray_heatmap_img[0][0]))
            # print(type(heatmap_mask[0][0]))
            # # test_heatmap = cv2.addWeighted(gray_heatmap_img,1,heatmap_mask,0.8,0)
            # # show_image('gray_heatmap_img', test_heatmap)
            #
            # model_time = datetime.datetime.utcnow().timestamp()
            #

            # pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            #     heatmaps_result.squeeze(0),
            #     offsets_result.squeeze(0),
            #     displacement_fwd_result.squeeze(0),
            #     displacement_bwd_result.squeeze(0),
            #     output_stride=output_stride,
            #     max_pose_detections=20,
            #     min_pose_score=0.1)

            # print((model_time - start_time)*1000)
            # print((decode_time - start_time)*1000)
            # print(heatmaps_result.squeeze(0).shape)
            # print(offsets_result.squeeze(0).shape)
            # print(displacement_fwd_result.squeeze(0).shape)
            # print(displacement_bwd_result.squeeze(0).shape)

            # keypoint_coords *= output_scale

            # if args.output_dir:
            # draw_image = posenet.draw_skel_and_kp(
            #     draw_image, pose_scores, keypoint_scores, keypoint_coords,
            #     min_pose_score=0.1, min_part_score=0.1)

                # cv2.imwrite(args.image_dir + '/test.png', draw_image)

            decode_time = datetime.datetime.utcnow().timestamp()
            print((decode_time - start_time)*1000)

        #     overlay_image = cv2.addWeighted(draw_image,0.6,mask,1,0)
            # cv2.imshow('result', overlay_image)
            # while(True):
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

        # if not args.notxt:
        #     print()
        #     print("Results for image: %s" % f)
        #     for pi in range(len(pose_scores)):
        #         if pose_scores[pi] == 0.:
        #             break
        #         print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
        #         for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
        #             print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

    # print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    print('------------')
    main()
