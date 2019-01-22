import torch
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=str, default='20190114155554.mp4')
parser.add_argument('--cam_width', type=int, default=1920)
parser.add_argument('--cam_height', type=int, default=1080)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    # model = model.cpu()
    output_stride = model.output_stride

    # cap = cv2.VideoCapture(args.cam_id)
    cap = cv2.VideoCapture('/home/hiep/Tracking_CCTV/CCTV_Data/video/' + args.cam_id)
    video = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,(args.cam_width,args.cam_height))

    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()
            # input_image = torch.Tensor(input_image).cpu()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        cv2.imshow('posenet', overlay_image)
        video.write(overlay_image)
        frame_count += 1
        print(frame_count)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))
    video.release()

if __name__ == "__main__":
    main()