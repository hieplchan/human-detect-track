import tensorflow as tf
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

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def main():

    with tf.Session(config=config) as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture('/home/hiep/Tracking_CCTV/CCTV_Data/video/' + args.cam_id)
        # cap = cv2.VideoCapture('/home/hiep/Tracking_CCTV/CCTV_Data/video/20190114155554.mp4')
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        # video = cv2.VideoWriter('video.avi',-1,1,(args.cam_width,args.cam_height))
        video = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,(args.cam_width,args.cam_height))

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
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