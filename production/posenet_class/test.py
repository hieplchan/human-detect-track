import time
import cv2
import torch

import posenet
import lucas_kanade

#Video load for test
cap = cv2.VideoCapture('D:/Work_space/Tracking_CCTV/CCTV_Data/Video/1.mp4')

# Posenet model setting and load
posenet_model_path = 'D:/Work_space/Tracking_CCTV/production/posenet_class/posenet/_models/mobilenet_v1_050_cpu.pth'

detector = posenet.Detector(1920, 1080)
detector.load_model(posenet_model_path)
tracktor = lucas_kanade.Tracktor(1920, 1080)

if __name__ == "__main__":
    with torch.no_grad():
        frame_num = 0
        while (True):
            res, draw_image = cap.read()
            start = time.time()

            if ((frame_num % 20) == 0):
                resultPoints, resultBoxs = detector.getFrameResult(draw_image)
            # resultPoints2, resultBoxs2 = detector_2.getFrameResult(draw_image)
            elif ((frame_num % 20) == 1):
                tracktor.detectorUpdate(draw_image, resultPoints, resultBoxs)
                decoded_image = posenet.drawResultBox(draw_image, resultBoxs)
                # decoded_image = lucas_kanade.drawResultBodiesPoint(draw_image, tracktor.bodies_points)
            else:
                resultPoints = tracktor.pointTrackCal(draw_image)

            decoded_image = lucas_kanade.drawResultPoint(draw_image, resultPoints)
            stop = time.time()
            print((stop - start)*1000)
            frame_num = frame_num + 1
            # # # cv2.imwrite('output/' + str(frame_num) + '.jpg', decoded_image)
            #
            cv2.imshow('decoded_image', decoded_image)
            cv2.waitKey(1)

        # time.sleep(60)
