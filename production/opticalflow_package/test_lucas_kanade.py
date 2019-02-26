import time
import cv2
import torch

import posenet
import lucas_kanade

#Video load for test
cap = cv2.VideoCapture('/media/hiep/DATA/Work_space/Tracking_CCTV/CCTV_Data/Video/pes3.mp4')

# Posenet model setting and load
posenet.MODEL_PATH = '/media/hiep/DATA/Work_space/Tracking_CCTV/production/posenet_package/posenet/_models/mobilenet_v1_050_gpu.pth'

model = posenet.load(posenet.MODEL_PATH, posenet.OUTPUT_STRIDE, posenet.DEVICE)
tracktor = lucas_kanade.Lucas_Kanade(posenet.CAM_WIDTH, posenet.CAM_HEIGHT)

if __name__ == "__main__":
    with torch.no_grad():
        frame_num = 0
        res, draw_image = cap.read()
        resultPoints, resultBoxs = posenet.getResultPointBox(model, draw_image)
        tracktor.frameUpdate(draw_image)
        tracktor.pointUpdate(resultPoints)

        while (True):
            res, draw_image = cap.read()
            start = time.time()

            if ((frame_num % 30) == 0):
                    resultPoints, resultBoxs = posenet.getResultPointBox(model, draw_image)
                    tracktor.frameUpdate(draw_image)
                    tracktor.pointUpdate(resultPoints)

            resultPoints = tracktor.pointTrackCal(draw_image)
            # decoded_image = posenet.drawResultBox(draw_image, resultBoxs)
            # decoded_image = posenet.drawResultPoint(draw_image, resultPoints)
            decoded_image = lucas_kanade.drawResultPoint(draw_image, resultPoints)

            stop = time.time()
            print((stop - start)*1000)
            frame_num = frame_num + 1
            # cv2.imwrite('output/' + str(frame_num) + '.jpg', decoded_image)

            cv2.imshow('decoded_image', decoded_image)
            cv2.waitKey(1)

        time.sleep(60)
