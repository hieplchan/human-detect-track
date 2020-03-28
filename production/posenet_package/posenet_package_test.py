import time
import cv2
import torch

import posenet

#Video load for test
cap = cv2.VideoCapture('/home/hiep/Desktop/Tracking_CCTV/CCTV_Data/Video/1.mp4')

# Posenet model setting and load
posenet.MODEL_PATH = '/home/hiep/Desktop/Tracking_CCTV/production/posenet_package/posenet/_models/mobilenet_v1_050_gpu.pth'

model = posenet.load(posenet.MODEL_PATH, posenet.OUTPUT_STRIDE, posenet.DEVICE)

if __name__ == "__main__":
    with torch.no_grad():
        frame_num = 0
        while (True):
            res, draw_image = cap.read()
            start = time.time()

            resultPoints, resultBoxs = posenet.getResultPointBox(model, draw_image)
            decoded_image = posenet.drawResultBox(draw_image, resultBoxs)
            decoded_image = posenet.drawResultPoint(draw_image, resultPoints)

            stop = time.time()
            print((stop - start)*1000)

            frame_num = frame_num + 1
            cv2.imwrite('output/' + str(frame_num) + '.jpg', decoded_image)

        time.sleep(60)
