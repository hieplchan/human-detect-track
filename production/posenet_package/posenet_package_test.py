import time
import cv2
import torch

import posenet

#Video load for test
cap = cv2.VideoCapture('/home/hiep/Desktop/Tracking_CCTV/CCTV_Data/Video/1.mp4')

# Posenet model setting and load
posenet.MODEL_PATH = '/home/hiep/Desktop/Tracking_CCTV/production/posenet_package/posenet/_models/mobilenet_v1_050_gpu.pth'

# posenet.params_reconfig()
model = posenet.load(posenet.MODEL_PATH, posenet.OUTPUT_STRIDE, posenet.DEVICE)

if __name__ == "__main__":
    with torch.no_grad():
        while (True):
            res, draw_image = cap.read()

            start = time.time()
            # Posenet
            resultPoints, resultBoxs = posenet.getResultPointBox(model, draw_image)
            boxs_image = posenet.drawResultBox(draw_image, resultBoxs)
            points_image = posenet.drawResultPoint(draw_image, resultPoints)

            stop = time.time()

            print((stop - start)*1000)

            cv2.imwrite('boxs_image.jpg', boxs_image)
            cv2.imwrite('points_image.jpg', points_image)

    time.sleep(60)
