import time
import cv2
import torch
import torch.multiprocessing as mp


import posenet
from posenet.params import SCALE_FACTOR, OUTPUT_STRIDE, THRESHOLD, TARGET_WIDTH, TARGET_HEIGHT, DEVICE
import lucas_kanade


#Video load for test
cap = cv2.VideoCapture('/home/hiep/Desktop/Tracking_CCTV/CCTV_Data/Video/1.mp4')

# Posenet model setting and load
posenet.MODEL_PATH = '/home/hiep/Desktop/Tracking_CCTV/production/opticalflow_package/posenet/_models/mobilenet_v1_050_gpu.pth'

def getResultPointBox(model, input_image):
    ''' Return good key point of multiple person '''
    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

model = posenet.load(posenet.MODEL_PATH, posenet.OUTPUT_STRIDE, posenet.DEVICE)
model.share_memory()
tracktor = lucas_kanade.Lucas_Kanade(posenet.CAM_WIDTH, posenet.CAM_HEIGHT)

if __name__ == "__main__":
    with torch.no_grad():
        frame_num = 0
        while (True):
            res, draw_image = cap.read()
            input_image = posenet.process_input(draw_image, TARGET_WIDTH, TARGET_HEIGHT, DEVICE)
            input_image.share_memory_()
            start = time.time()
            processes = []

            for i in range(3): # No. of processes
                p = mp.Process(target=getResultPointBox, args=(model, input_image))
                p.start()
                processes.append(p)

            for p in processes: p.join()

            # posenet.getResultPointBox(model, draw_image)
            # posenet.getResultPointBox(model, draw_image)

            # if ((frame_num % 20) == 0):
            #     resultPoints, resultBoxs = posenet.getResultPointBox(model, draw_image)
            # elif ((frame_num % 20) == 1):
            #     tracktor.detectorUpdate(draw_image, resultPoints, resultBoxs)
            #     # decoded_image = posenet.drawResultBox(draw_image, resultBoxs)
            #     # decoded_image = lucas_kanade.drawResultBodiesPoint(draw_image, tracktor.bodies_points)
            # else:
            #     resultPoints = tracktor.pointTrackCal(draw_image)

            # decoded_image = lucas_kanade.drawResultPoint(draw_image, resultPoints)
            stop = time.time()
            print((stop - start)*1000)
            frame_num = frame_num + 1
            # cv2.imwrite('output/' + str(frame_num) + '.jpg', decoded_image)

            # cv2.imshow('decoded_image', decoded_image)
            # cv2.waitKey(1)

        time.sleep(60)
