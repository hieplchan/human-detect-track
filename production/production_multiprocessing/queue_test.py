import time
import cv2
import torch

import posenet
from posenet.params import SCALE_FACTOR, OUTPUT_STRIDE, THRESHOLD, TARGET_WIDTH, TARGET_HEIGHT, DEVICE
from posenet.decode_multi import decode_multiple_poses
import lucas_kanade

import torch.multiprocessing as mp


#Video load for test
cap = cv2.VideoCapture('/home/hiep/Desktop/Tracking_CCTV/CCTV_Data/Video/1.mp4')

# Posenet model setting and load
posenet.MODEL_PATH = '/home/hiep/Desktop/Tracking_CCTV/production/opticalflow_package/posenet/_models/mobilenet_v1_050_gpu.pth'

global model
model = posenet.load(posenet.MODEL_PATH, posenet.OUTPUT_STRIDE, posenet.DEVICE)
model.share_memory()

def getResultPointBox(queue):
    ''' Return good key point of multiple person '''
    global model

    while(True):
        input = queue.get(True)
        time_mark = time.time()
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input[0])
        print(str(input[1]) + ' - '+ str((time.time() - time_mark)*1000))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    image_queue = mp.Queue()

    with torch.no_grad():
        the_pool = mp.Pool(1, getResultPointBox,(image_queue,))

        for id in range(0, 100):
            res, draw_image = cap.read()
            input_image = posenet.process_input(draw_image, TARGET_WIDTH, TARGET_HEIGHT, DEVICE)
            input_image.share_memory_()
            image_queue.put([input_image, id])
            image_queue.put([input_image, id])
            image_queue.put([input_image, id])
            image_queue.put([input_image, id])
            time.sleep(0.5)
