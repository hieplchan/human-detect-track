import timeit
import torch
import time
from posenet_minimum import posenet
from posenet_minimum.utils import params
import cv2

img_path = params.INPUT_IMG_TEST_DIR + '1.jpg'
cap = cv2.VideoCapture(params.VIDEO_PATH + params.VIDEO_NAME)

model = posenet.load_model(params.OUTPUT_STRIDE)

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    with torch.no_grad():


        while (True):
            start = time.time()
            input_image, draw_image, output_scale = posenet.read_imgfile(img_path, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)
            # input_image, draw_image, output_scale = posenet.read_cap(cap, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)
            print(draw_image.shape)
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            # print(timeit.timeit("model = posenet.load_model(params.OUTPUT_STRIDE)", setup="from __main__ import posenet, params", number=1)*1000)
            # print(timeit.timeit("input_image, draw_image, output_scale = posenet.read_imgfile(img_path, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)", setup="from __main__ import posenet, params, img_path", number=1)*1000)
            # print(timeit.timeit("heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)", setup="from __main__ import model, input_image", number=1)*1000)
            stop = time.time()
            print((stop - start)*1000)
            cv2.imshow('Test', draw_image)

    time.sleep(60)
