import timeit
import time

from posenet_minimum import posenet
from posenet_minimum.utils import params

img_path = params.INPUT_IMG_TEST_DIR + '1.jpg'

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    model = posenet.load_model(params.OUTPUT_STRIDE)
    input_image, draw_image, output_scale = posenet.read_imgfile(img_path, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)
    # heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

    # print(timeit.timeit("model = posenet.load_model(params.OUTPUT_STRIDE)", setup="from __main__ import posenet, params", number=1)*1000)
    # print(timeit.timeit("input_image, draw_image, output_scale = posenet.read_imgfile(img_path, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)", setup="from __main__ import posenet, params, img_path", number=1)*1000)

    print(timeit.timeit("heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)", setup="from __main__ import model, input_image", number=100)*1000)

    time.sleep(60)
