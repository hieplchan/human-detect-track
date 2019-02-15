import timeit
import time

from posenet_minimum import posenet
from posenet_minimum.utils import params

img_path = params.INPUT_IMG_TEST_DIR + '1.jpg'

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    # model = posenet.load_model(params.OUTPUT_STRIDE)
    print(timeit.timeit("model = posenet.load_model(params.OUTPUT_STRIDE)", setup="from __main__ import posenet, params", number=1)*1000)


    # input_image, draw_image, output_scale = posenet.read_imgfile(img_path, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)

    # input_image = torch.Tensor(input_image).cuda()
    # heatmaps_result = model(input_image)

    # print((model_load_time - start_time)*1000)
    # print((read_imgfile_time - model_load_time)*1000)

    # while(True):
    #     pass
    time.sleep(60)
