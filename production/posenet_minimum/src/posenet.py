import torch

from posenet_minimum import posenet
from posenet_minimum.utils import params

def main():
    model = posenet.load_model(params.POSENET_MODEL_NUM, params.OUTPUT_STRIDE)
    model = model.cuda()

    # img_path = params.INPUT_IMG_TEST_DIR + '1.jpg'
    # print(img_path)
    # input_image, draw_image, output_scale = posenet.read_imgfile(img_path, scale_factor = params.SCALE_FACTOR, output_stride = params.OUTPUT_STRIDE)
    # input_image = torch.Tensor(input_image).cuda()
    # heatmaps_result = model(input_image)

    while(True):
        pass

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    main()
