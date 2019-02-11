import torch

from posenet_lucas_kanade import posenet
from posenet_lucas_kanade.posenet.constants import *
from posenet_lucas_kanade.posenet.utils import *

from posenet_lucas_kanade.utils.params import *

def main():
    # Load posenet model - can be cpu() or cuda()
    model = posenet.load_model(POSENET_MODEL_NUM)
    model = model.cpu()

    # Output stride - default 16
    output_stride = model.output_stride

    # Load image in folder for testing:
    filenames = [f.path for f in os.scandir(INPUT_IMG_TEST_DIR) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    for f in filenames:
        # Convert draw_image (HEIGHT, WIDTH, 3) to input_image (1, 3, ~HEIGHT*SCALE_FACTOR, ~WIDTH*SCALE_FACTOR) - scale [~1/SCALE_FACTOR ~1/SCALE_FACTOR]
        input_image, draw_image, output_scale = posenet.read_imgfile(f, scale_factor = SCALE_FACTOR, output_stride = OUTPUT_STRIDE)

        # Pytorch in reduce memory mode
        with torch.no_grad():
            #Convert to torch data type torch.Size([1, 3, ~HEIGHT*SCALE_FACTOR, ~WIDTH*SCALE_FACTOR])
            input_image = torch.Tensor(input_image).cpu()

            # Posenet compute - return heatmap torch.Size([1, 17, ???, ???])
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
            print(heatmaps_result.shape)


if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    main()
