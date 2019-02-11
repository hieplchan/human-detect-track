import torch
import math
import cv2
import numpy as np

from posenet_lucas_kanade import posenet
from posenet_lucas_kanade.posenet.constants import *
from posenet_lucas_kanade.posenet.utils import *

from posenet_lucas_kanade.utils.params import *

# Load posenet model - can be cpu() or cuda()
model = posenet.load_model(POSENET_MODEL_NUM, OUTPUT_STRIDE)
model = model.cpu()

def main():

    # Output stride - default 16
    output_stride = model.output_stride

    ''' Image folder test '''
    # # Load image in folder for testing:
    # filenames = [f.path for f in os.scandir(INPUT_IMG_TEST_DIR) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    #
    # for f in filenames:
    #     # Convert draw_image (HEIGHT, WIDTH, 3) to input_image (1, 3, ~HEIGHT*SCALE_FACTOR, ~WIDTH*SCALE_FACTOR) - scale [~1/SCALE_FACTOR ~1/SCALE_FACTOR]
    #     input_image, draw_image, output_scale = posenet.read_imgfile(f, scale_factor = SCALE_FACTOR, output_stride = OUTPUT_STRIDE)
    #     main_processing(input_image, draw_image, output_scale)

    ''' Video file test '''
    print(VIDEO_PATH + VIDEO_NAME)
    cap = cv2.VideoCapture(VIDEO_PATH + VIDEO_NAME)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    while cap.isOpened():
        input_image, draw_image, output_scale = posenet.read_cap(cap, scale_factor = SCALE_FACTOR, output_stride = OUTPUT_STRIDE)
        main_processing(input_image, draw_image, output_scale)
    video.release()

def heatmap_inspection(heatmaps_result, draw_image, scale_factor, output_stride):
    # print('----- heatmap_inspection -----')
    # Get heatmap above threshole
    np_heatmap = heatmaps_result[0].cpu().numpy()
    below_threshold_indices = np_heatmap < THRESHOLD
    np_heatmap[below_threshold_indices] = 0

    ''' Face mask '''
    np_heatmap_face_mask = np.zeros(np_heatmap[0].shape, np.float)
    np_heatmap_face_mask = np_heatmap[0] + np_heatmap[1] + np_heatmap[2] + np_heatmap[3] + np_heatmap[4]

    ''' All mask'''
    np_heatmap_all_mask = np_heatmap_face_mask
    for i in range(4, len(np_heatmap)):
        np_heatmap_all_mask += np_heatmap[i]

    # Equal mask
    ispeople_indices = np_heatmap_all_mask > 0
    np_heatmap_all_mask[ispeople_indices] = 255
    # Not equal mask
    # np_heatmap_all_mask = np_heatmap_all_mask/np.max(np_heatmap_all_mask)*255
    np_heatmap_all_mask = np_heatmap_all_mask.astype(np.uint8)
    np_heatmap_all_mask = cv2.resize(np_heatmap_all_mask, (draw_image.shape[1], draw_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    np_heatmap_all_mask = cv2.cvtColor(np_heatmap_all_mask, cv2.COLOR_GRAY2BGR)
    show_image('Overlay picture', cv2.addWeighted(draw_image, 0.3, np_heatmap_all_mask, 0.5, 0))

def show_image(name, img):
    cv2.imshow(name, img)
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main_processing(input_image, draw_image, output_scale):

    # Pytorch in reduce memory mode
    with torch.no_grad():
        # Convert to torch data type torch.Size([1, 3, ~HEIGHT*SCALE_FACTOR, ~WIDTH*SCALE_FACTOR])
        input_image = torch.Tensor(input_image).cpu()

        start_time = time.time()
        # Posenet compute - return heatmap torch.Size([1, 17, int(HEIGHT*SCALE_FACTOR/OUTPUT_STRIDE + 1), int(WIDTH*SCALE_FACTOR/OUTPUT_STRIDE + 1)])
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
        stop_time = time.time()
        print('Compute time: ' + str((stop_time - start_time)*1000))

        # Heatmap result inspection
        heatmap_inspection(heatmaps_result, draw_image, SCALE_FACTOR, OUTPUT_STRIDE)

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    main()
