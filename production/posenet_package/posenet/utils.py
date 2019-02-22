import cv2
import torch
import numpy as np

def process_input(source_img, targer_width, target_height, device):
    source_img = cv2.normalize(source_img, None, 0, 255, cv2.NORM_MINMAX)
    input_img = cv2.resize(source_img, (targer_width, target_height), interpolation = cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, targer_width)
    input_img = torch.as_tensor(input_img, device = device)
    return input_img, source_img


def read_cap(cap, targer_width, target_height, device):
    res, img = cap.read()
    if not res:
        raise IOError("Video Error")
    return process_input(img, targer_width, target_height, device)

def read_imgfile(path, targer_width, target_height, device):
    img = cv2.imread(path)
    return process_input(img, targer_width, target_height, device)

def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(1)
    # while(True):
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
