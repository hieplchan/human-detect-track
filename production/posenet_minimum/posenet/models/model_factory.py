import os
import torch

from .mobilenet_v1 import MobileNetV1
from posenet_minimum import POSENET_MODEL_DIR
from posenet_minimum.utils.params import device

def load_model(output_stride):
    model_path = os.path.join(POSENET_MODEL_DIR, 'mobilenet_v1_050_cpu.pth')
    model = MobileNetV1(output_stride=output_stride)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print('Is CUDA: ' + str(next(model.parameters()).is_cuda))

    return model
