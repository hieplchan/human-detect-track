import torch

from posenet.models.mobilenet_v1 import MobileNetV1
from posenet import params

def load_model():
    model = MobileNetV1(output_stride = params.OUTPUT_STRIDE)
    model.load_state_dict(torch.load(params.MODEL_PATH, map_location=params.DEVICE))
    model.to(params.DEVICE)
    return model
