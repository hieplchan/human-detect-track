import torch
import os

from .mobilenet_v1 import MobileNetV1
from posenet_lucas_kanade import POSENET_MODEL_DIR

def load_model(model_id, output_stride, model_dir=POSENET_MODEL_DIR):
    model_path = os.path.join(model_dir, 'mobilenet_v1_050.pth')

    model = MobileNetV1(output_stride=output_stride)
    load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    return model
