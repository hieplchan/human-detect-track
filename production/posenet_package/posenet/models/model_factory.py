import torch

from posenet.models.mobilenet_v1 import MobileNetV1

def load_model(model_path, output_stride, device):
    model = MobileNetV1(output_stride = output_stride)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.to(device)
    return model
