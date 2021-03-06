import torch
import os
import sys
from torchsummary import summary
from torch.autograd import Variable

from .mobilenet_v1 import MobileNetV1, MOBILENET_V1_CHECKPOINTS
from posenet_lucas_kanade import POSENET_MODEL_DIR

DEBUG_OUTPUT = False

def load_model(model_id, output_stride, model_dir=POSENET_MODEL_DIR):
    model_path = os.path.join(model_dir, MOBILENET_V1_CHECKPOINTS[model_id] + '.pth')
    if not os.path.exists(model_path):
        print('Cannot find models file %s, converting from tfjs...' % model_path)
        from posenet_lucas_kanade.posenet.converter.tfjs2pytorch import convert
        convert(model_id, model_dir, check=False)
        assert os.path.exists(model_path)

    model = MobileNetV1(model_id, output_stride=output_stride)
    load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    # Print all parameter load from google posenet pre-trained
    # for key, value in load_dict.items():
    #     print(key, value.shape)

    # Size of one parameter of MobileNetV1
    # torch.tensor(float) take 72 bytes, float 24 bytes, decimal 80 bytes
    # print(sys.getsizeof(load_dict['heatmap.weight'][0][0][0][0]))
    # print(load_dict['heatmap.weight'][0][0][0][0])
    # a = 5.5
    # print(sys.getsizeof(a))
    # print(type(a))

    # Check if model is using CUDA
    # print(next(model.parameters()).is_cuda)

    # Size of model (scale = 1, stride = 16)
    # summary(model, (3, 1073, 1921))

    return model
