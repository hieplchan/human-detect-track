import cv2
import numpy as np
import torch

class Posenet:
    ''' Posenet person detector worker '''

    def __init__(self, width, height, scale_factor = 0.5,
                        output_stride = 16, threshold = 0.4):
        self.cam_width = width
        self.cam_height = height
        self.scale_factor = scale_factor
        self.output_stride = output_stride
        self.threshold = threshold
        self.device = torch.device('cpu')
        self.frame_num = 0

        # Calculate compatible size for mobilenetv1
        self.target_width = (int(self.cam_width * self.scale_factor) // self.output_stride) * self.output_stride + 1
        self.target_height = (int(self.cam_height * self.scale_factor) // self.output_stride) * self.output_stride + 1
        self.scale = np.array([self.cam_height/self.target_height, self.cam_width/self.target_width])

    def load_model(model_path):
        model = MobileNetV1(output_stride = output_stride)
        model.load_state_dict(torch.load(model_path, map_location = device))
        model.to(device)
        return model
        pass

    def reload_model(model_path):
        ''' Reload model if params change, dispose old model memory '''
        pass

    def getFrameResult(draw_image):
        ''' Process one frame and return detected points, boxs '''
        pass
