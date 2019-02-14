import torch
import math
import cv2
import numpy as np

from posenet_minimum import posenet
from posenet_minimum.utils.params import *

def main():
    model = posenet.load_model(POSENET_MODEL_NUM, OUTPUT_STRIDE)
    model = model.cuda()

    print('Loading model done')
    
    while(True):
        print()

if __name__ == "__main__":
    print('***** START PROGRAMME *****')
    main()