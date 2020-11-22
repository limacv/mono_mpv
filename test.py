from dataset.WSVD import WSVD_Img
from dataset.StereoBlur import StereoBlur_Img
import cv2
import numpy as np
# import OpenEXR
#
# path = "/home/lmaag/xgpu-scratch/mali_data/StereoBlur/HD720-01-17-02-47/disparity_left/0001.exr"
# file = OpenEXR.InputFile(path)

dataset = StereoBlur_Img(True)
for data in dataset:
    print(data)
