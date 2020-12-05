from dataset.WSVD import WSVD_Img
from dataset.StereoBlur import *
from dataset.MannequinChallenge import *
from models.flow_utils import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from util.visflow import flow_to_png_middlebury

# import OpenEXR
#
# path = "/home/lmaag/xgpu-scratch/mali_data/StereoBlur/HD720-01-17-02-47/disparity_left/0001.exr"
# file = OpenEXR.InputFile(path)


np.random.seed(666)
torch.manual_seed(666)
dataset = StereoBlur_Seq(True, mode='resize', max_skip=30)
dataset.augmenter.outwid, dataset.augmenter.outhei = 1280, 720
flowestim = FlowEstimator(False, "sintel")
# for data in dataset:
if True:
    data = dataset[15]
    refims, tarims, disps, uncertains, isleft = data
    _, _, imhei, imwid = refims.shape
    flow = flowestim(refims[0:1], refims[1:2])
    roux, rouy = 10, 10
    flownp = flow[0, :, ::roux, ::rouy].permute(1, 2, 0).cpu().numpy()
    hei, wid, _ = flownp.shape
    pt1_x, pt1_y = np.meshgrid(np.linspace(0, roux * (wid - 1), wid), np.linspace(0, rouy * (hei - 1), hei))
    pt1 = np.stack([pt1_x, pt1_y], axis=-1)
    pt2 = pt1 + flownp
    h_mat = cv2.findHomography(pt1.reshape(-1, 2), pt2.reshape(-1, 2), method=cv2.RANSAC, ransacReprojThreshold=3)
    # f_mat = cv2.findFundamentalMat(pt1.reshape(-1, 2), pt2.reshape(-1, 2), method=cv2.FM_RANSAC)
    # cv2.computeCorrespondEpilines()
    print(np.linalg.eig(h_mat[0]))
    plt.imshow(h_mat[1].reshape(72, -1))
    plt.figure()
    plt.imshow(refims[0].permute(1, 2, 0).cpu())
    plt.figure()
    plt.imshow(refims[1].permute(1, 2, 0).cpu())
    plt.figure()
    plt.imshow(flow[0, 0].cpu())
    plt.figure()
    plt.imshow(flow[0, 1].cpu())
    plt.show()
