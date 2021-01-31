import os
import sys
sys.path.append("..")
import numpy as np
import cv2
import torch
from models.mpi_network import *
from models.mpi_utils import *
from models.loss_utils import *
from torch.utils.data import DataLoader

from dataset.RealEstate10K import RealEstate10K_Img
from dataset.MannequinChallenge import MannequinChallenge_Img


save_prefix = "/d1/scratch/PI/psander/mali_data/MannequinChallenge/visualize"


def main(istrain=True):
    # trainset = RealEstate10K_Img(istrain,  black_list=True)
    trainset = MannequinChallenge_Img(istrain,  black_list=True)
    save_path = os.path.join(save_prefix, trainset.trainstr)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Checking trainset (len {len(trainset)})...", flush=True)

    trainloader = DataLoader(trainset, 1, True)
    for i, datas in enumerate(trainloader):
        refim, tarim, refextrin, tarextrin, intrin, pt2ds, ptzs_gt = datas
        disp_gt = 1 / ptzs_gt
        disp_gt, sortidx = torch.sort(disp_gt)
        pt2ds = pt2ds[:, sortidx, :]
        disp_gt = disp_gt[:, 20:-20]
        pt2ds = pt2ds[:, 0, 20:-20]
        disp_gt = (disp_gt - disp_gt[:, 0]) / (disp_gt[:, -1] - disp_gt[:, 0])
        sparsedepth = draw_sparse_depth(refim, pt2ds, disp_gt)
        sparsedepth = sparsedepth[:, :, ::-1]
        cv2.imwrite(os.path.join(save_path, f"{trainset._cur_file_base}_depth.jpg"), sparsedepth)
        print(i, flush=True)
        if i > 1000:
            break


main(True)
