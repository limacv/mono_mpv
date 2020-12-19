import os
import sys
sys.path.append("..")
import numpy as np
import cv2
import torch
from models.mpi_network import *
from models.mpi_utils import *
from models.loss_utils import *
import matplotlib.pyplot as plt

from dataset.MannequinChallenge import MannequinChallenge_Img, MannequinChallenge_root

save_prefix = os.path.join(MannequinChallenge_root, "visualize")


def main(istrain=True):
    trainset = MannequinChallenge_Img(istrain, black_list=True)
    save_path = os.path.join(save_prefix, trainset.trainstr)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Checking trainset (len {len(trainset)})...", flush=True)
    for i in range(len(trainset)):
        datas = trainset[i]
        datas = [_t.unsqueeze(0).cuda() for _t in datas]
        refim, tarim, refextrin, tarextrin, intrin, pt2ds, ptzs_gt = datas

        depth = 1 / ptzs_gt
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        sparsedepth = draw_sparse_depth(refim, pt2ds, depth_norm)
        sparsedepth = sparsedepth[:, :, ::-1]
        cv2.imwrite(os.path.join(save_path, f"{trainset._cur_file_base}_depth.jpg"), sparsedepth)


main(True)
main(False)
