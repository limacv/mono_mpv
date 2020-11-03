import os
import sys
import numpy as np
import time
import datetime
import shutil
import cv2

sys.path.append("..")
from dataset.RealEstate10K import RealEstate10K, RealEstate10K_root
from models.loss_utils import draw_sparse_depth


def main(istrain=True):
    trainset = RealEstate10K(istrain, subset_byfile=True)

    print(f"Checking trainset (len {len(trainset)})...", flush=True)
    for i in range(len(trainset)):
        datas = trainset[i]
        refim, tarim, refextrin, tarextrin, intrin, pt2ds, ptzs_gt = datas
        sparsedepth = draw_sparse_depth(refim.unsqueeze(0), pt2ds.unsqueeze(0), 1 / ptzs_gt.unsqueeze(0))
        sparsedepth = sparsedepth[:, :, ::-1]
        cv2.imshow("vis", sparsedepth)
        key = cv2.waitKey(10)
        if key > 0:
            print(f"{i}-th data toggled, path {trainset._curvideo_trim_path}")

main(True)
main(False)
