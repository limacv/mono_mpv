import os
import sys
sys.path.append("..")
import numpy as np
import cv2
import torch
from models.mpi_network import *
from models.mpi_utils import *
from models.loss_utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset.RealEstate10K import RealEstate10K_Img, RealEstate10K_root

model = MPINet(32)
model.load_state_dict(torch.load("../log/MPINet/mpinet_ori.pth")["state_dict"])
model.cuda()

save_prefix = "/scratch/PI/psander/mali_data/RealEstate10K/visualize"


def main(istrain=True):
    trainset = RealEstate10K_Img(istrain, subset_byfile=True)
    save_path = os.path.join(save_prefix, trainset.trainstr)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Checking trainset (len {len(trainset)})...", flush=True)
    for i in range(len(trainset)):
        datas = trainset[i]
        datas = [_t.unsqueeze(0).cuda() for _t in datas]
        refim, tarim, refextrin, tarextrin, intrin, pt2ds, ptzs_gt = datas
        with torch.no_grad():
            netout = model(refim.cuda())
            mpi, blend_weight = netout2mpi(netout, refim, ret_blendw=True)
            depth = make_depths(32).type_as(mpi).unsqueeze(0)
            disparity = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)
            ptdis_e = torchf.grid_sample(disparity.unsqueeze(1), pt2ds.unsqueeze(1)).squeeze(1).squeeze(1)
            scale = torch.exp(torch.log(ptdis_e * ptzs_gt).mean(dim=-1, keepdim=True))
            ptdis_e /= scale

        diff = torch.log(ptdis_e * ptzs_gt)
        diffmean = torch.pow(diff, 2).mean()
        print(f"diff = {diffmean}")

        depth_hist = np.histogram((1 / ptzs_gt).cpu().numpy(), 100)
        diff_hist = np.histogram(diff.cpu().numpy(), 100)
        plt.plot(depth_hist[1][1:], depth_hist[0])
        plt.savefig(os.path.join(save_path, f"{trainset._cur_file_base}_zhisto.jpg"))
        plt.clf()
        plt.plot(diff_hist[1][1:], diff_hist[0], 'r')
        plt.savefig(os.path.join(save_path, f"{trainset._cur_file_base}_zhistodiff.jpg"))
        plt.clf()

        sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt)
        sparsediff = draw_sparse_depth(refim, pt2ds, torch.clamp(diff + 0.5, 0, 1), "JET")
        sparsedepth = sparsedepth[:, :, ::-1]
        sparsediff = sparsediff[:, :, ::-1]
        cv2.imwrite(os.path.join(save_path, f"{trainset._cur_file_base}_depth.jpg"), sparsedepth)
        cv2.imwrite(os.path.join(save_path, f"{trainset._cur_file_base}_diff.jpg"), sparsediff)



main(True)
main(False)
