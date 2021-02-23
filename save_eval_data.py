import evaluator
import torch
import os
from utils import select_evalset, mkdir_ifnotexist

datasetname = "StereoVideo"  # StereoVideo, NvidiaNovelView
datasetcfg = {
    "resolution": (448, 800),  # (540, 960)
    "max_baseline": 3,

    "seq_len": 7,
    "maxskip": 0
}
save_path = "Z:\\tmp\\EvaluationAndResults\\Datas\\SB_len7_skip0\\"

mkdir_ifnotexist(save_path)
dataset = select_evalset(datasetname, **datasetcfg)
for i, data in enumerate(dataset):
    torch.save(data, os.path.join(save_path, f"{i}.pth"))
