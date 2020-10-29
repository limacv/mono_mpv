from models.ModelWithLoss import ModelandLoss
from models.mpi_network import MPINet
from models.mpi_utils import *
import torch
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import os


state_dict_path = "./log/MPINet/mpinet_ori.pth"
video_path = "D:\\MSI_NB\\source\\data\\RealEstate10K\\testtmp\\bbb7a5bc03290eed\\video_Trim.mp4"
videoout_path = "D:\\MSI_NB\\source\\data\\Visual\\disparity.mp4"

model = MPINet(32)
model.load_state_dict(torch.load(state_dict_path)["state_dict"])
modelloss = ModelandLoss(model, {})

cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter()

while True:
    ret, img = cap.read()
    if not ret:
        break

    hei, wid, _ = img.shape
    img_tensor = ToTensor()(img)
    mpi = modelloss.infer_forward(img_tensor)
    depthes = make_depths(32)
    disparity = estimate_disparity_torch(mpi, depthes)
    disp0 = (disparity[0] * 255 * depthes[-1]).detach().cpu().type(torch.uint8).numpy()
    visdisp = cv2.applyColorMap(disp0, cv2.COLORMAP_HOT)

    if not out.isOpened():
        out.open(videoout_path, 828601953, 30., (wid, hei), True)
    out.write(visdisp)

out.release()
