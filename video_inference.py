from models.ModelWithLoss import ModelandLoss
from models.mpi_network import MPINet
from models.mpi_utils import *
import torch
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import os
from models.mpi_utils import *


state_dict_path = "./log/MPINet1104_003211.pth"
# state_dict_path = "./log/MPINet/mpinet_ori.pth"
video_path = "D:\\MSI_NB\\source\\data\\RealEstate10K\\testtmp\\ccc439d4b28c87b2\\video_Trim.mp4"
out_prefix = "D:\\MSI_NB\\source\\data\\Visual"
videoout_path = os.path.join(out_prefix, "disparity.mp4")
videoout1_path = os.path.join(out_prefix, "newview.mp4")
mpiout_path = os.path.join(out_prefix, "mpi")

model = MPINet(32).cuda()
state_dict = torch.load(state_dict_path, map_location='cuda:0')
torch.save({ "state_dict": state_dict["state_dict"]}, state_dict_path)
model.load_state_dict(state_dict["state_dict"])
modelloss = ModelandLoss(model, {})

cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter()
out1 = cv2.VideoWriter()
frameidx = 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    hei, wid, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(img).cuda()
    mpi = modelloss.infer_forward(img_tensor, mode='pad_reflect')

    if frameidx == 0:
        save_mpi(mpi, mpiout_path)
    depthes = make_depths(32).cuda()
    disparity = estimate_disparity_torch(mpi, depthes)

    target_pose = torch.tensor(
        [[1.0, 0.0, 0.0, -0.05],
         [0.0, 1.0, 0.0, 0],
         [0.0, 0.0, 1.0, -0.05]]
    ).type_as(mpi).unsqueeze(0)
    source_pose = torch.tensor(
        [[1.0, 0.0, 0.0, 0],
         [0.0, 1.0, 0.0, 0],
         [0.0, 0.0, 1.0, 0]]
    ).type_as(mpi).unsqueeze(0)
    intrin = torch.tensor(
        [[wid / 2, 0.0, wid / 2],
         [0.0, hei / 2, hei / 2],
         [0.0, 0.0, 1.0]]
    ).type_as(mpi).unsqueeze(0)
    view = render_newview(mpi, source_pose, target_pose, intrin, depthes)

    disp0 = (disparity[0] * 255 * depthes[-1]).detach().cpu().type(torch.uint8).numpy()
    visdisp = cv2.applyColorMap(disp0, cv2.COLORMAP_HOT)
    visview = (view * 255).type(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
    visview = cv2.cvtColor(visview, cv2.COLOR_RGB2BGR)

    if not out.isOpened():
        out.open(videoout_path, 828601953, 30., (wid, hei), True)
        out1.open(videoout1_path, 828601953, 30., (wid, hei), True)
    out.write(visdisp)
    out1.write(visview)

    frameidx += 1

out.release()
