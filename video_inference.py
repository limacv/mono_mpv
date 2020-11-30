from models.ModelWithLoss import *
from models.mpi_network import MPINet
from models.mpv_network import MPVNet
from models.mpi_utils import *
from models.loss_utils import *
import torch
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import os
from models.mpi_utils import *

mpiout_path = None
outframeidx = None


# state_dict_path = "./log/checkpoint/DBG_scratch.pth"
# state_dict_path = "./log/MPINet/mpinet_ori.pth"
state_dict_path = "./log/checkpoint/mpinet_baseline_291503_r3.pth"
video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-07-16-53-18.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\RealEstate10K\\testtmp\\ccc439d4b28c87b2\\video_Trim.mp4"
# testtmp\\ccc439d4b28c87b2 -> test_set  traintmp\\01bfb80e5b8fe757 -> used in dbg
saveprefix = "test"

out_prefix = "D:\\MSI_NB\\source\\data\\Visual"
videoout_path = os.path.join(out_prefix, saveprefix + "_disparity.mp4")
videoout1_path = os.path.join(out_prefix, saveprefix + "_newview.mp4")
mpiout_path = os.path.join(out_prefix, saveprefix)
# outframeidx = 27  # 6  # 27

model = MPINet(24).cuda()
# model = MPVNet(32).cuda()
state_dict = torch.load(state_dict_path, map_location='cuda:0')
# torch.save({"state_dict": state_dict["state_dict"]}, state_dict_path)
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict)
# modelloss = ModelandSVLoss(model, {"loss_weights":{},"device_ids":[0]})
modelloss = ModelandTimeLoss(model, {"loss_weights":{},"device_ids":[0]})

# ## ### #### ##### ###### ####### ######## ####### ###### ##### #### ### ## #

cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter()
out1 = cv2.VideoWriter()
if outframeidx is not None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, outframeidx)

frameidx = 0
while True:
    ret, img = cap.read()
    if not ret or frameidx > 100:
        break

    hei, wid, _ = img.shape
    if wid > hei * 2:
        img = img[:, :wid//2]
        hei, wid, _ = img.shape
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(img).cuda()
    mpi = modelloss.infer_forward(img_tensor, mode='pad_reflect')

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

    visdisp = draw_dense_disp(disparity, depthes[-1])[:, :, ::-1]
    # disp0 = (disparity[0] * 255 * depthes[-1]).detach().cpu().type(torch.uint8).numpy()
    # visdisp = cv2.applyColorMap(disp0, cv2.COLORMAP_HOT)
    visview = (view * 255).type(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
    visview = cv2.cvtColor(visview, cv2.COLOR_RGB2BGR)
    if frameidx == 0 and outframeidx is not None:
        if mpiout_path is not None:
            save_mpi(mpi, mpiout_path)
        cv2.imwrite(videoout_path + ".jpg", visdisp)
        # cv2.imwrite(videoout1_path + ".jpg", visview)
        break

    if not out.isOpened():
        out.open(videoout_path, 828601953, 30., (wid, hei), True)
        out1.open(videoout1_path, 828601953, 30., (wid, hei), True)
    out.write(visdisp)
    out1.write(visview)

    frameidx += 1

out.release()
