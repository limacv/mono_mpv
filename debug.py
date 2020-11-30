from dataset.StereoBlur import *
from models.ModelWithLoss import *
from models.mpi_network import *
from models.hourglass import *
from models.mpi_utils import *
from models.loss_utils import *
from models.mpifuse_network import *
import torch
from torch.nn.parallel import DataParallel
import numpy as np
from tensorboardX import SummaryWriter
import multiprocessing as mp
from trainer import select_module, select_modelloss

np.random.seed(666)
torch.manual_seed(666)


def main(kwargs):
    device_ids = kwargs["device_ids"]
    batchsz = kwargs["batchsz"]
    model = select_module("Full")
    if "checkpoint" in kwargs:
        model.load_state_dict(torch.load(kwargs["checkpoint"])["state_dict"])
    else:
        initial_weights(model)
    model.cuda()
    modelloss = select_modelloss("full")(model, kwargs)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    dataset = StereoBlur_Seq(True, mode='crop', seq_len=2)
    modelloss = DataParallel(modelloss, device_ids)
    for i in range(int(14000)):
        datas_all = [[]] * 6
        for dev in range(2):
            datas = dataset[0]
            datas_all = [ds_ + [d_] for ds_, d_ in zip(datas_all, datas)]

        datas = [torch.stack(data, dim=0).cuda() for data in datas_all]
        _val_dict = modelloss.module.valid_forward(*datas, visualize=True)
        loss_dict = modelloss(*datas, step=i)
        loss = loss_dict["loss"]
        loss = loss.mean()
        loss_dict = loss_dict["loss_dict"]
        loss_dict = {k: v.mean() for k, v in loss_dict.items()}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # output iter infomation
        loss_str = " | ".join([f"{k}:{v:.3f}" for k, v in loss_dict.items()])
        print(f"loss:{loss:.3f} | {loss_str}", flush=True)
        if i % 25 == 0:
            datas = [d_[:batchsz] for d_ in datas]
            _val_dict = modelloss.module.valid_forward(*datas, visualize=True)

        if i % 100 == 0 and "savefile" in kwargs.keys():
            torch.save(model.state_dict(), kwargs["savefile"])


# complete cfg:
# cuda_device=0
# checkpoint="./log/MPINet/mpinet_ori.pth", [opt]
# logdir="./log/run/dbg_/", [opt]
# mpioutdir="./log/Debug1", [opt]
# savefile="./log/Debug_all.pth", [opt]
# loss_cfg={"pixel_loss": 1,
#        "smooth_loss": 0.5,
#        "depth_loss": 0.1},


main({
    "device_ids": [0],
    # "device_ids": [0, 1, 2, 3, 4, 5, 6, 7],
    # "checkpoint": "./log/MPINet/mpinet_ori.pth",
    "batchsz": 1,
    # "checkpoint": "./log/MPINet/mpinet_ori.pth",
    # "savefile": "./log/DBG_pretrain.pth",
    "logdir": "./log/run/debug_svscratch",
    "savefile": "./log/checkpoint/debug_svscratch.pth",
    "loss_weights": {"pixel_loss": 1,
                     "smooth_loss": 0.5,
                     "depth_loss": 0.1,
                     "templ1_loss": 1,
                     "tempdepth_loss": 0.01},
})
# good data list
# 01bfb80e5b8fe757
# 0b49ce385d8cdadc
# 1ab9594c7015ecdc
# 1d6248d4db20fdc3
# 02ecb10e9363e210
# 2a7c6acafe4422b4
# 2aabcf66c52b7343
# 2aacd794386dbdc3
# 2af41ef10045d1c3
# 2bfe6a3923cf9a83
# 2cf0df5ede50bd1d
# 3b22f4f99f2bf2f5
