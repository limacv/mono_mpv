from dataset.RealEstate10K import *
from models.ModelWithLoss import ModelandSVLoss
from models.mpi_network import MPINet
from models.mpi_utils import *
from models.loss_utils import *
import torch
import numpy as np
from tensorboardX import SummaryWriter
import multiprocessing as mp

np.random.seed(5)
torch.manual_seed(0)


def main(kwargs):
    torch.cuda.set_device(kwargs["cuda_device"])
    model = MPINet(32)
    if "checkpoint" in kwargs:
        model.load_state_dict(torch.load(kwargs["checkpoint"])["state_dict"])
    else:
        model.initial_weights()
    model.cuda()
    modelloss = ModelandSVLoss(model, kwargs["loss_cfg"])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)

    if "logdir" in kwargs.keys():
        log_dir = kwargs["logdir"]
        tensorboard = SummaryWriter(log_dir)
    else:
        tensorboard = None

    dataset = RealEstate10K_Img(True, subset_byfile=True)
    for i in range(int(1e7)):
        datas = dataset.getitem_bypath(os.path.join(dataset.root, "01bfb80e5b8fe757.txt"))

        datas = [data.unsqueeze(0).cuda() for data in datas]

        loss, loss_dict = modelloss.train_forward(*datas, step=i)
        for lossname, lossval in loss_dict.items():
            if tensorboard is not None:
                tensorboard.add_scalar(lossname, lossval, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # output iter infomation
        loss_str = " | ".join([f"{k}:{v:.3f}" for k, v in loss_dict.items()])
        print(f"loss:{loss:.3f} | {loss_str}")
        if i % 1000 == 0:
            _val_dict = modelloss.valid_forward(*datas, visualize=True)
            for k in _val_dict.keys():
                if "vis_" in k and tensorboard is not None:
                    tensorboard.add_image(k, _val_dict[k], i, dataformats='HWC')
                elif "save_" in k and tensorboard is not None and "mpioutdir" in kwargs:
                    save_mpi(_val_dict[k], kwargs["mpioutdir"])

        if (i + 1) % 5000 == 0 and "savefile" in kwargs.keys():
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


if __name__ == "__main__":
    mp.set_start_method("spawn")
    processes = []

    cfg1 = {
        "cuda_device": 0,
        "checkpoint": "./log/MPINet/mpinet_ori.pth",
        "logdir": "./log/run/dbg_pretrain/",
        "mpioutdir": "./log/DBG_pretrain",
        "savefile": "./log/DBG_pretrain.pth",
        "loss_cfg": {"pixel_loss": 1,
                     "smooth_loss": 0.5,
                     "depth_loss": 0.1},
    }

    cfg2 = {
        "cuda_device": 1,
        "logdir": "./log/run/dbg_scratch/",
        "mpioutdir": "./log/DBG_scratch",
        "savefile": "./log/DBG_scratch.pth",
        "loss_cfg": {"pixel_loss": 1,
                     "smooth_loss": 0.5,
                     "depth_loss": 0.1}
    }

    cfg3 = {
        "cuda_device": 2,
        "logdir": "./log/run/dbg_scratchssim/",
        "mpioutdir": "./log/DBG_scratchssim",
        "savefile": "./log/DBG_scratchssim.pth",
        "loss_cfg": {"pixel_loss_cfg": 'ssim',
                     "pixel_loss": 1,
                     "smooth_loss": 0.5,
                     "depth_loss": 0.1}
    }
    process = mp.Process(target=main, args=(cfg1,))
    process.start()
    processes.append(process)
    process = mp.Process(target=main, args=(cfg2,))
    process.start()
    processes.append(process)
    process = mp.Process(target=main, args=(cfg3,))
    process.start()
    processes.append(process)

    print("waiting...", flush=True)
    for process in processes:
        process.join()
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
