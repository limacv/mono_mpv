import os
import sys
from datetime import datetime
import time
import torch
import numpy as np
import trainer
import warnings

cfg = {
    "cuda_device": 0,  # default device
    "device_ids": [0, 1, 2, 3, 4, 5, 6, 7],  # will be added by experiments, list of GPU used
    "gpu_num": 8,
    # const configuration <<<<<<<<<<<<<<<<
    "log_prefix": "./log/",
    "tensorboard_logdir": "run/",
    "mpi_outdir": "mpi/",
    "checkpoint_dir": "checkpoint/",

    "write_validate_result": True,
    "validate_num": 32,
    "valid_freq": 1000,
    "train_report_freq": 10,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "id": "",
    "comment": "",

    "trainset": "RealEstate10K_seq",
    "evalset": "RealEstate10K_seq",
    "model_name": "MPVNet",
    "modelloss_name": "time",
    "batch_size": 1,
    "num_epoch": 1000,
    "savepth_iter_freq": 2000,
    "sample_num_per_epoch": -1,  # < 0 means randompermute
    "lr": 1e-4,
    "check_point": "no",
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "smooth_loss": 0.5,
        "depth_loss": 0.1,
        "pixel_std_loss": 0.5,
        "temporal_loss": 0.5
        # "sparse_loss": 0.1,
        # "smooth_tar_loss": 0.5,
    },
}

# TODO:
#   >>> data distributedDataParallel
#   >>> validation on video data
#   >>> adjustment to current framework:
#       >> modify the training so that it skip the first frame
#       >> modify the network structure so that it can accept arbitrary input size


def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = "mpvnet_onseq"
    cfg["comment"] = "parallel expirment on video data"

    # the settings for debug
    # please comment this
    # cfg["id"] = "testest"
    # cfg["comment"] = "testest"
    # cfg["train_report_freq"] = 1
    # cfg["batch_size"] = 1
    # cfg["gpu_num"] = 1
    # cfg["device_ids"] = [0]

    print("Cuda available devices:")
    devices_num = torch.cuda.device_count()
    for device in range(devices_num):
        print(f"{device}: {torch.cuda.get_device_name(device)}")
    print(f"------------- start running (PID: {os.getpid()})--------------", flush=True)

    torch.manual_seed(0)
    torch.cuda.set_device(cfg["cuda_device"])
    np.random.seed(0)

    trainer.train(cfg)


if __name__ == "__main__":
    main(cfg)
