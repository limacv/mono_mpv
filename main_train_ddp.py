import os
import sys
from datetime import datetime
import time
import torch
import numpy as np
import torch.distributed
import torch.backends.cudnn
import trainer_distributed as trainer
import argparse
import random

cfg = {
    "local_rank": 0,  # will set later
    "world_size": 10,
    # const configuration <<<<<<<<<<<<<<<<
    "log_prefix": "./log/",
    "tensorboard_logdir": "run/",
    "mpi_outdir": "mpi/",
    "checkpoint_dir": "checkpoint/",

    "write_validate_result": True,
    "validate_num": -1,
    "valid_freq": 200,
    "train_report_freq": 5,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "id": "",
    "comment": "",

    "trainset": "stereovideo_seq",
    "evalset": "stereovideo_seq",
    "model_name": "Fullv30",
    "modelloss_name": "fullv2",
    "batch_size": 1,
    "num_epoch": 5000,  # actually it's num_iter
    "savepth_iter_freq": 500,
    "lr": 1e-4,
    "check_point": {
        "": "no.pth"
    },
    "loss_weights": {
        "pixel_loss_cfg": 'vgg',
        "pixel_loss": 0.2,
        "smooth_loss": 0.05,
        "smooth_flowgrad_loss": 0.05,
        "depth_loss": 5,
        # "pixel_std_loss": 0.5,
        # "temporal_loss": 0.5,
        "tempdepth_loss": 1,
        "temporal_loss_mode": "mse"
        # "splat_mode": "bilinear",
        # "dilate_mpfin": True,
        # "alpha2mpf": True,

        # "flow_epe": 1,
        # "flow_smth": 0.1,
        # "flow_smth_ord": 1,
        # "flow_smth_bw": False
        # "aflow_includeself": True,
        # "sflow_loss": 0.1

        # "sparse_loss": 0.1,
        # "smooth_tar_loss": 0.5,
    },
}


# TODO
#   \problems
#   >>> temporal consistency when training in large dataset
#   >>> inpainting performance (pending to be test)
#   \code
#   >>> temporal consistency in lower resolution
#   >>> smooth loss, temporal loss regarding the alpha distribution parameters
#   >>>
#   >>> ready for the evaluating metric!!!!!

def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = "v30_down_dts"
    cfg["comment"] = "full model of v2 pipeline trained on stereovideo dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    cfg["local_rank"] = args.local_rank

    # the settings for debug
    # please comment this
    if "LOGNAME" in os.environ.keys() and os.environ["LOGNAME"] == 'jrchan':
        print("Debug Mode!!!", flush=True)
        cfg["comment"] = "Dont't forget to change comment" * 100
        cfg["world_size"] = 2
        cfg["train_report_freq"] = 1
    else:
        import warnings
        warnings.filterwarnings("ignore")

    print("Cuda available devices:")
    devices_num = torch.cuda.device_count()
    for device in range(devices_num):
        print(f"{device}: {torch.cuda.get_device_name(device)}")
    print(f"------------- start running (PID: {os.getpid()} Rank: {cfg['local_rank']})--------------", flush=True)
    torch.cuda.set_device(cfg["local_rank"])

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(0)
    random.seed(0)
    torch.distributed.init_process_group('nccl', world_size=cfg["world_size"], init_method='env://')

    trainer.train(cfg)


if __name__ == "__main__":
    main(cfg)
