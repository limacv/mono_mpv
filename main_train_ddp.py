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
    "tensorboard_logdir": "run1/",
    "mpi_outdir": "mpi/",
    "checkpoint_dir": "checkpoint/",

    "write_validate_result": True,
    "validate_num": 64,
    "valid_freq": 500,
    "train_report_freq": 5,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "id": "",
    "comment": "",

    "trainset": "m+r+s_seq",
    "evalset": "m+r+s_seq",
    "model_name": "Fullv5",
    "modelloss_name": "fulljoint",
    "batch_size": 1,
    "num_epoch": 2000,
    "savepth_iter_freq": 400,
    "lr": 5e-5,
    "check_point": {
        "": "no.pth"
    },
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "net_smth_loss_fg": 0.25,
        # "net_smth_loss_bg": 0.5,
        "depth_loss": 1,

        "alpha_thick_in_disparity": False,
        # "tempdepth_loss": 1,
        # "temporal_loss_mode": "msle",
        # "tempdepth_loss_milestone": [5e3, 10e3],

        "mask_warmup": 0.25,
        "mask_warmup_milestone": [1e18, 2e18],
        "bgflow_warmup": 1,
        "bgflow_warmup_milestone": [4e3, 6e3],
        "net_warmup": 0.5,
        "net_warmup_milestone": [1e18, 2e18],
        # "aflow_fusefgpct": False,
    },
}


# TODO
#   \current problem:
#   >>> transparency issue: the transparency tend to be 0.5, which is not good
#       >> find best way for parameterization the alpha planes
#   >>> check whether recurrent necessary / temporal loss weight
#   \evalutaion
#   >>> evaluator for MyDatasetClass
#   >>> decide what dataset to use for depth
#   >>> implement various video depth methods

def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = "V5Ori_aindepth_s105"
    cfg["comment"] = "bg force nontransparency"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    cfg["local_rank"] = args.local_rank

    # the settings for debug
    # please comment this
    if "LOGNAME" in os.environ.keys() and os.environ["LOGNAME"] == 'jrchan':
        print("Debug Mode!!!", flush=True)
        cfg["comment"] = "Dont't forget to change comment" * 50
        cfg["world_size"] = 2
        cfg["train_report_freq"] = 1
        cfg["valid_freq"] = 20
    else:
        import warnings
        warnings.filterwarnings("ignore")

    print("Cuda available devices:")
    devices_num = torch.cuda.device_count()
    for device in range(devices_num):
        print(f"{device}: {torch.cuda.get_device_name(device)}")
    print(f"------------- start running (PID: {os.getpid()} Rank: {cfg['local_rank']})--------------", flush=True)
    torch.cuda.set_device(cfg["local_rank"])

    seed = np.random.randint(0, 10000)
    print(f"RANK_{cfg['local_rank']}: random seed = {seed}")
    cfg["comment"] += f", random seed = {seed}"
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    torch.distributed.init_process_group('nccl', world_size=cfg["world_size"], init_method='env://')

    trainer.train(cfg)


if __name__ == "__main__":
    main(cfg)
