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
    "validate_num": 32,
    "valid_freq": 700,
    "train_report_freq": 5,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "id": "",
    "comment": "",

    "trainset": "m+r+s_seq",
    "evalset": "m+r+s_seq",
    "model_name": "V5Nset2",
    "modelloss_name": "fulljoint",
    "batch_size": 1,
    "num_epoch": 2000,
    "savepth_iter_freq": 400,
    "lr": 5e-5,
    "check_point": {
        "": "V52setcnn_121011_r0.pth"
    },
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "net_smth_loss": 1,
        "depth_loss": 1,

        "scale_mode": "fix",
        # "scale_scaling": 1,

        "upmask_magaware": True,
        "mask_warmup": 1,
        "mask_warmup_milestone": [1e18, 2e18],
        # "bgflow_warmup": 1,
        # "bgflow_warmup_milestone": [2e3, 4e3],
        # "net_warmup": 0,
        # "net_warmup_milestone": [1e18, 2e18],
        # "aflow_fusefgpct": False,

        "tempnewview_mode": "biflow",
        "tempnewview_loss": 0,
    },
}


# TODO
#   \current problem:
#   >>> still want to handle the temporal consistency issue
#   \evalutaion
#   >>> evaluator for MyDatasetClass
#   >>> decide what dataset to use for depth
#   >>> implement various video depth methods

def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = "V52setcnn_sfixs"
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

    seed = 6558  # np.random.randint(0, 10000)
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