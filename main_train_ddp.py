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
    "unique_id": "FinalLDI",

    "write_validate_result": True,
    "validate_num": 64,
    "valid_freq": 2000,
    "train_report_freq": 20,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "id": "",
    "comment": "",

    "trainset": "m+r+s_seq",
    "evalset": "stereovideo_seq",
    "model_name": "LDI",
    "modelloss_name": "fulljoint",
    "batch_size": 1,
    "num_epoch": 500,
    "savepth_iter_freq": 400,
    "lr": 2e-4,
    "lr_milestones": [12e3, 24e3, 36e3],
    "lr_values": [0.5, 0.25, 0.125],
    "check_point": {
        "": "FinalLDI_r0.pth"
    },
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "net_smth_loss": 0.5,
        "depth_loss": 1,
        "flownet_dropout": 1,

        "bg_supervision": 0.1,

        "scale_mode": "adaptive",
        # "scale_scaling": 1,

        "upmask_magaware": True,
        "mask_warmup": 1,
        "mask_warmup_milestone": [1e18, 2e18],
        # "bgflow_warmup": 1,
        # "bgflow_warmup_milestone": [2e3, 4e3],
        # "aflow_fusefgpct": False,

        # "tempnewview_mode": "biflow",
        # "tempnewview_loss": 0,
    },
}


# TODO
#   * List:
#       implement LDI
#   * Implement:
#       temporal consistency - learning blind temporal consistency - ok
#       temporal consistency - blind + dvp
#       temporal consistency - naive filtering in original resolution
#       novel view synthesis - LDI [+temp]
#       novel view synthesis - MPI [+temp]
#       depth from video - WSVD - ok
#       depth from video - MiDaVS - ok
#       depth from video - MannequinChanllenge (single-frame methods) - ok
#   * When have free time:
#       temporal consistency - blind temporal consistency
#       depth from video - MannequinChanllenge (two-frame methods)
#       novel view synthesis - 3D Ken-Burn
#       novel view synthesis - Synsin
#   * Evaluator:
#       run in realtime


def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = "FinalLDI"
    cfg["comment"] = "use the final stereo_video as test"

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
