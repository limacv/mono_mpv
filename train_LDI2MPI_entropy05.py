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

unique_id = "LDI2MPI_cel_tmp"

cfg = {
    "local_rank": 0,  # will set later
    "world_size": 10,
    # const configuration <<<<<<<<<<<<<<<<
    "log_prefix": "./log/",
    "tensorboard_logdir": "ldi2mpi_log/",
    "checkpoint_dir": "ldi2mpi_checkpoint/",
    "unique_id": unique_id,

    "write_validate_result": True,
    "validate_num": 64,
    "valid_freq": 1000,
    "train_report_freq": 5,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "id": "",
    "comment": "",

    "trainset": "m+r+s_seq",
    "evalset": "stereovideo_seq",
    "model_name": "RGBAD",
    "modelloss_name": "fulljoint",
    "batch_size": 1,
    "num_epoch": 120,
    "savepth_iter_freq": 200,
    "lr": 2e-4,
    "lr_milestones": [12e3, 24e3, 36e3],
    "lr_values": [0.5, 0.5, 0.5],
    "check_point": {
        "": "LDI2MPI_newCEL5_r0.pth",
        # "": "Ultly3_r0.pth"
    },
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        # "net_smth_loss": 0.2,
        "depth_loss": 0.5,
        "disp_smth_loss": 0.1,
        "flownet_dropout": 1,

        "alpha_entropy": 0.05,
        "alpha_temp": 1,

        "bg_supervision": 1,
    },
}


def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = unique_id
    cfg["comment"] = "old cross entropy loss, fix a small bug"

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

    seed = 147  # np.random.randint(0, 10000)
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
