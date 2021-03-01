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
    "unique_id": "ablation01_svtemp",

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
    "model_name": "MPINetv2",
    "modelloss_name": "svjoint",
    "batch_size": 1,
    "num_epoch": 500,
    "savepth_iter_freq": 441 * 2,
    "lr": 2e-4,
    "lr_milestones": [12e3, 24e3, 36e3],
    "lr_values": [0.5, 0.25, 0.125],
    "check_point": {
        "": "ablation01_svtemp_r0.pth"
    },
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "smooth_loss": 0.5,
        "depth_loss": 0.5,  # need to figure out

        "scale_mode": "adaptive",

        "temporal_loss": 0.9,
    },
}


def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = "AB01_svtemp"
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

    seed = 8353  # np.random.randint(0, 10000)
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
