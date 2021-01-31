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
    "validate_num": 32,
    "valid_freq": 500,
    "train_report_freq": 5,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "id": "",
    "comment": "",

    "trainset": "realestate10k_img",
    "evalset": "realestate10k_seq",
    "model_name": "MPINetv2",
    "modelloss_name": "sv",
    "batch_size": 2,
    "num_epoch": 1000,
    "savepth_iter_freq": 300,
    "lr": 3e-5,
    "check_point": "no.pth",
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "smooth_loss": 0.5,
        "depth_loss": 0.2,  # need to figure out
        
        # "temporal_loss": 0.9,
        # "pixel_std_loss": 0.5,
        # "temporal_loss": 0.5,
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


def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = "raSV_pretrain_Ronlyrmsmth"
    cfg["comment"] = "single frame method baseline (fine-tuning on my dataset)"

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

    seed = 2276  # np.random.randint(0, 10000)
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
