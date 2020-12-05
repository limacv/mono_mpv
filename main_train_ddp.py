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
    "world_size": 8,
    # const configuration <<<<<<<<<<<<<<<<
    "log_prefix": "./log/",
    "tensorboard_logdir": "run/",
    "mpi_outdir": "mpi/",
    "checkpoint_dir": "checkpoint/",

    "write_validate_result": True,
    "validate_num": 32,
    "valid_freq": 150,
    "train_report_freq": 10,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "id": "",
    "comment": "",

    "trainset": "StereoBlur_seq",
    "evalset": "StereoBlur_seq",
    "model_name": "MPI_FlowGrad",
    "modelloss_name": "disp_flowgrad",
    "batch_size": 1,
    "num_epoch": 9900,
    "savepth_iter_freq": 300,
    "lr": 1e-4,
    "check_point": "no",
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "smooth_loss": 0.05,
        "smooth_flowgrad_loss": 0.05,
        "depth_loss": 2.5,
        # "pixel_std_loss": 0.5,
        # "temporal_loss": 0.5

        # "flow_epe": 1,
        # "flow_smth": 5,
        # "flow_smth_ord": 1,
        # "flow_smth_bw": False
        # "sparse_loss": 0.1,
        # "smooth_tar_loss": 0.5,
    },
}

# todo Current Problem:
#   2. the mpf
#   1. the temporal still not smooth enough


# to be compare:
#   1. end

# TODO List:
#   >>> refine the flow
#   >>> try recurrent that learn residual
#   >>> evaluate the temporal smoothness
#   >>> [opt] think about mpimodel that decomposite the scene into static background and forground
#   >>> [opt] aggregate more frames
#   >>> [opt] more neat log function


def main(cfg):
    """
    Please specify the id and comment!!!!!!!!!
    """
    cfg["id"] = "flowgradin_flowin"
    cfg["comment"] = "try to align the gradient of disparity and flow"

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
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    torch.distributed.init_process_group('nccl', world_size=cfg["world_size"], init_method='env://')

    trainer.train(cfg)


if __name__ == "__main__":
    main(cfg)
