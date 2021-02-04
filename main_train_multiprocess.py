import os
import sys
from datetime import datetime
import time
import torch
import torch.nn as nn
import numpy as np

from dataset.RealEstate10K import RealEstate10K_Img

from util.config import Experiments
import trainer
import multiprocessing as mp

# from util.config import fakeMultiProcessing as mp
# fakeDeviceNum = 100

cfg = {
    "cuda_device": 0,  # default device
    "device_ids": [0],  # will be added by experiments, list of GPU used
    "gpu_num": 1,
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
    "id": "<Please add id in experiments, by default it's the current time>",
    "comment": "<Please add comment in experiments>",
    "trainset": "RealEstate10K_seq",
    "model_name": "MPVNet",
    "modelloss_name": "time",
    "batch_size": 12,
    "num_epoch": 1000,
    "savepth_iter_freq": 1000,
    "sample_num_per_epoch": -1,  # < 0 means randompermute
    "lr": 1e-3,
    "check_point": "no",
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "smooth_loss": 0.5,
        "depth_loss": 0.1,
        "sparse_loss": 0.1,
        "pixel_std_loss": 0.5,
        "smooth_tar_loss": 0.5,
        "temporal_loss": 0.5
    },
}


def main(cfg):
    # the settings for debug
    if 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "MSI":
        cfg["train_report_freq"] = 1
        cfg["batch_size"] = 1
        cfg["gpu_num"] = 1
        cfg["device_ids"] = [0]
    else:
        sys.stdout = open(f"./stdout_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w")

    torch.manual_seed(0)
    torch.cuda.set_device(cfg["cuda_device"])
    np.random.seed(0)

    print(f"------------- start running (PID: {os.getpid()})--------------", flush=True)
    trainer.train(cfg)


if __name__ == "__main__":
    # Get cuda devices
    print(f"Current dir: {os.getcwd()}")
    devices_num = torch.cuda.device_count()
    print("Cuda available devices:")
    for device in range(devices_num):
        print(f"{device}: {torch.cuda.get_device_name(device)}")
    print("--------------------------------------------", flush=True)

    # ////////////////////////////////////////////////////////////////////////////////////////
    # This is where you can add experiments                                                 //
    experiments = Experiments(cfg, False)  # will add first experiment as default           //
    experiments.add_experiment({"comment": "test run on video data",
                                "id": "video_testrun",
                                "gpu_num": 1})
    """
    experiments.add_experiment({"comment": "good data, original implementation, with sparse loss",
                                "gpu_num": 4,
                                "loss_weights": {"sparse_loss": 0.0001},
                                })
    """
    # End of adding experiments                                                             //
    # ////////////////////////////////////////////////////////////////////////////////////////

    # fake process and device for debug reason, please delete afterwards
    waittime = 61  # postpond 61 second to prevent same stdout file
    if 'fakeDeviceNum' in globals():
        devices_num, waittime = 100, 1

    if len(experiments) > devices_num:
        print(f"Warning: not enough devices, only do first {devices_num} experiments", flush=True)

    mp.set_start_method("spawn")
    processes = []
    for idx, experiment_cfg in enumerate(experiments):
        if idx == devices_num:
            break
        process = mp.Process(target=main, args=(experiment_cfg,))
        print(f"\n>>>>>>>>Start Process {idx + 1}, with following changes:")
        print(experiments.get_info_str(), flush=True)
        process.start()
        processes.append(process)

        time.sleep(waittime)  # postpond 61 second to prevent same stdout file

    print(f"=======Successfully start all {len(processes)} processes!===========", flush=True)
    for process in processes:
        process.join()
