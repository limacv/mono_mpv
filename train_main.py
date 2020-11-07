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
    # const configuration <<<<<<<<<<<<<<<<
    "cuda_device": 0,
    "log_prefix": "./log/",
    "tensorboard_logdir": "run/",

    "write_validate_result": True,
    "validate_num": 32,
    "valid_freq": 200,
    "train_report_freq": 10,

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "comment": "<Please add comment in experiments>",
    "model_name": "MPINet",
    "batch_size": 2,
    "num_epoch": 1000,
    "save_epoch_freq": 200,
    "sample_num_per_epoch": 2000,
    "lr": 0.00002,
    "check_point": "mpinet_ori.pth",
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "smooth_loss": 0.5,
        "depth_loss": 0.1,
        "sparse_loss": 0,
    },
}


# TODO:
#  >>> test that my render-new-view is same as tensorflow implementation
#  >>> train network only on one video
#  >>> refine dataset
#  >>> figuring out why the scale is odd
#  >>> add mono temporal loss
#  >>> should add other metric
#  >>> try LSTM model
#  >>> smooth term too small
#  >>> the new dataset
#  >>> add black list to dataset


def main(cfg):
    # the settings for debug
    if 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "MSI":
        cfg["train_report_freq"] = 1
        cfg["batch_size"] = 1
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
    experiments.add_experiment({"comment": "original implementation from scratch",
                                "check_point": "no",
                                })
    experiments.add_experiment({"comment": "original implementation from scratch, with sparse loss",
                                "check_point": "no",
                                "loss_weights": {
                                    "sparse_loss": 0.0005,
                                },
                                })
    experiments.add_experiment({"comment": "original implementation from scratch, with sparse loss ssim",
                                "check_point": "no",
                                "loss_weights": {
                                    "pixel_loss_cfg": 'ssim',
                                    "sparse_loss": 0.0005,
                                },
                                })
    """
    experiments.add_experiment({"comment": "newdata, newsz, ssim loss, from pretrained",
                                "loss_weights": {
                                    "sparse_loss": 0,
                                }})
    experiments.add_experiment({"comment": "newdata, newsz, sparse loss, ssim loss, from pretrained",
                                "loss_weights": {
                                    "sparse_loss": 0.0001,
                                }})
    experiments.add_experiment({"comment": "newdata, newsz, sparse loss, ssim loss, from scratch", "check_point": "no"})
    experiments.add_experiments(["loss_weights", "pixel_loss_cfg"], ["ternary", "l1"])
    experiments.add_experiment({"comment": "the depth remove first 5% and last 95%",
                                "loss_weights": {
                                    "smooth_loss": 0.5,
                                    "sparse_loss": 0},
                                })
    experiments.add_experiment({"comment": "the depth remove first 5% and last 95%",
                                "loss_weights": {
                                    "smooth_loss": 0.5,
                                    "sparse_loss": 0.002},
                                })
    experiments.add_experiment({"comment": "the depth remove first 5% and last 95%",
                                "loss_weights": {
                                    "smooth_loss": 1.5,
                                    "sparse_loss": 0},
                                })
    experiments.add_experiment({"comment": "the depth remove first 5% and last 95%, from scratch",
                                "check_point": "no",
                                "loss_weights": {
                                    "pixel_loss_cfg": 'ssim',
                                    "smooth_loss": 1.5,
                                    "sparse_loss": 0.002},
                                })
    experiments.add_experiment({"comment": "experiment run, only depth loss",
                                "loss_weights": {"pixel_loss_cfg": 'l1',
                                                 "pixel_loss": 0,
                                                 "smooth_loss": 0,
                                                 "depth_loss": 0.1,
                                                 "sparse_loss": 0, }
                                })
    experiments.add_experiment({"comment": "experiment run, depth loss and smooth",
                                "loss_weights": {"pixel_loss_cfg": 'l1',
                                                 "pixel_loss": 0,
                                                 "smooth_loss": 0.5,
                                                 "depth_loss": 0.1,
                                                 "sparse_loss": 0, }
                                })
    experiments.add_experiment({"comment": "experiment run, only pixel",
                                "loss_weights": {"pixel_loss_cfg": 'l1',
                                                 "pixel_loss": 1,
                                                 "smooth_loss": 0,
                                                 "depth_loss": 0,
                                                 "sparse_loss": 0, }
                                })
    experiments.add_experiment({"comment": "from scratch, only depth",
                                "check_point": "no",
                                "loss_weights": {"pixel_loss_cfg": 'l1',
                                                 "pixel_loss": 0,
                                                 "smooth_loss": 0,
                                                 "depth_loss": 0.1,
                                                 "sparse_loss": 0, }
                                })
    experiments.add_experiment({"comment": "from scratch, depth + smooth",
                                "check_point": "no",
                                "loss_weights": {"pixel_loss_cfg": 'l1',
                                                 "pixel_loss": 0,
                                                 "smooth_loss": 0.5,
                                                 "depth_loss": 0.1,
                                                 "sparse_loss": 0, }
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
