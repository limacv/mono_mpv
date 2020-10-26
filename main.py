import os
import sys
from datetime import datetime
import time
import torch
import torch.nn as nn
import numpy as np

from dataset import RealEstate10K
from models.ModelWithLoss import ModelandSuLoss, ModelandUnsuLoss
from util.config import Experiments
import trainer
import multiprocessing as mp
# from util.config import fakeMultiProcessing as mp
# fakeDeviceNum = 100

cfg = {
    # const configuration <<<<<<<<<<<<<<<<
    "cuda_device": 0,
    "log_prefix": "./log/",
    "stdout_to_file": True,
    "run_dir": "run/",
    # "display_validate_result": False,
    "write_validate_result": True,
    "validate_num": 128,  # the number of pairs that perform validate -1 means all
    "valid_freq": 200,
    "train_report_freq": 10,

    # dataset <<<<<<<<<<<<<<<<<<
    "all_video": False,
    # "dataset_path": "/scratch/PI/psander/mali_data/Sintel/training/final/alley_2",
    "dataset_path": "/scratch/PI/psander/mali_data/DAVIS/480p/gold-fish",

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "comment": "<Please add comment in experiments>",
    "model_name": "RAFTNet",
    "batch_size": 1,
    "num_epoch": 100,
    "save_epoch": [5 * i for i in range(1, 30)],
    "sample_num_per_epoch": 2000,
    "lr": 2e-6,
    "weight_decay": 0,
    "check_point": 0,
    "occ_cfg": {
        "estimate_occ": True,  # if not, occ are set to all zero
        "soft_occ_map": False,  # when hard occ_map, the grad will stop naturally
        "stop_grad_at_occ": True,
    },
    "loss_weights": {
        "census_loss": 1,  # 2.5,
        # "photo_loss": 5.,
        # "ssim_loss": 5.,
        "smooth_loss": 1,  # 2.5,
        # "flow_loss": 0.2,
        # "occ_loss": 1,

        "level_weight": [1., 0.5, 0.25, 0.125],  # correspond to flow level
        "smooth_to_level": 3,  # 0 means only to finest level
        "flow_to_level": 3,  # 0 means only to finest level
        "occ_down_level": 3,  # 0 means at original resolution

        # "hsv_at_finest": False,  # input image as hsv format at finest level default False
        "smooth_normalize": 'x_dx',
        "smooth_order": 1,
    },

    # <<<<<<<<<<<<<<<<<<<<pretraining configure<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    "pretrain": False,
    "pretrain_cfg": {  # this will overwrite configure
        "num_epoch": 100,
        "save_epoch": [10, 15, 20, 30, 35, 40, 45],
        "lr": 1e-5,
        "weight_decay": 1e-7,
        "batch_size": 4,
        "dataset_path": "/scratch/PI/psander/mali_data/FlyingChairs2/",
        "loss_weights": {
            "su_epe_loss": 1.,
            # "level_weight": [0.4, 0.3, 0.1, 0.1, 0.1, 0]
            "level_weight": [0.6, 0.4, 0.0, 0.0, 0.0, 0]
        }
    }
}


# TODO:
#   1. repo single view mpi
def main(cfg):

    # the settings for debug
    if 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "MSI":
        cfg["stdout_to_file"] = False
        # cfg["display_validate_result"] = True
        # cfg["write_validate_result"] = False
        cfg["train_report_freq"] = 1
        cfg["batch_size"] = 1
        # cfg["dataset_path"] = "D:\\MSI_NB\\source\\dataset\\Sintel\\training\\final\\alley_2"
        cfg["dataset_path"] = "D:\\MSI_NB\\source\\dataset\\DAVIS-2017-Unsupervised-trainval-480p\\" \
                              "DAVIS\\JPEGImages\\480p\\gold-fish"

        cfg["pretrain_cfg"]["dataset_path"] = "D:\\MSI_NB\\source\\dataset\\FlyingChairs2"
        cfg["pretrain_cfg"]["batch_size"] = 1

    # preprocess the config
    # --------------------------
    if cfg["stdout_to_file"]:
        sys.stdout = open(f"./stdout_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w")
    cfg["model_name"] += "_pretrain" if cfg["pretrain"] else ""
    cfg["check_point"] = cfg["log_prefix"] + cfg["model_name"] + "/" + str(cfg["check_point"]) + ".ckpt"
    torch.manual_seed(0)
    torch.cuda.set_device(cfg["cuda_device"])
    np.random.seed(0)

    print(f"------------- start running (PID: {os.getpid()})--------------", flush=True)
    if cfg["pretrain"]:
        print("Start pretraining!")
        cfg.update(cfg["pretrain_cfg"])
        dataset = FlyingChairs(cfg["dataset_path"])
        model = select_module(cfg["model_name"])
        modelloss = ModelandSuLoss(model, cfg["loss_weights"].copy())
    else:
        if cfg["all_video"]:
            dataset = get_sintel_all_seq(os.path.dirname(cfg["dataset_path"]))
        elif "Sintel" in cfg["dataset_path"]:
            dataset = SintelSeq(cfg["dataset_path"])
        elif "DAVIS" in cfg["dataset_path"]:
            dataset = DAVISSeq(cfg["dataset_path"])
        model = select_module(cfg["model_name"])
        modelloss = ModelandUnsuLoss(model, cfg["loss_weights"].copy(), cfg["occ_cfg"].copy())
    trainer.train(modelloss, dataset, cfg)


def select_module(name: str) -> nn.Module:
    if "IRRPWCNet" in name:
        pass
    else:
        raise ValueError("unrecognized modelin name: {}".format(name))

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
    experiments.add_experiment({"comment": "repo last week, try on alley_1",
                                "occ_cfg": {"estimate_occ": False}})

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
        print(f"\n>>>>>>>>Start Process {idx+1}, with following changes:")
        print(experiments.get_info_str(), flush=True)
        process.start()
        processes.append(process)

        time.sleep(waittime)  # postpond 61 second to prevent same stdout file

    print(f"=======Successfully start all {len(processes)} processes!===========", flush=True)
    for process in processes:
        process.join()
