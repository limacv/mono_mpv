from datetime import datetime
import time
import os
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split, RandomSampler
from tensorboardX import SummaryWriter

from models.ModelWithLoss import *
from models.mpi_network import MPINet
from models.mpi_utils import save_mpi
from dataset.RealEstate10K import RealEstate10K_Img


def select_module(name: str) -> nn.Module:
    if "MPINet" in name:
        return MPINet(32)
    else:
        raise ValueError(f"unrecognized modelin name: {name}")


def select_modelloss(name: str) -> object:
    name = name.lower()
    if "single_view" in name or "sv" in name:
        return ModelandSVLoss
    elif "time" in name or "temporal" in name:
        return ModelandTimeLoss
    else:
        raise ValueError(f"unrecognized modelloss name: {name}")


def train(cfg: dict):
    # ------------------------------
    # figuring out configuration
    # -----------------------------
    model = select_module(cfg["model_name"])
    modelloss = ModelandSVLoss(model, cfg["loss_weights"])

    num_epoch = cfg["num_epoch"]
    batch_sz = cfg["batch_size"]
    lr = cfg["lr"]

    save_epoch_freq = cfg["save_epoch_freq"]
    log_prefix = cfg["log_prefix"]
    save_path = os.path.join(log_prefix, cfg["model_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        check_point = torch.load(os.path.join(save_path, cfg["check_point"]))
    except FileNotFoundError:
        print(f"cannot open check point file {cfg['check_point']}")
        check_point = {}

    begin_epoch = 0
    if len(check_point) > 0:
        model.load_state_dict(check_point["state_dict"])
        begin_epoch = check_point["epoch"]
        print(f"load state dict of epoch:{begin_epoch}")
    else:
        model.initial_weights()
        print(f"initial the model weights")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    if "optimizer" in check_point.keys():
        check_point["optimizer"]["param_groups"][0]["lr"] = lr
        optimizer.load_state_dict(check_point["optimizer"])

    evalset = RealEstate10K_Img(is_train=False, subset_byfile=True)
    validate_gtnum = cfg["validate_num"] if 0 < cfg["validate_num"] < len(evalset) else len(evalset)

    datasubset = Subset(evalset, torch.randperm(len(evalset))[:validate_gtnum])
    evaluatedata = DataLoader(datasubset, batch_size=batch_sz, shuffle=False, pin_memory=True, drop_last=True,)
    validate_freq = cfg["valid_freq"]

    trainset = RealEstate10K_Img(is_train=True, subset_byfile=True)
    train_report_freq = cfg["train_report_freq"]
    start_time = time.time()
    trainingdata = DataLoader(trainset, batch_sz, pin_memory=True, drop_last=True,
                              sampler=RandomSampler(trainset, True, cfg["sample_num_per_epoch"]))

    unique_timestr = datetime.now().strftime("%m%d_%H%M%S")  # used to track different runs
    tsb_log_dir = os.path.join(log_prefix, cfg["tensorboard_logdir"])
    log_dir = tsb_log_dir + unique_timestr
    tensorboard = SummaryWriter(log_dir)
    # write configure of the current training
    cfg_out = tsb_log_dir + "configs.txt"
    with open(cfg_out, 'a') as cfg_outfile:
        loss_str = ', '.join([f'{k}:{v}' for k, v in cfg['loss_weights'].items()])
        cfg_str = f"\n-------------------------------------------------\n" \
                  f"|Name: {log_dir}|\n"  \
                  f"-----------------\n" \
                  f"Comment: {cfg['comment']}\n" \
                  f"Dataset: train:{trainset.name}, len={len(trainset)}, eval:{evalset.name}, len={len(evalset)}\n" \
                  f"Model: {cfg['model_name']}\n" \
                  f"Loss: {loss_str}\n" \
                  f"Training: checkpoint={cfg['check_point']}, bsz={batch_sz}, lr={lr}\n" \
                  f"Validation: gtnum={validate_gtnum}, freq={validate_freq}\n" \
                  f"\n"
        cfg_outfile.write(cfg_str)

    # =============================================
    # kick-off training
    # =============================================
    for epoch in range(begin_epoch, num_epoch):
        for iternum, datas in enumerate(trainingdata):
            step = len(trainingdata) * epoch + iternum
            # one epoch
            step_scheduler = 1e10 if "mpinet_ori.pth" in cfg["check_point"] else step
            loss, loss_dict = modelloss.train_forward(*datas, step=step_scheduler)

            # recored iter loss
            tensorboard.add_scalar("final_loss", float(loss), step)
            for lossname, lossval in loss_dict.items():
                tensorboard.add_scalar(lossname, lossval, step)

            # output loss message
            if iternum % train_report_freq == 0:
                time_per_iter = (time.time() - start_time) / train_report_freq
                # output iter infomation
                loss_str = " | ".join([f"{k}:{v:.3f}" for k, v in loss_dict.items()])
                curlr = optimizer.state_dict()["param_groups"][0]["lr"]
                print(f"    iter {iternum}/{len(trainingdata)}::loss:{loss:.3f} | {loss_str} "
                      f"| lr:{curlr} | time:{time_per_iter:.3f}s", flush=True)
                start_time = time.time()

            # perform evaluation
            if iternum % validate_freq == 0:
                print("  eval...", end="")

                val_dict, val_display = {}, None
                for i, evaldatas in enumerate(evaluatedata):
                    _val_dict = modelloss.valid_forward(*evaldatas, visualize=(i == 0))

                    if i == 0:
                        for k in _val_dict.copy().keys():
                            if "vis_" in k:
                                tensorboard.add_image(k, _val_dict[k], step, dataformats='HWC')
                                _val_dict.pop(k, None)
                            if "save_" in k:
                                save_mpi(_val_dict.pop(k), f"{save_path}{unique_timestr}")

                    val_dict = {k: val_dict[k] + v if k in val_dict.keys() else v
                                for k, v in _val_dict.items()}

                val_dict = {k: v / len(evaluatedata) for k, v in val_dict.items()}
                for val_name, val_value in val_dict.items():
                    tensorboard.add_scalar(val_name, val_value, step)
                print(" eval:" + ' | '.join([f"{k}: {v:.3f}" for k, v in val_dict.items()]), flush=True)

            # first evaluate and then backward so that first evaluate is always the same
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % save_epoch_freq == 0:
                torch.save({
                    "state_dict": model.cpu().state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "cfg": cfg_str,
                    # "optimizer": optimizer.state_dict()
                }, f"{save_path}{unique_timestr}.pth")
                print(f"checkpoint saved {epoch}{unique_timestr}.pth", flush=True)
                model.cuda()

        # output epoch info
        print(f"epoch {epoch} ================================")

    tensorboard.close()
