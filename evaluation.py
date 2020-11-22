from main_train_MPV_sp import select_module
import torch
import numpy as np
from dataset.sintel_seq import SintelSeq
from models.ModelWithLoss import ModelandUnsuLoss
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import trange

cfg = {
    "model_name": "RAFTNet",
    "check_point": "0_sintel",

    # Const cfg
    "data_path": "D:/MSI_NB/source/dataset/Sintel/training/final/",
}


def main():
    log_prefix = "./log/" + cfg["model_name"] + '/'
    check_point_file = log_prefix + str(cfg["check_point"]) + '.ckpt'
    log_file = log_prefix + "errors.txt"

    model = select_module(cfg["model_name"])
    model.load_state_dict(torch.load(check_point_file)["state_dict"])
    modelandloss = ModelandUnsuLoss(model, {}, {})

    with open(log_file, 'a') as errfile:
        errfile.write(f"\n\n=============================================\n"
                      f"model_name: {cfg['model_name']}, check_point: {cfg['check_point']}\n"
                      f"Evaluation result: \n")
        errfile.writelines(" name      \t| aepe_nocc\t\t| aepe_all\t\t| occ_iou\n")

    # record
    name_list = []
    aepe_nocc_list = []
    aepe_all_list = []
    occ_iou_i_list = []
    occ_iou_u_list = []
    for seq_root, seq_paths, _ in os.walk(cfg["data_path"]):
        for seq_path in seq_paths:

            dataset = SintelSeq(seq_root + seq_path).eval()
            dataset_aepe = []
            print(f"processing: {seq_path}")
            aepe_all, aepe_nocc, iou_i, iou_u = 0, 0, 0, 0
            for frameidx in trange(len(dataset)):
                frame0, frame1, gtflow, gtocc = dataset[frameidx]
                res_dict, _ = modelandloss.valid_forward(frame0, frame1, gtflow, gtocc, False)

                aepe_all += res_dict["val_EPE_all"]
                aepe_nocc += res_dict["val_EPE_nocc"]
                iou_i += res_dict["val_OCCiou_i"]
                iou_u += res_dict["val_OCCiou_u"]

            # record result and output file
            seq_path = seq_path.ljust(10, ' ')
            aepe_nocc /= len(dataset)
            aepe_all /= len(dataset)
            iou_i /= len(dataset)
            iou_u /= len(dataset)
            name_list.append(seq_path)
            aepe_nocc_list.append(aepe_nocc)
            aepe_all_list.append(aepe_all)
            occ_iou_i_list.append(iou_i)
            occ_iou_u_list.append(iou_u)
            with open(log_file, 'a') as errfile:
                errfile.writelines(f"{seq_path}\t| {aepe_nocc:.5f}\t\t| {aepe_all:.5f}\t\t| {iou_i/iou_u:.5f}\n")
                errfile.flush()

    seq_path = "ALL".ljust(10, ' ')
    aepe_nocc = sum(aepe_nocc_list) / len(aepe_nocc_list)
    aepe_all = sum(aepe_all_list) / len(aepe_all_list)
    iou_i, iou_u = sum(occ_iou_i_list), sum(occ_iou_u_list)

    with open(log_file, 'a') as errfile:
        errfile.writelines(f"{seq_path}\t| {aepe_nocc:.5f}\t\t| {aepe_all:.5f}\t\t| {iou_i / iou_u:.5f}\n")
        errfile.flush()


if __name__ == "__main__":
    main()
