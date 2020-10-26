from main import select_module
import torch
import numpy as np
from dataset.sintel_seq import SintelSeq
from models.ModelWithLoss import ModelandUnsuLoss
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import trange

model_name = "RAFTNet"
data_savepath = "./log/" + model_name + "/viz_coarse.npy"
error_savepath = "./log/" + model_name + "/errors.txt"
check_point = "./log/" + model_name + "/0.ckpt"
data_path = "D:/MSI_NB/source/data/Sintel/"
item_name = "training/final/"

err_resolution = 1  # divide flow error into 0.01 bin
max_error = 1000
flow_resolution = 1
max_flow = 300

epe_x = np.arange(0, max_error, err_resolution)
flow_x = np.arange(0, max_flow, flow_resolution)
flow_axis, epe_axis = np.meshgrid(flow_x[:-1], epe_x[:-1])
epe_count = np.zeros_like(epe_axis)


if os.path.isfile(data_savepath):
    data = np.load(data_savepath, allow_pickle=True)
    flow_axis = data.item()["x"]
    epe_axis = data.item()["y"]
    epe_count = data.item()["z"]
    epe_count = np.log(epe_count + 1)
    # plot the error histogram
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(flow_axis[:299], epe_axis[:299], epe_count[:299], cmap=cm.coolwarm, linewidth=0)
    ax.set_xlabel("EPE Mag")
    ax.set_ylabel("Flow Mag")
    ax.set_zlabel("log(number)")
    plt.show()
    # exit()


def main():
    global epe_count
    model = select_module(model_name)
    model.load_state_dict(torch.load(check_point)["state_dict"])
    modelandloss = ModelandUnsuLoss(model, {})
    with open("log/RAFTNet/errors.txt", 'a') as errfile:
        errfile.writelines("Sintel final training:\n")
    data_img_path = data_path + item_name
    for seq_root, seq_paths, _ in os.walk(data_img_path):
        for seq_path in seq_paths:
            dataset = SintelSeq(seq_root + seq_path)
            dataset_aepe = []
            print(f"processing: {seq_path}")
            for frameidx in trange(dataset.framenum - 1):
                frame0, frame1 = dataset.get_tensorframe(frameidx), dataset.get_tensorframe(frameidx + 1)
                frame0, frame1 = frame0.unsqueeze(0).cuda(), frame1.unsqueeze(0).cuda()
                gtflow, gtocc = dataset.get_gtflow(frameidx), dataset.get_gtocc(frameidx)
                eflow, eocc = modelandloss.infer_forward(frame0, frame1)
                eflow = eflow[0].permute(1, 2, 0).cpu().numpy()

                gtflow = gtflow.reshape(-1, 2)
                eflow = eflow.reshape(-1, 2)
                gtocc = gtocc.reshape(-1)
                gtflow_mag = np.linalg.norm(gtflow, axis=-1)
                errflow_mag = np.linalg.norm(gtflow - eflow, axis=-1)
                epe = np.mean(errflow_mag)
                dataset_aepe.append(epe)

                gtflow_mag = gtflow_mag[gtocc < 0.5]
                errflow_mag = errflow_mag[gtocc < 0.5]
                epe_count_cur, x, y = np.histogram2d(gtflow_mag, errflow_mag, [flow_x, epe_x])
                epe_count = epe_count_cur.T + epe_count

            with open("log/RAFTNet/errors.txt", 'a') as errfile:
                dataset_aepe = sum(dataset_aepe) / len(dataset_aepe)
                errfile.writelines(f"{seq_path}: epe={dataset_aepe}\n")
            np.save(data_savepath, {"x": flow_axis, "y": epe_axis, "z": epe_count})

    # final save
    np.save(data_savepath, {"x": flow_axis, "y": epe_axis, "z": epe_count})
    print(f"saved to {data_savepath}")
    # plot the error histogram
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(flow_axis, epe_axis, epe_count, cmap=cm.coolwarm, linewidth=0)
    ax.set_xlabel("Flow Mag")
    ax.set_ylabel("EPE Mag")
    ax.set_zlabel("number")
    plt.show()


if __name__ == "__main__":
    main()
