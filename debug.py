from utils import *

np.random.seed(6666)
torch.manual_seed(6666)


def main(kwargs):
    batchsz = kwargs["batchsz"]
    model = select_module("Fullv41")

    smart_load_checkpoint("./log/checkpoint/", kwargs, model)

    model.cuda()
    modelloss = select_modelloss("fullv2")(model, kwargs)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    dataset = select_dataset("stereovideo_seq", True, {})
    for i in range(int(14000)):
        datas_all = [[]] * 7
        for dev in range(1):
            datas = dataset[0]
            datas_all = [ds_ + [d_] for ds_, d_ in zip(datas_all, datas)]

        datas = [torch.stack(data, dim=0).cuda() for data in datas_all]
        with torch.autograd.set_detect_anomaly(True):
            loss_dict = modelloss(*datas, step=i)
            loss = loss_dict["loss"]
            loss = loss.mean()
            loss_dict = loss_dict["loss_dict"]
            optimizer.zero_grad()
            loss.backward()
        _val_dict = modelloss.valid_forward(*datas, visualize=True)
        optimizer.step()
        loss_dict = {k: v.mean() for k, v in loss_dict.items()}

        # output iter infomation
        loss_str = " | ".join([f"{k}:{v:.3f}" for k, v in loss_dict.items()])
        print(f"loss:{loss:.3f} | {loss_str}", flush=True)
        if i % 25 == 0:
            datas = [d_[:batchsz] for d_ in datas]
            _val_dict = modelloss.valid_forward(*datas, visualize=True)

        if i % 100 == 0 and "savefile" in kwargs.keys():
            torch.save(model.state_dict(), kwargs["savefile"])


# complete cfg:
# cuda_device=0
# checkpoint="./log/MPINet/mpinet_ori.pth", [opt]
# logdir="./log/run/dbg_/", [opt]
# mpioutdir="./log/Debug1", [opt]
# savefile="./log/Debug_all.pth", [opt]
# loss_cfg={"pixel_loss": 1,
#        "smooth_loss": 0.5,
#        "depth_loss": 0.1},


main({
    "device_ids": [0],
    # "device_ids": [0, 1, 2, 3, 4, 5, 6, 7],
    "check_point": {
        # "": "v4_raftnet_masksupervise_162138_r0.pth",
    },
    "batchsz": 1,
    # "checkpoint": "./log/MPINet/mpinet_ori.pth",
    # "savefile": "./log/DBG_pretrain.pth",
    "logdir": "./log/run/debug_svscratch",
    "savefile": "./log/checkpoint/debug_svscratch.pth",
    "loss_weights": {"pixel_loss": 1,
                     "pixel_loss_cfg": "vgg",
                     "smooth_loss": 0.5,
                     "depth_loss": 0.1,
                     "depth_loss_mode": "hat",
                     "depth_loss_ord": 0.5,
                     "depth_loss_rmresidual": True,
                     "mask_loss": 1,
                     "mpi_flowgrad_in": False,
                     "templ1_loss": 1,
                     "tempdepth_loss": 0.01,
                     "dilate_mpfin": False,
                     "alpha2mpf": True,
                     "aflow_fusefgpct": True,
                     "sflow_mode": "backward",
                     "pipe_optim_frame0": True,
                     "aflow_residual": True,
                     "aflow_includeself": False,
                     "flow_epe": 0.1,
                     "flow_smth": 0.1,
                     # "sflow_loss": 0.1,
                     "smooth_flowgrad_loss": 0.1,
                     "temporal_loss_mode": "mse"},
})
# good data list
# 01bfb80e5b8fe757
# 0b49ce385d8cdadc
# 1ab9594c7015ecdc
# 1d6248d4db20fdc3
# 02ecb10e9363e210
# 2a7c6acafe4422b4
# 2aabcf66c52b7343
# 2aacd794386dbdc3
# 2af41ef10045d1c3
# 2bfe6a3923cf9a83
# 2cf0df5ede50bd1d
# 3b22f4f99f2bf2f5
