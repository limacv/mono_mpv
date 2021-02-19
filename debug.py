from utils import *
import random

seed = 3358  # np.random.randint(0, 10000)
print(f"random seed = {seed}")
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)


def main(kwargs):
    batchsz = kwargs["batchsz"]
    model = select_module(kwargs["modelname"])

    smart_load_checkpoint("./log/checkpoint/", kwargs, model)
    lr_scheduler = ParamScheduler(
        milestones=[10e3, 50e3, 100e3, 150e3],
        values=[2, 1, 0.5, 0.2]
    )
    lr_scheduler.get_value(120e3)
    modelloss = select_modelloss(kwargs["pipelinename"])(model, kwargs)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    optimizer.param_groups[0]["lr"] = 1
    dataset = select_dataset(kwargs["datasetname"], kwargs["istrain"], {})
    for i in range(int(14000)):
        datas_all = [[]] * 7
        for dev in range(batchsz):
            datas = dataset[6]
            datas_all = [ds_ + [d_] for ds_, d_ in zip(datas_all, datas)]

        datas = [torch.stack(data, dim=0).cuda() for data in datas_all]
        if kwargs["istrain"]:
            loss_dict = modelloss(*datas, step=i)
            loss = loss_dict["loss"]
            loss = loss.mean()
            loss_dict = loss_dict["loss_dict"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_dict = {k: v.mean() for k, v in loss_dict.items()}

            # output iter infomation
            loss_str = " | ".join([f"{k}:{v:.3f}" for k, v in loss_dict.items()])
            print(f"loss:{loss:.3f} | {loss_str}", flush=True)
        else:
            _val_dict = modelloss.valid_forward(*datas, visualize=True)
            val_str = " | ".join([f"{k}: {v:.3f}" for k, v in _val_dict.items() if "val_" in k])
            print(f"{val_str}")

        # if i % 25 == 0:
        #     datas = [d_[:batchsz] for d_ in datas]
        #     _val_dict = modelloss.valid_forward(*datas, visualize=True)
        #
        # if i % 100 == 0 and "savefile" in kwargs.keys():
        #     torch.save(model.state_dict(), kwargs["savefile"])


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
    "modelname": "V5NewBG",  # MPINetv2, Fullv6, Fullv5.Fullv5resnet, V5Nset[res]N, AB_alpha, AB_nonet, AB_up, AB_svdbg
    "pipelinename": "fulljoint",  # sv, disp_img, fullv2, fullsvv2, fulljoint, svjoint
    "datasetname": "mannequinchallenge_seq",
    # stereovideo_img, stereovideo_seq, mannequinchallenge_img, mannequinchallenge_seq, mannequin+realestate_img
    # mannequin+realestate_seq, m+r+s_seq, realestate10k_seq, realestate10k_img
    "istrain": True,
    "check_point": {
        # "": "mpinet_ori.pth",
    },

    "device_ids": [0],
    # "device_ids": [0, 1, 2, 3, 4, 5, 6, 7],
    "batchsz": 1,
    # "checkpoint": "./log/MPINet/mpinet_ori.pth",
    # "savefile": "./log/DBG_pretrain.pth",
    "logdir": "./log/run/debug_svscratch",
    "savefile": "./log/checkpoint/debug_svscratch.pth",
    "loss_weights": {"pixel_loss_cfg": 'l1',
                     "pixel_loss": 1,
                     "scale_mode": "adaptive",
                     "flownet_dropout": 1,
                     # "net_smth_loss_fg": 0.5,
                     # "net_smth_loss_bg": 0.5,
                     "depth_loss": 1,
                     "depth_loss_mode": "fine",
                     "alpha_thick_in_disparity": False,
                     "tempdepth_loss_milestone": [2e3, 4e3],
                     "mask_warmup": 0.5,
                     "mask_warmup_milestone": [1e18, 2e18],
                     "bgflow_warmup": 1,
                     "bgflow_warmup_milestone": [4e3, 6e3],
                     # "net_warmup": 0.5,
                     # "net_warmup_milestone": [1e18, 2e18],
                     # "aflow_fusefgpct": False,
                     "bg_supervision": 0.1,
                     "net_smth_loss": 1,
                     "tempnewview_mode": "biflow",
                     "tempnewview_loss": 1,
                     # "net_std": 0.2,
                     "upmask_magaware": True
                     },
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
