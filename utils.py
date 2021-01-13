import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from models.mpi_network import *
from models.mpv_network import *
from models.mpi_flow_network import *
from models.mpifuse_network import *
from models.hourglass import *

from dataset.MannequinChallenge import *
from dataset.RealEstate10K import *
from dataset.StereoBlur import *
from dataset.StereoVideo import *

from models.ModelWithLoss import *

plane_num = 24


def select_module(name: str) -> nn.Module:
    if "MPINet" == name:
        return MPINet(plane_num)
    elif "MPINetv2" == name:
        return MPINetv2(plane_num)
    elif "hourglass" == name:
        return Hourglass(plane_num)
    elif "MPVNet" == name:
        return MPVNet(plane_num)
    elif "MPI2InMPF" == name:
        return MPINet2In(plane_num)
    elif "MPIReccuNet" == name:
        return MPIReccuNet(plane_num)
    elif "MPIRecuFlowNet" == name:
        return MPIRecuFlowNet(plane_num)
    elif "Full" == name:
        return nn.ModuleDict({
            "MPI": MPINetv2(plane_num),
            "Fuser": MPIFuser(plane_num)
        })
    elif "MPFNet" == name:
        return nn.ModuleDict({
            "MPI": MPINetv2(plane_num),
            "MPF": MPFNet(plane_num)
        })
    elif "MPFNetv2" == name:
        return nn.ModuleDict({
            "MPI": MPI_FlowGrad(plane_num),
            "MPF": MPFNet(plane_num)
        })
    elif "MPI_FlowGrad" == name:
        return MPI_FlowGrad(plane_num)
    # elif "Fullv1" == name:
    #     return nn.ModuleDict({
    #         "MPI": MPI_Fullv1(plane_num),
    #         "MPF": MPFNet(plane_num)
    #     })
    elif "Fullv20" == name:
        return nn.ModuleDict({
            "MPI": MPI_alpha(plane_num),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": AMPFNetAIn2D(plane_num)
        })
    elif "Fullv21" == name:
        return nn.ModuleDict({
            "MPI": MPI_alpha(plane_num),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetAIn(plane_num)
        })
    elif "Fullv22" == name:
        return nn.ModuleDict({
            "MPI": MPI_alpha(plane_num),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetDIn()
        })
    elif "Fullv220" == name:
        return nn.ModuleDict({
            "MPI": MPI_alpha(plane_num),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv221" == name:
        return nn.ModuleDict({
            "MPI": MPI_alpha(plane_num),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskInOut()
        })
    elif "Fullv2" == name:  # the ultimate model
        return nn.ModuleDict({
            "MPI": MPI_alpha(plane_num),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv23a" == name:
        return nn.ModuleDict({
            "MPI": MPI_alpha(plane_num),
            "SceneFlow": SceneFlowNet_alpha(plane_num),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv23i" == name:
        return nn.ModuleDict({
            "MPI": MPI_alpha(plane_num),
            "SceneFlow": SceneFlowNet_img(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv241" == name:
        return nn.ModuleDict({
            "MPI": MPI_gaussian1(plane_num),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv242" == name:
        return nn.ModuleDict({
            "MPI": MPI_gaussian2(plane_num),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv243" == name:
        return nn.ModuleDict({
            "MPI": MPI_square(plane_num, 2),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv244" == name:
        return nn.ModuleDict({
            "MPI": MPI_square(plane_num, 3),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv245" == name:
        return nn.ModuleDict({
            "MPI": MPI_square(plane_num, 2 + 2),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    elif "Fullv246" == name:
        return nn.ModuleDict({
            "MPI": MPI_square(plane_num, 3 + 3),
            "SceneFlow": SceneFlowNet(),
            "AppearanceFlow": ASPFNetWithMaskOut()
        })
    else:
        raise ValueError(f"unrecognized modelin name: {name}")


def select_modelloss(name: str):
    name = name.lower()
    if "sv" == name:
        return ModelandSVLoss
    elif "disp_img" == name:
        return ModelandDispLoss
    # elif "fullv1" == name:
    #     return ModelandFullv1Loss
    # elif "fullsvv1" == name:
    #     return ModelandFullSVv1Loss
    elif "fullv2" == name:
        return PipelineV2
    elif "fullsvv2" == name:
        return PipelineV2SV
    elif "fullv3" == name:
        return PipelineV3
    else:
        raise ValueError(f"unrecognized modelloss name: {name}")


def select_dataset(name: str, istrain: bool, cfg) -> Dataset:
    name = name.lower()
    if istrain:
        mode = "crop"
        seq_len = 5
    else:
        mode = "resize"
        seq_len = 10
    if "realestate10k_seq" in name:
        return RealEstate10K_Seq(istrain, seq_len=seq_len)
    elif "realestate10k_img" in name:
        return RealEstate10K_Img(istrain)
    elif "stereoblur_img" in name:
        return StereoBlur_Img(istrain)
    elif "stereoblur_seq" in name:
        return StereoBlur_Seq(istrain, seq_len=seq_len)
    elif "stereovideo_img" in name:
        return StereoVideo_Img(istrain, mode=mode)
    elif "stereovideo_seq" in name:
        return StereoVideo_Seq(istrain, seq_len=seq_len, mode=mode)
    elif "stereovideo_test" in name:
        return StereoVideo_Seq(istrain, seq_len=seq_len, test=True, mode=mode)
    elif "mannequinchallenge_img" in name:
        return MannequinChallenge_Img(istrain)
    elif "mannequinchallenge_seq" in name:
        return MannequinChallenge_Seq(istrain, seq_len=seq_len)
    elif "mannequin+realestate_img" in name:
        dataset = ConcatDataset([RealEstate10K_Img(istrain), MannequinChallenge_Img(istrain)])
        dataset.name = "mannequin+realestate_img"
        return dataset
    elif "mannequin+realestate_seq" in name:
        dataset = ConcatDataset([RealEstate10K_Seq(istrain, seq_len=seq_len),
                                 MannequinChallenge_Seq(istrain, seq_len=seq_len)])
        dataset.name = "mannequin+realestate_seq"
        return dataset
    else:
        raise NotImplementedError(f"dataset name {name} not recognized")


def mkdir_ifnotexist(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def smart_load_checkpoint(root, cfg, model: nn.Module):
    """
    root: root dir of checkpoint file
    cfgcheckpoint: {prefix: filepath} or filepath
    model: the loaded model
    return: begin_epoch
    """
    cfgcheckpoint = cfg["check_point"]
    if not isinstance(cfgcheckpoint, dict):
        cfgcheckpoint = {"": cfgcheckpoint}

    device = f"cuda:{cfg['local_rank']}" if "local_rank" in cfg.keys() else "cpu"

    # initializing weights
    initial_weights(model)
    newstate_dict = model.state_dict()
    begin_epoch = 0
    for prefix, path in cfgcheckpoint.items():
        try:
            check_point = torch.load(os.path.join(root, path), map_location=device)
        except FileNotFoundError:
            print(f"cannot open check point file {path}, initial the model")
            return begin_epoch

        temp_state_dict = {k: v for k, v in check_point["state_dict"].items() if k.startswith(prefix)}
        if len(temp_state_dict) == 0:
            newstate_dict.update(
                {f"{prefix}" + k_: v_ for k_, v_ in check_point["state_dict"].items()}
            )
        else:
            newstate_dict.update(temp_state_dict)
        if prefix == "" and "epoch" in check_point.keys():
            begin_epoch = check_point["epoch"]
        if "cfg" in check_point.keys():
            print(f"load checkpoint {path} with config: \n{check_point['cfg']}\n")
    model.load_state_dict(newstate_dict)
    print(f"load state dict, epoch starting from: {begin_epoch}")
    return 0  # begin_epoch
