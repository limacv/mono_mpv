import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from models.mpi_network import *
from models.mpv_network import *
from models.mpi_flow_network import *
from models.mpifuse_network import *
from models.hourglass import *

from dataset.ConcatDataset import ConcatDatasetMy
from dataset.MannequinChallenge import *
from dataset.RealEstate10K import *
from dataset.StereoBlur import *
from dataset.StereoVideo import *
from models.rdn import RDN
from models.ModelWithLoss import *

from dataset.NvidiaNovelView import *

plane_num = 32


def str2bool(s_):
    try:
        s_ = int(s_)
    except ValueError:
        try:
            s_ = float(s_)
        except ValueError:
            if s_ == 'True':
                s_ = True
            elif s_ == 'False':
                s_ = False
    return s_


def select_module(name: str) -> nn.Module:
    if "MPINet" == name:
        return MPINet(plane_num)
    elif "MPINetv2" == name:
        return MPINetv2(plane_num)
    elif "hourglass" == name:
        return Hourglass(plane_num)
    elif "V5Nset_T" == name:
        num_set = 2
        return nn.ModuleDict({
            "MPI": MPI_V5Nset(plane_num, "cnn", num_set, thickmax=0.2),
            "SceneFlow": None,
            "AppearanceFlow": AFNet_HR_netflowin(netcnl=num_set * 3)
        })
    elif "V5Nset" in name:
        num_set = int(name[-1:])
        backbonename = "resnet" if "res" in name else "cnn"
        return nn.ModuleDict({
            "MPI": MPI_V5Nset(plane_num, backbonename, num_set),
            "SceneFlow": None,
            "AppearanceFlow": AFNet_HR_netflowin(netcnl=num_set * 3)
        })
    elif "V6Nset" in name:
        num_set = int(name[-1:])
        return nn.ModuleDict({
            "MPI": MPI_V6Nset(plane_num, num_set),
            "SceneFlow": None,
            "AppearanceFlow": AFNet_HR_netflowin(netcnl=num_set * 3)
        })
    elif "V6NewBG" == name:
        num_set = 2
        return nn.ModuleDict({
            "MPI": MPI_V6Nset(plane_num, num_set),
            "SceneFlow": None,
            "AppearanceFlow": AFNet(netcnl=num_set * 3)
        })
    elif "V5NewBG" == name:
        num_set = 2
        return nn.ModuleDict({
            "MPI": MPI_V5Nset(plane_num, num_set=num_set),
            "SceneFlow": None,
            "AppearanceFlow": AFNet(netcnl=num_set * 3)
        })
    elif "AB_alpha" == name:
        return nn.ModuleDict({
            "MPI": MPI_AB_alpha(plane_num),
            "SceneFlow": None,
            "AppearanceFlow": AFNet_HR_netflowin(netcnl=plane_num - 1)
        })
    elif "AB_up" == name:
        num_set = 2
        return nn.ModuleDict({
            "MPI": MPI_AB_up(plane_num, num_set=num_set),
            "SceneFlow": None,
            "AppearanceFlow": AFNet_HR_netflowin(netcnl=num_set * 3)
        })
    elif "AB_nonet" == name:
        return nn.ModuleDict({
            "MPI": MPI_AB_nonet(plane_num, num_set=2),
            "SceneFlow": None,
            "AppearanceFlow": AFNet_HR_netflowin(netcnl=2 * 3)
        })
    elif "AB_svdbg" == name:
        num_set = 2
        return nn.ModuleDict({
            "MPI": MPI_V6Nset(plane_num, num_set),
            "SceneFlow": None,
            "AppearanceFlow": AFNet_AB_svdbg(netcnl=num_set * 3)
        })
    else:
        raise ValueError(f"unrecognized modelin name: {name}")


def select_modelloss(name: str):
    name = name.lower()
    if "sv" == name:
        return ModelandSVLoss
    elif "disp_img" == name:
        return ModelandDispLoss
    elif "svjoint" == name:
        return ModelandLossJoint
    elif "fullv2" == name:
        return PipelineV2
    elif "fullsvv2" == name:
        return PipelineV2SV
    elif "fullv3" == name:
        return PipelineV3
    elif "fullv4" == name:
        return PipelineFiltering
    elif "fulljoint" == name:
        return PipelineJoint
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
        return RealEstate10K_Seq(istrain, seq_len=seq_len, mode=mode)
    elif "realestate10k_img" in name:
        return RealEstate10K_Img(istrain, mode=mode)
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
        return MannequinChallenge_Img(istrain, mode=mode)
    elif "mannequinchallenge_seq" in name:
        return MannequinChallenge_Seq(istrain, seq_len=seq_len, mode=mode)
    elif "mannequin+realestate_img" in name:
        dataset = ConcatDatasetMy([
            MannequinChallenge_Img(istrain, mode=mode),
            RealEstate10K_Img(istrain, mode=mode)
        ], [1, 0.025])
        dataset.name = "mannequin+realestate_img"
        return dataset
    elif "mannequin+realestate_seq" in name:
        dataset = ConcatDatasetMy([
            MannequinChallenge_Seq(istrain, seq_len=seq_len, mode=mode),
            RealEstate10K_Seq(istrain, seq_len=seq_len, mode=mode)
        ], [1, 0.025])
        dataset.name = "mannequin+realestate_seq"
        return dataset
    elif "m+r+s_seq" in name:
        dataset = ConcatDatasetMy([
            MannequinChallenge_Seq(istrain, seq_len=seq_len, mode=mode),
            RealEstate10K_Seq(istrain, seq_len=seq_len, mode=mode),
            StereoVideo_Seq(istrain, seq_len=seq_len, mode=mode)
        ], frequency=[1, 0.025, 1])
        dataset.name = "manne+realestate+stereovideo_seq"
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
    return begin_epoch


def smart_select_pipeline(checkpoint, force_modelname="auto", force_pipelinename="auto", **kwargs):
    cfg_str = torch.load(checkpoint)
    if "cfg" in cfg_str.keys():
        cfg_str = cfg_str["cfg"]
        loss_weights = cfg_str[cfg_str.find("\nLoss: ") + 7:
                               cfg_str.find('\n', cfg_str.find("\nLoss: ") + 1)].split(', ')
        loss_weights = [l_ for l_ in loss_weights if '[' not in l_ and ']' not in l_]
        loss_weights = {loss.split(':')[0]: str2bool(loss.split(':')[1]) for loss in loss_weights}
        modelname = cfg_str[cfg_str.find("Model: ") + 7: cfg_str.find(',', cfg_str.find("Model: "))]
        modellossname = cfg_str[cfg_str.find("ModelLoss: ") + 11: cfg_str.find('\n', cfg_str.find("ModelLoss: "))]
        if modellossname == "fulljoint":
            modellossname = "fullv2"
        elif modellossname == "svjoint":
            modellossname = "disp_img"
        modelname = modelname if force_modelname == "auto" else force_modelname
        modellossname = modellossname if force_pipelinename == "auto" else force_pipelinename
    else:
        if force_modelname == "auto" or force_pipelinename == "auto":
            raise RuntimeError("Cannot decide model for the given checkpoint")
        modelname, modellossname = force_modelname, force_pipelinename
        loss_weights = {}

    model = select_module(modelname).cuda()
    smart_load_checkpoint('', {"check_point": checkpoint}, model)
    pipeline = select_modelloss(modellossname)(model, {"loss_weights": loss_weights})
    return pipeline


def select_evalset(name: str, **kwargs):
    if "NvidiaNovelView" == name:
        return NvidiaNovelView(**kwargs)
    elif "StereoVideo" == name:
        return StereoVideo_Eval(**kwargs)
    else:
        raise RuntimeError(f"{name} not recognized")

