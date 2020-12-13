import torch
from models.mpi_network import *
from models.mpifuse_network import *
from models.mpi_utils import *
from models.mpi_flow_network import *
#
# state_dict1 = torch.load('./flowgradin_041055_r0.pth')
# state_dict2 = torch.load('./mpfnet_learnonlympf_031801_r0.pth')
#
# state_dict1 = {"MPI." + k: v for k, v in state_dict1["state_dict"].items()}
# state_dict2 = {k: v for k, v in state_dict2["state_dict"].items() if "MPF." in k}
#
# state_dict1.update(state_dict2)
# torch.save({"state_dict": state_dict1, "epoch": 300}, "./flowgradin_mpfnet.pth")

MPFNet
