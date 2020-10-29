from dataset.RealEstate10K import RealEstate10K
from models.ModelWithLoss import ModelandLoss
from models.mpi_network import MPINet
import torch
import numpy as np

np.random.seed(5)
torch.manual_seed(0)

model = MPINet(32)
model.load_state_dict(torch.load("./log/MPINet/mpinet_ori.pth")["state_dict"])
modelloss = ModelandLoss(model, {
        "pixel_loss": 1,
        "smooth_loss": 0.5,
        "depth_loss": 0.1,
    })

dataset = RealEstate10K(False)
datas = dataset[1]
datas = [data.unsqueeze(0) for data in datas]

modelloss.valid_forward(*datas, visualize=True)
modelloss.train_forward(*datas)
for data in dataset:
    print(data)
