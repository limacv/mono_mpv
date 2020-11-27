from .models.flow_utils import *
from .models.mpi_network import *
from .models.mpi_utils import *

model = MPINet(32)
model.cuda()
model.load_state_dict(torch.load(".\\log\\checkpoint\\stereoblur_img_230031.pth")["state_dict"])

flow_est = FlowEstimator(True)

