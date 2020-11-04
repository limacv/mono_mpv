import torch

weight = torch.load("./mpinet_ori.pth")
new_weight = {"state_dict": weight,
              "epoch": 0}
torch.save(new_weight, "./mpinet_ori.pth")
