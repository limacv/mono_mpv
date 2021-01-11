import torch

weight = torch.load("../log/checkpoint/mpinet_ori.pth")
newweight = {}
for k, v in weight["state_dict"].items():
    if "output" in k:
        continue
    newweight[k] = v
new_weight = {"state_dict": newweight,
              "epoch": 0}
torch.save(new_weight, "../log/checkpoint/mpinet_pretrain.pth")
