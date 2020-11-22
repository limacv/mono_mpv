import numpy as np
import torch
import torch.nn.functional as torchf
import torch.nn as nn
from torchvision.transforms import ToTensor
import os
from .RAFT_network import RAFTNet

RAFT_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/raft-small.pth")


class FlowEstimator:
    def __init__(self, small=True):
        self.model = RAFTNet(small=small)
        self.model.cuda()
        state_dict = torch.load(RAFT_weights_path, map_location="cuda:0")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.totensor = ToTensor()
        self.offset = None

    def estimate_flow_torch(self, im1: torch.Tensor, im2: torch.Tensor):
        """
        im1, im2 are tensor from (0, 1)
        """
        if im1.dim() == 3:
            im2 = im2.unsqueeze(0).cuda()
            im1 = im1.unsqueeze(0).cuda()

        with torch.no_grad():
            flow = self.model(im1, im2)
        return flow

    def estimate_flow_np(self, im1: np.ndarray, im2: np.ndarray):
        im1 = self.totensor(im1).unsqueeze(0).cuda()
        im2 = self.totensor(im2).unsqueeze(0).cuda()
        with torch.no_grad():
            flow = self.model(im1, im2)
        return flow

    def estimate_flow(self, im1, im2, np_out=False):
        if isinstance(im1, np.ndarray):
            flow = self.estimate_flow_np(im1, im2)
        elif isinstance(im2, torch.Tensor):
            flow = self.estimate_flow_torch(im1, im2)
        else:
            raise NotImplementedError

        if np_out:
            flow = flow.cpu().numpy()
        return flow

    def warp_by_flow(self, im1, flow):
        if isinstance(im1, np.ndarray):
            im1 = self.totensor(im1).unsqueeze(0)
        elif im1.dim() == 3:
            im1 = im1.unsqueeze(0)
        im1 = im1.cuda()

        _, _, hei, wid = im1.shape
        if self.offset is None or self.offset.shape[-2] != (hei, wid):
            offsety, offsetx = torch.meshgrid([
                    torch.linspace(0, hei - 1, hei),
                    torch.linspace(0, wid - 1, wid)
                ])
            self.offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).type(im1.type()).cuda()

        flow += self.offset
        im1_warpped = torchf.grid_sample(im1, flow)
        return im1_warpped
