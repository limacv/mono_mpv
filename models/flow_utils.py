import numpy as np
import torch
import torch.nn.functional as torchf
import torch.nn as nn
from torchvision.transforms import ToTensor
import os
from .RAFT_network import RAFTNet

RAFT_path = {
    "small": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/raft-small.pth"),
    "sintel": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/raft-sintel.pth"),
    "kitti": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/raft-kitti.pth")
}


class FlowEstimator(nn.Module):
    def __init__(self, small=True, weight_name="small"):
        super().__init__()
        self.model = RAFTNet(small=small)
        self.model.cuda()
        state_dict = torch.load(RAFT_path[weight_name], map_location="cuda:0")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.totensor = ToTensor()
        self.offset = None

    def forward(self, im1: torch.Tensor, im2: torch.Tensor):
        """
        im1, im2 are tensor from (0, 1)
        """
        if im1.dim() == 3:
            im2 = im2.unsqueeze(0).cuda()
            im1 = im1.unsqueeze(0).cuda()
        im1 = im1.cuda()
        im2 = im2.cuda()

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
            flow = self.forward(im1, im2)
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


# Please use this
# Global_Flow_Estimator = FlowEstimator(False, 'sintel')
max_flow_thresh = 1e5

# visualize the flow
# ======================================================================================


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def flow_to_png_middlebury(flow: np.ndarray):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    if flow.shape[-1] == 2:
        u = flow[:, :, 0]
        v = flow[:, :, 1]
    elif flow.shape[0] == 2:
        u = flow[0]
        v = flow[1]
    else:
        raise ValueError(f"shape {flow.shape} doesn't like a flow")

    idxUnknow = (abs(u) > max_flow_thresh) | (abs(v) > max_flow_thresh)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)
