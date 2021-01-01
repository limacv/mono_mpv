import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchf
from typing import Sequence

from ._modules import BasicUpdateBlock, SmallUpdateBlock
from ._modules import BasicEncoder, SmallEncoder
from ._modules import CorrBlock
from ._modules import coords_grid, upflow8


# default args: alternate_corr = False; mixed_precision = False; small = False;
class RAFTNet(nn.Module):
    def __init__(self, small=False):
        super(RAFTNet, self).__init__()
        if small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_radius = 4

        if small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=0)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=0)
            self.update_block = SmallUpdateBlock(hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=0)
            self.update_block = BasicUpdateBlock(hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    @staticmethod
    def initialize_flow(shapeformat, init_flow=None):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = shapeformat.shape
        coords0 = coords_grid(N, H, W).to(shapeformat.device)
        coords1 = coords_grid(N, H, W).to(shapeformat.device)
        if init_flow is not None:
            batchsz, cnl, hei, wid = init_flow.shape
            coords1[:, :, :hei, :wid] += init_flow

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    @staticmethod
    def upsample_flow(flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = torchf.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def encode_only(self, frame):
        if frame.min() >= 0:
            frame = 2 * frame - 1.
        fmap = self.fnet(frame)
        return [fmap, frame]

    def forward(self, image1, image2, init_flow=None, iters=12):
        """ Estimate optical flow between pair of frames """
        # for compatiability
        if isinstance(image1, Sequence):
            fmap1, image1 = image1
            fmap2, image2 = image2
        elif isinstance(image1, torch.Tensor):
            image1 = 2 * image1 - 1.0
            image2 = 2 * image2 - 1.0
            image1 = image1.contiguous()
            image2 = image2.contiguous()
            # run the feature network
            fmap1, fmap2 = self.fnet([image1, image2])
        else:
            raise ValueError("RAFT: image not Tensor or list of Tensor")

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        hdim = self.hidden_dim
        cdim = self.context_dim

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(fmap1, init_flow)

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up[:, :, :image1.shape[2], :image1.shape[3]])

        return flow_predictions[-1]
