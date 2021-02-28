import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as torchf
import numpy as np
import cv2
from .mpi_utils import *
from typing import Sequence
from typing import List, Sequence, Union, Tuple, Dict

'''
implement all kinds of losses
'''
const_use_sobel = True


class ParamScheduler:
    def __init__(self, milestones, values, mode='linear'):
        """
        milestone should be sorted
        mode shoule be eighter one string or list of string that len > len(milestone-1)
        """
        assert len(milestones) == len(values), "scheduler::len(milestones) == len(values)"
        if isinstance(mode, List):
            assert len(mode) >= len(milestones) - 1, "scheduler::len(mode) >= len(milestones) - 1"
        self.milestone = np.array(milestones, dtype=np.longlong)
        self.values = values
        self.modes = mode[:(len(milestones) - 1)] if isinstance(mode, List) else [mode] * (len(milestones) - 1)

        assert all([m_ in ['linear', 'step', 'expo'] for m_ in self.modes])

    def get_value(self, step):
        stage_idx = np.searchsorted(self.milestone, step)
        if stage_idx == 0:
            return self.values[0]
        elif stage_idx == len(self.milestone):
            return self.values[-1]
        else:
            id0, id1 = stage_idx - 1, stage_idx
            mode = self.modes[id0]
            val0, val1 = self.values[id0], self.values[id1]
            pct = (step - self.milestone[id0]) / (self.milestone[id1] - self.milestone[id0])
            if mode == 'linear':
                return val0 + (val1 - val0) * pct
            elif mode == 'step':
                return val0 if pct < 0.5 else val1
            elif mode == 'expo':
                # todo: implement this
                return val0 + (val1 - val0) * pct
            else:
                return val0 + (val1 - val0) * pct


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.blocks = torch.nn.ModuleList([
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),
            torchvision.models.vgg16(pretrained=True).features[16:23].eval()
        ])
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, im, target):
        batchsz, cnl, hei, wid = im.shape
        im = (im - self.mean) / self.std
        target = (target-self.mean) / self.std
        loss = torch.tensor([0]).type_as(im)
        x = torch.stack([im, target], dim=0).reshape(-1, cnl, hei, wid)  # alone batch dim
        for block in self.blocks:
            x = block(x)
            batchsz, cnl, hei, wid = x.shape
            loss += torchf.l1_loss(x.reshape(2, -1, cnl, hei, wid)[0],
                                   x.reshape(2, -1, cnl, hei, wid)[1])
        return loss


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.patchsz = 2 * 1 + 1
        self.filter_func = nn.AvgPool2d(self.patchsz, 1, 0)

    def compute_mean(self, _x):
        return self.filter_func(torchf.pad(_x, [1, 1, 1, 1], 'replicate'))

    def forward(self, x, y):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        u_x = self.compute_mean(x)
        u_y = self.compute_mean(y)
        u_x_u_y = u_x * u_y
        u_x_u_x = u_x.pow(2)
        u_y_u_y = u_y.pow(2)

        sigma_x = self.compute_mean(x * x) - u_x_u_x
        sigma_y = self.compute_mean(y * y) - u_y_u_y
        sigma_xy = self.compute_mean(x * y) - u_x_u_y

        ssim_n = (2 * u_x_u_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (u_x_u_x + u_y_u_y + c1) * (sigma_x + sigma_y + c2)
        ssim = ssim_n / ssim_d
        ssim = torch.mean(ssim, dim=1)
        dist = torch.clamp((- ssim + 1.) / 2, 0, 1).mean()
        return dist


class PhotoL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, im, im_warp, order=1):
        assert im.dim() == 4 and im_warp.dim() == 4
        diff = torch.abs(im - im_warp) + 0.0001
        diff = torch.sum(diff, dim=1)
        if order != 1:
            diff = torch.pow(diff, order)
        return diff  # * scale


# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
class TernaryLoss(nn.Module):
    def __init__(self, max_distance=1):
        super().__init__()
        self.patchsz = 2 * max_distance + 1
        self.max_dist = max_distance

    def forward(self, im, im_warp):
        t1 = self._ternary_transform(im)
        t2 = self._ternary_transform(im_warp)
        dist = self._hamming_distance(t1, t2)
        mask = self._valid_mask(im, self.max_dist)

        return dist * mask

    def _ternary_transform(self, image):
        intensities = self._rgb_to_grayscale(image) * 255
        out_channels = self.patchsz * self.patchsz
        w = torch.eye(out_channels).view((out_channels, 1, self.patchsz, self.patchsz))
        weights = w.type_as(image)
        patches = torchf.conv2d(intensities, weights, padding=self.max_dist)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    @staticmethod
    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    @staticmethod
    def _hamming_distance(_t1, _t2):
        _dist = torch.pow(_t1 - _t2, 2)
        dist_norm = _dist / (0.1 + _dist)
        dist_mean = torch.mean(dist_norm, 1)  # instead of sum
        return dist_mean

    @staticmethod
    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, h - 2 * padding, w - 2 * padding).type_as(t)
        _mask = torchf.pad(inner, [padding] * 4)
        return _mask


def draw_dense_disp(disp: torch.Tensor, scale) -> np.ndarray:
    display = (disp[0].detach() * 255 * scale).type(torch.uint8).cpu().numpy()
    display = cv2.cvtColor(cv2.applyColorMap(display, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
    return display


def draw_sparse_depth(im: torch.Tensor, poses: torch.Tensor, depth: torch.Tensor, colormap="HOT") -> np.ndarray:
    if im.dim() == 4:
        im = im[0]
    if poses.dim() == 3:
        poses = poses[0]
    if depth.dim() == 2:
        depth = depth[0]
    im_np = (im.detach().permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy()
    hei, wid, _ = im_np.shape
    poses = (poses.detach().cpu().numpy() + 1.) * np.array([wid, hei]).reshape(1, 2) / 2.
    poses = poses.astype(np.float32)
    depth = (depth.detach() * 255).type(torch.uint8).cpu().numpy().reshape(-1, 1)
    if colormap == "JET":
        depth_color = cv2.cvtColor(cv2.applyColorMap(depth, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).squeeze(1)
    else:
        depth_color = cv2.cvtColor(cv2.applyColorMap(depth, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB).squeeze(1)
    im_np = cv2.UMat(im_np)
    for color, pos in zip(depth_color, poses):
        cv2.circle(im_np, (pos[0], pos[1]), 2,
                   (int(color[0]), int(color[1]), int(color[2])), -1)
    return cv2.UMat.get(im_np)


def select_photo_loss(mode: str) -> nn.Module:
    if mode in ["L1", "l1", "l1_loss", "L1_loss"]:
        return PhotoL1Loss()
    elif mode in ["ssim", "SSIM", "ssim_loss", "SSIM_loss"]:
        return SSIMLoss()
    elif mode in ["ternary", "ternary_loss"]:
        return TernaryLoss()
    elif mode in ["vgg", "VGG"]:
        return VGGPerceptualLoss()
    else:
        raise NotImplementedError(f"loss_utils::{mode} not recognized")


# from ARFlow
gaussian_kernel = \
    torch.tensor((
        (0.095332,	0.118095,	0.095332),
        (0.118095,	0.146293,	0.118095),
        (0.095332,	0.118095,	0.095332),
    )).reshape((1, 1, 3, 3)).type(torch.float32)


def gradient(data, order=1):
    data_cnl = data.shape[1]
    sobel_x = torch.tensor(((1., 0, -1.),
                            (2., 0, -2.),
                            (1., 0, -1.))).repeat(data_cnl, 1, 1, 1).reshape((data_cnl, 1, 3, 3)).type_as(data)
    sobel_y = torch.tensor(((1., 2., 1.),
                            (0, 0, 0),
                            (-1., -2., -1.))).repeat(data_cnl, 1, 1, 1).reshape((data_cnl, 1, 3, 3)).type_as(data)

    data_dx, data_dy = data, data
    for i in range(order):
        data_dx = torchf.conv2d(torchf.pad(data_dx, [1, 1, 1, 1], 'replicate'), sobel_x, groups=data_cnl)
        data_dy = torchf.conv2d(torchf.pad(data_dy, [1, 1, 1, 1], 'replicate'), sobel_y, groups=data_cnl)
    # D_dy = dataset[:, :, 1:] - dataset[:, :, :-1]
    # D_dx = dataset[:, :, :, 1:] - dataset[:, :, :, :-1]
    return data_dx, data_dy


def smooth_grad(disp: torch.Tensor, image: torch.Tensor, e_min=0.05, g_min=0.02, inverseedge=False):
    """
    edge-aware smooth loss
    :param disp: [B, H, W] or [B, 1, H, W]
    :param image: [B, 3, H, W]
    :return: ret: smoothness map [B, H, W]
    """
    if disp.dim() == 3:
        disp = disp.unsqueeze(-3)
    img_dx, img_dy = gradient(image)
    disp_dx, disp_dy = gradient(disp)
    batchsz, _, hei, wid = image.shape

    grad_im = (img_dx.abs() + img_dy.abs()).sum(dim=-3, keepdim=True)
    grad_im_max = torch.max(grad_im.reshape(batchsz, -1), dim=-1)[0].reshape(-1, 1, 1, 1)
    grad_disp = disp_dx.abs() + disp_dy.abs()

    weights = torch.min(grad_im / (e_min * grad_im_max), torch.tensor(1.).type_as(disp))
    if not inverseedge:
        weights = - weights + 1
        
    smooth = torch.max(grad_disp - g_min, torch.tensor(0.).type_as(disp))
    return weights * smooth

