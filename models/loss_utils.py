import torch
import torch.nn as nn
import torch.nn.functional as torchf
from typing import List, Sequence, Union, Tuple, Dict

'''
implement all kinds of losses
'''
const_use_sobel = True


def photometric_loss(im, im_warp, order=1):
    # scale = 3. / im.shape[1]  # normalize the loss to as if there were three channels
    diff = torch.abs(im - im_warp) + 0.0001  # why?
    diff = torch.sum(diff, dim=1) / 3.
    if order != 1:
        diff = torch.pow(diff, order)
    return diff  # * scale


# from ARFlow
gaussian_kernel = \
    torch.tensor((
        (0.095332,	0.118095,	0.095332),
        (0.118095,	0.146293,	0.118095),
        (0.095332,	0.118095,	0.095332),
    )).reshape((1, 1, 3, 3)).type(torch.float32)


def ssim_loss(im, im_warp):
    x, y = im, im_warp
    # data_cnl = x.shape[1]
    # global gaussian_kernel
    # gaussian_kernel = gaussian_kernel.type_as(x).repeat((data_cnl, 1, 1, 1))
    #
    # def filter_func(_x):
    #     return torchf.conv2d(torchf.pad(_x, [1, 1, 1, 1], 'replicate'), gaussian_kernel, groups=data_cnl)
    patch_size = 2 * 1 + 1
    filter_func = nn.AvgPool2d(patch_size, 1, 0)

    def compute_mean(_x):
        return filter_func(torchf.pad(_x, [1, 1, 1, 1], 'replicate'))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    u_x = compute_mean(x)
    u_y = compute_mean(y)
    u_x_u_y = u_x * u_y
    u_x_u_x = u_x.pow(2)
    u_y_u_y = u_y.pow(2)

    sigma_x = compute_mean(x * x) - u_x_u_x
    sigma_y = compute_mean(y * y) - u_y_u_y
    sigma_xy = compute_mean(x * y) - u_x_u_y

    ssim_n = (2 * u_x_u_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (u_x_u_x + u_y_u_y + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    ssim = torch.mean(ssim, dim=1, keepdim=True)
    dist = torch.clamp((- ssim + 1.) / 2, 0, 1)
    return dist


# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def ternary_loss(im, im_warp, max_distance=1):
    """
    measure similarity of im and im_warp
    :param im: Bx3xHxW
    :param max_distance:
    """
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = torchf.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(_t1, _t2):
        _dist = torch.pow(_t1 - _t2, 2)
        dist_norm = _dist / (0.1 + _dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        _mask = torchf.pad(inner, [padding] * 4)
        return _mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask


def gradient(data, order=1):
    data_cnl = data.shape[1]
    sobel_x = torch.tensor(((0.5, 0, -0.5),
                            (1., 0, -1.),
                            (0.5, 0, -0.5))).repeat(data_cnl, 1, 1, 1).reshape((data_cnl, 1, 3, 3)).type_as(data)
    sobel_y = torch.tensor(((0.5, 1., 0.5),
                            (0, 0, 0),
                            (-0.5, -1., -0.5))).repeat(data_cnl, 1, 1, 1).reshape((data_cnl, 1, 3, 3)).type_as(data)

    data_dx, data_dy = data, data
    for i in range(order):
        data_dx = torchf.conv2d(torchf.pad(data_dx, [1, 1, 1, 1], 'replicate'), sobel_x, groups=data_cnl)
        data_dy = torchf.conv2d(torchf.pad(data_dy, [1, 1, 1, 1], 'replicate'), sobel_y, groups=data_cnl)
    # D_dy = dataset[:, :, 1:] - dataset[:, :, :-1]
    # D_dx = dataset[:, :, :, 1:] - dataset[:, :, :, :-1]
    return data_dx, data_dy


def smooth_grad(disp: torch.Tensor, image: torch.Tensor, e_min=0.1, g_min=0.05):
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

    grad_im = (img_dx.abs() + img_dy.abs()).sum(dim=-3, keepdim=True)
    grad_disp = disp_dx.abs() + disp_dy.abs()

    weights = - torch.min(grad_im / (e_min * grad_im.max()), torch.tensor(1.).type_as(disp)) + 1.
    smooth = torch.max(grad_disp - g_min, torch.tensor(0.).type_as(disp))
    return weights.squeeze(-3) * smooth.squeeze(-3)

