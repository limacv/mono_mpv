import numpy as np
import torch
import torch.nn.functional as torchf
import torchvision.utils
import cv2
import os
from typing import *
from .flow_utils import flow_to_png_middlebury

default_d_near = 1
default_d_far = 1000


def save_mpi(mpi: torch.Tensor, path):
    """
    mpi: tensor of shape [B, L, 4, H, W] or [L, 4, H, W]
    only first mpi will be saved
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if mpi.dim() == 5:
        mpi = mpi[0]
    mpi = (mpi.detach() * 255).type(torch.uint8)
    permute = torch.tensor([2, 1, 0, 3], dtype=torch.long)
    mpi = mpi[:, permute].permute(0, 2, 3, 1).cpu().numpy()  # rgb to bgr
    for i, layer in enumerate(mpi):
        cv2.imwrite(os.path.join(path, f"mpi{i:02d}.png"), layer)


def estimate_disparity_np(mpi: np.ndarray, min_depth=1, max_depth=100):
    """Compute disparity map from a set of MPI layers.
    mpi: np.ndarray or torch.Tensor
    From reference view.

    Args:
      layers: [..., L, H, W, C+1] MPI layers, back to front.
      depths: [..., L] depths for each layer.

    Returns:
      [..., H, W, 1] Single-channel disparity map from reference viewpoint.
    """
    num_plane, height, width, chnl = mpi.shape
    disparities = np.linspace(1. / max_depth, 1. / min_depth, num_plane)
    disparities = disparities.reshape(-1, 1, 1, 1)

    alpha = mpi[..., -1:]
    alpha = alpha * np.concatenate([np.cumprod(1 - alpha[::-1], axis=0)[::-1][1:],
                                    np.ones([1, height, width, 1])], axis=0)
    disparity = (alpha * disparities).sum(axis=0)
    # Weighted sum of per-layer disparities:
    return disparity.squeeze(-1)


def make_depths(num_plane, min_depth=default_d_near, max_depth=default_d_far):
    return torch.reciprocal(torch.linspace(1. / max_depth, 1. / min_depth, num_plane, dtype=torch.float32))


def estimate_disparity_torch(mpi: torch.Tensor, depthes: torch.Tensor, blendweight=None, retbw=False):
    """Compute disparity map from a set of MPI layers.
    mpi: tensor of shape B x LayerNum x 4 x H x W
    depthes: tensor of shape [B x LayerNum] or [B x LayerNum x H x W] (means different depth for each pixel]
    blendweight: optional blendweight that to reduce reduntante computation
    return: tensor of shape B x H x W
    """
    assert (mpi.dim() == 5)
    batchsz, num_plane, _, height, width = mpi.shape
    disparities = torch.reciprocal(depthes)
    if disparities.dim() != 4:
        disparities = disparities.reshape(batchsz, num_plane, 1, 1).type_as(mpi)

    alpha = mpi[:, :, -1, ...]  # alpha.shape == B x LayerNum x H x W
    if blendweight is None:
        blendweight = torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                                 torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    renderweight = alpha * blendweight
    disparity = (renderweight * disparities).sum(dim=1)

    if retbw:
        return disparity, blendweight
    else:
        return disparity


def netout2mpi(netout: torch.Tensor, img: torch.Tensor, bg_pct=1., ret_blendw=False):
    """
    Tranfer network output to multiplane image. i.e.:
    B x (LayerNum -1 + 3) x H x W ---> B x LayerNum x 4 x H x W
    img: [B, 3, H, W]
    bg_pct: the pecent of background and network output should be 0 < x <= 1
    netout: [B, 34, H, W]
    return: [B, 32, 4, H, W] and [B, 32, H, W] blend weight
    """
    batchsz, _, height, width = netout.shape
    alpha = netout[:, :-3, :, :]
    alpha = torch.cat([torch.ones([batchsz, 1, height, width]).type_as(alpha), alpha], dim=1)
    blend = torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                       torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    blend = blend.unsqueeze(2)
    if bg_pct == 1:
        img_bg = netout[:, -3:, :, :]
    else:
        img_bg = netout[:, -3:, :, :] * bg_pct + img * (1. - bg_pct)
    layer_rgb = blend * img.unsqueeze(1) + (-blend + 1.) * img_bg.unsqueeze(1)
    mpi = torch.cat([layer_rgb, alpha.unsqueeze(2)], dim=2)
    if ret_blendw:
        return mpi, blend.squeeze(2)
    else:
        return mpi


def alpha2mpi(alphas: torch.Tensor, fg: torch.Tensor, bg: torch.Tensor, bg_pct=1., blend_weight=None):
    """
    alphas: B x LayerNum x H x W
    fg / bg: B x LayerNum x c x H x W or B x c x H x W
    """
    batchsz, planenum, height, width = alphas.shape
    if blend_weight is None:
        blend_weight = torch.cat([torch.cumprod(- torch.flip(alphas, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                                  torch.ones([batchsz, 1, height, width]).type_as(alphas)], dim=1)

    if blend_weight.dim() == 4:
        blend_weight = blend_weight.unsqueeze(2)

    if fg.dim() == 4:  # layer dim
        fg = fg.unsqueeze(1)
    if bg.dim() == 4:  # layer dim
        bg = bg.unsqueeze(1)
    if bg_pct != 1.:
        bg = bg * bg_pct + fg * (1. - bg_pct)

    rgb = blend_weight * fg + (-blend_weight + 1.) * bg
    mpi = torch.cat([rgb, alphas.unsqueeze(2)], dim=2)
    return mpi


def netoutupdatempi_maskfree(netout: torch.Tensor, img: torch.Tensor, mpi_last, bg_pct=1., ret_blendw=False):
    """
    visiable_region -> img
    invisable_region -> +- mpi_last_valid -> mpi_last
                        └- mpi_last_invalid -> background
    Tranfer network output to multiplane image. i.e.:
    B x (LayerNum -1 + 3) x H x W ---> B x LayerNum x 4 x H x W
    img: [B, 3, H, W]
    mpi_last: [B, layernum, 4, H, W]
    bg_pct: the pecent of background and network output should be 0 < x <= 1
    netout: [B, 34, H, W]
    return: [B, 32, 4, H, W] and [B, 32, H, W] blend weight
    """
    batchsz, layernum, _, height, width = mpi_last.shape
    batchsz, cnl, height, width = netout.shape
    assert cnl == layernum - 1 + 3
    alpha = netout[:, :-3, :, :]
    alpha = torch.cat([torch.ones([batchsz, 1, height, width]).type_as(alpha), alpha], dim=1)
    blend = torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                       torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    blend = blend.unsqueeze(2)
    blend_mpi = (-blend + 1.) * mpi_last[:, :, -1:]
    blend_bg = (-blend + 1.) * (1 - mpi_last[:, :, -1:])

    if bg_pct == 1:
        img_bg = netout[:, -3:, :, :]
    else:
        img_bg = netout[:, -3:, :, :] * bg_pct + img * (1. - bg_pct)

    layer_rgb = blend * img.unsqueeze(1) + \
                blend_bg * img_bg.unsqueeze(1) + \
                blend_mpi * mpi_last[:, :, :3]
    mpi = torch.cat([layer_rgb, alpha.unsqueeze(2)], dim=2)
    if ret_blendw:
        return mpi, blend.squeeze(2)
    else:
        return mpi


def netoutupdatempi_withmask(netout: torch.Tensor, img: torch.Tensor, bg_pct=1., ret_blendw=False):
    """
    visiable_region -> img
    invisable_region -> +- mask==1 -> mpi_last
                        └- mask==0 -> background
    Tranfer network output to multiplane image. i.e.:
    B x (LayerNum -1 + 3) x H x W ---> B x LayerNum x 4 x H x W
    img: [B, 3, H, W]
    bg_pct: the pecent of background and network output should be 0 < x <= 1
    netout: [B, 34, H, W]
    return: [B, 32, 4, H, W] and [B, 32, H, W] blend weight
    """
    raise NotImplementedError


def overcompose(mpi: torch.Tensor, blendweight=None, ret_mask=False, blend_content=None) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    compose mpi back to front
    mpi: [B, 32, 4, H, W]
    blendweight: [B, 32, H, W] for reduce reduntant computation
    blendContent: [B, layernum, cnl, H, W], if not None,
    return: image of shape [B, 4, H, W]
        [optional: ] mask of shape [B, H, W], soft mask that
    """
    batchsz, num_plane, _, height, width = mpi.shape
    alpha = mpi[:, :, -1, ...]  # alpha.shape == B x LayerNum x H x W
    if blendweight is None:
        blendweight = torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                                 torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    renderw = alpha * blendweight

    content = mpi[:, :, :3, ...] if blend_content is None else blend_content
    rgb = (content * renderw.unsqueeze(2)).sum(dim=1)
    if ret_mask:
        return rgb, blendweight
    else:
        return rgb


def warp_homography(homos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """
    apply differentiable homography
    homos: [B x D x 3 x 3]
    images: [B x D x 4 x H x W]
    """
    batchsz, planenum, cnl, hei, wid = images.shape
    y, x = torch.meshgrid([torch.arange(hei), torch.arange(wid)])
    x, y = x.type_as(images), y.type_as(images)
    one = torch.ones_like(x)
    grid = torch.stack([x, y, one], dim=-1)  # grid: B x D x H x W x 3
    new_grid = homos.unsqueeze(-3).unsqueeze(-3) @ grid.unsqueeze(-1)
    new_grid = (new_grid.squeeze(-1) / new_grid[..., 2:3, 0])[..., 0:2]  # grid: B x D x H x W x 2
    new_grid = new_grid / torch.tensor([wid / 2, hei / 2]).type_as(new_grid) - 1.
    warpped = torchf.grid_sample(images.reshape(batchsz * planenum, cnl, hei, wid),
                                 new_grid.reshape(batchsz * planenum, hei, wid, 2), align_corners=True)
    return warpped.reshape(batchsz, planenum, cnl, hei, wid)


def warp_homography_withdepth(homos: torch.Tensor, images: torch.Tensor, depth: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Please note that homographies here are not scale invariant. make sure that rotation matrix has 1 det. R.det() == 1.
    apply differentiable homography
    homos: [B x D x 3 x 3]
    images: [B x D x 4 x H x W]
    depth: [B x D] or [B x D x 1] (depth in *ref space*)
    ret:
        the warpped mpi
        the warpped depth
    """
    batchsz, planenum, cnl, hei, wid = images.shape
    y, x = torch.meshgrid([torch.arange(hei), torch.arange(wid)])
    x, y = x.type_as(images), y.type_as(images)
    one = torch.ones_like(x)
    grid = torch.stack([x, y, one], dim=-1).reshape(1, 1, hei, wid, 3, 1)
    if depth.dim() == 4:
        depth = depth.reshape(batchsz, planenum, 1, hei, wid)
    else:
        depth = depth.reshape(batchsz, planenum, 1, 1, 1)

    new_grid = homos.unsqueeze(-3).unsqueeze(-3) @ grid
    new_depth = depth.reshape(batchsz, planenum, 1, 1) / new_grid[..., -1, 0]
    new_grid = (new_grid.squeeze(-1) / new_grid[..., 2:3, 0])[..., 0:2]  # grid: B x D x H x W x 2
    new_grid = new_grid / torch.tensor([wid / 2, hei / 2]).type_as(new_grid) - 1.
    warpped = torchf.grid_sample(images.reshape(batchsz * planenum, cnl, hei, wid),
                                 new_grid.reshape(batchsz * planenum, hei, wid, 2), align_corners=True)
    return warpped.reshape(batchsz, planenum, cnl, hei, wid), new_depth


def compute_homography(src_extrin: torch.Tensor, src_intrin: torch.Tensor,
                       tar_extrin: torch.Tensor, tar_intrin: torch.Tensor,
                       normal: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """
    compute homography matrix, detail pls see https://en.wikipedia.org/wiki/Homography_(computer_vision)
        explanation: assume P, P1, P2 be coordinate of point in plane in world, ref, tar space
        P1 = R1 @ P + t1               P2 = R2 @ P + t2
            so P1 = R @ P2 + t   where:
                R = R1 @ R2^T, t = t1 - R @ t2
        n^T @ P1 = d be plane equation in ref space,
            so in tar space: n'^T @ P2 = d'  where:
                n' = R^T @ n,    d' = d - n^T @ t

        so P1 = R @ P2 + d'^-1 t @ n'T @ P2 = (R + t @ n'^T @ R / (d - n^T @ t)) @ P2
    src_extrin/tar_extrin: [B, 3, 4] = [R | t]
    src_intrin/tar_intrin: [B, 3, 3]
    normal: [B, 3] normal of plane in *reference space*
    distances: [B, D] offset of plaen in *ref space*
        so the plane equation: n^T @ P1 = d  ==>  n'^T
    return: [B, D, 3, 3]
    """
    batchsz, _, _ = src_extrin.shape
    pad_tail = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(batchsz, 1, 1).type_as(src_extrin)
    src_pose_4x4 = torch.cat([
        src_extrin, pad_tail
    ], dim=-2)
    tar_pose_4x4 = torch.cat([
        tar_extrin, pad_tail
    ], dim=-2)
    pose = src_pose_4x4 @ torch.inverse(tar_pose_4x4)
    # rotation = R1 @ R2^T
    # translation = (t1 - R1 @ R2^T @ t2)
    rotation, translation = pose[..., :3, :3], pose[..., :3, 3:].squeeze(-1)
    distances_tar = -(normal.unsqueeze(-2) @ translation.unsqueeze(-1)).squeeze(-1) + distances

    # [..., 3, 3] -> [..., D, 3, 3]
    # multiply extra rotation because normal is in reference space
    homo = rotation.unsqueeze(-3) + (translation.unsqueeze(-1) @ normal.unsqueeze(-2) @ rotation).unsqueeze(-3) \
           / distances_tar.unsqueeze(-1).unsqueeze(-1)
    homo = src_intrin.unsqueeze(-3) @ homo @ torch.inverse(tar_intrin.unsqueeze(-3))
    return homo


def render_newview(mpi: torch.Tensor, srcextrin: torch.Tensor, tarextrin: torch.Tensor,
                   srcintrin: torch.Tensor, tarintrin: torch.Tensor,
                   depths: torch.Tensor, ret_mask=False, ret_dispmap=False) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                 Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """
    mpi: [B, 32, 4, H, W]
    srcpose&tarpose: [B, 3, 4]
    depthes: tensor of shape [Bx LayerNum]
    intrin: [B, 3, 3]
    ret: ref_view_images[, mask][, disparitys]
    """
    batchsz, planenum, _, hei, wid = mpi.shape

    planenormal = torch.tensor([0, 0, 1]).reshape(1, 3).repeat(batchsz, 1).type_as(mpi)
    distance = depths.reshape(batchsz, planenum)
    with torch.no_grad():
        # switching the tar/src pose since we have extrinsic but compute_homography uses poses
        # srcextrin = torch.tensor([1, 0, 0, 0,  # for debug usage
        #                           0, 1, 0, 0,
        #                           0, 0, 1, 0]).reshape(1, 3, 4).type_as(intrin)
        # tarextrin = torch.tensor([np.cos(0.3), -np.sin(0.3), 0, 0,
        #                           np.sin(0.3), np.cos(0.3), 0, 0,
        #                           0, 0, 1, 1.5]).reshape(1, 3, 4).type_as(intrin)
        homos = compute_homography(srcextrin, srcintrin, tarextrin, tarintrin,
                                   planenormal, distance)
    if not ret_dispmap:
        mpi_warp = warp_homography(homos, mpi)
        return overcompose(mpi_warp, ret_mask=ret_mask)
    else:
        mpi_warp, mpi_depth = warp_homography_withdepth(homos, mpi, distance)
        disparitys = estimate_disparity_torch(mpi_warp, mpi_depth)
        return overcompose(mpi_warp, ret_mask=ret_mask), disparitys


def shift_newview(mpi: torch.Tensor, disparities: torch.Tensor, ret_mask=False, ret_dispmap=False) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                 Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """
    mpi: [B, LayerNum, 4, H, W]
    depthes: tensor of shape [B x LayerNum], means the shift pixel number for each layer
        when have positive value, means translate to left (usually left view as reference will have positive disp)
    ret: ref_view_images[, mask][, disparitys]
    """
    batchsz, planenum, cnl, hei, wid = mpi.shape
    bpnum = batchsz * planenum
    affine = torch.eye(2, 3).reshape(1, 1, 2, 3).repeat(batchsz, planenum, 1, 1).type_as(disparities)
    affine[:, :, 0, -1] = (disparities / ((wid - 1) / 2))

    grid = torchf.affine_grid(affine.reshape(bpnum, 2, 3), [bpnum, cnl, hei, wid])
    mpi_warp = torchf.grid_sample(mpi.reshape(bpnum, cnl, hei, wid), grid, align_corners=True)
    mpi_warp = mpi_warp.reshape(batchsz, planenum, cnl, hei, wid)

    if ret_dispmap:
        disparitys = estimate_disparity_torch(mpi_warp, torch.reciprocal(disparities))
        return overcompose(mpi_warp, ret_mask=ret_mask), disparitys
    else:
        return overcompose(mpi_warp, ret_mask=ret_mask)


def warp_flow(content: torch.Tensor, flow: torch.Tensor, offset=None, pad_mode="zeros", mode="bilinear"):
    """
    content: [..., cnl, H, W]
    flow: [..., 2, H, W]
    """
    assert content.dim() == flow.dim()
    orishape = content.shape
    cnl, hei, wid = content.shape[-3:]
    mpi = content.reshape(-1, cnl, hei, wid)
    flow = flow.reshape(-1, 2, hei, wid).permute(0, 2, 3, 1)

    if offset is None:
        y, x = torch.meshgrid([torch.arange(hei), torch.arange(wid)])
        x, y = x.type_as(mpi), y.type_as(mpi)
        offset = torch.stack([x, y], dim=-1)
    grid = offset.reshape(1, hei, wid, 2) + flow
    normanator = torch.tensor([(wid - 1) / 2, (hei - 1) / 2]).reshape(1, 1, 1, 2).type_as(grid)
    warpped = torchf.grid_sample(mpi, grid / normanator - 1., padding_mode=pad_mode, mode=mode, align_corners=True)
    return warpped.reshape(orishape)


class MPVWriter:
    def __init__(self, path):
        self.out = cv2.VideoWriter()
        self.outa = cv2.VideoWriter()
        self.path = path
        self.patha = path.split('.')[0] + "_alpha.mp4"
        self.size = (0, 0)
        self.planenum = 0

    def write(self, mpi: torch.Tensor, post=True):
        mpi = (mpi * 255).type(torch.uint8)

        mpipad = torchvision.utils.make_grid(mpi, nrow=8, padding=0)
        cnl, hei, wid = mpipad.shape
        mpipad = mpipad.cpu().numpy()
        rgb = mpipad[:3][::-1].transpose(1, 2, 0)
        a = mpipad[3]

        if not self.out.isOpened():
            self.out.open(self.path, 828601953, 30., (wid, hei), True)
            self.outa.open(self.patha, 828601953, 30., (wid, hei), False)
            if not self.out.isOpened():
                raise RuntimeError(f"MPVWriter::cannot open {self.path}")
            if not self.outa.isOpened():
                raise RuntimeError(f"MPVWriter:;cannot open {self.patha}")
        self.out.write(rgb)
        self.outa.write(a)

    def __del__(self):
        self.out.release()


class MPFWriter:
    def __init__(self, path, mpfout=True, spfout=True):
        self.out = cv2.VideoWriter()
        self.outa = cv2.VideoWriter()
        self.path = path.split('.')[0] + "_mpf.mp4"
        self.patha = path.split('.')[0] + "_final.mp4"
        self.mpfout, self.finalout = mpfout, spfout
        self.size = (0, 0)
        self.planenum = 0

    def write(self, mpi: torch.Tensor, mpf: torch.Tensor):
        if self.mpfout:
            mpfpad = torchvision.utils.make_grid(mpf, nrow=8, padding=0)
            mpfvis = flow_to_png_middlebury(mpfpad.cpu().numpy())
            hei, wid, _ = mpfvis.shape

            if not self.out.isOpened():
                self.out.open(self.path, 828601953, 30., (wid, hei), True)
                if not self.out.isOpened():
                    raise RuntimeError(f"MPVWriter::cannot open {self.path}")
            self.out.write(mpfvis)

        if self.finalout:
            final = overcompose(mpi.unsqueeze(0), blend_content=mpf.unsqueeze(0))[0]
            finalvis = flow_to_png_middlebury(final.cpu().numpy())
            hei, wid, _ = finalvis.shape
            if not self.outa.isOpened():
                self.outa.open(self.patha, 828601953, 30., (wid, hei), True)
                if not self.outa.isOpened():
                    raise RuntimeError(f"MPVWriter:;cannot open {self.patha}")
            self.outa.write(finalvis)

    def __del__(self):
        self.out.release()


class NetWriter:
    def __init__(self, path):
        self.out = cv2.VideoWriter()
        self.path = path.split('.')[0] + "_net.mp4"
        self.size = (0, 0)

    def write(self, net: torch.Tensor):
        net = (net * 255).type(torch.uint8)
        if net.shape[1] == 10:
            ones = torch.ones_like(net[:, 0:1])
            net = torch.cat([net[:, :2], ones, net[:, 2:4], ones, net[:, 4:]], dim=1)
        layer1, layer2, imfg, imbg = torch.split(net.squeeze(0), 3, dim=0)
        savefig = torchvision.utils.make_grid([layer1, layer2, imfg, imbg], nrow=2, padding=0)

        cnl, hei, wid = savefig.shape
        mpipad = savefig.cpu().numpy()
        rgb = mpipad[:3][::-1].transpose(1, 2, 0)

        if not self.out.isOpened():
            self.out.open(self.path, 828601953, 30., (wid, hei), True)
            if not self.out.isOpened():
                raise RuntimeError(f"MPVWriter::cannot open {self.path}")
        self.out.write(rgb)

    def __del__(self):
        self.out.release()


def dilate(alpha: torch.Tensor):
    """
    alpha: B x L x H x W
    """
    batchsz, layernum, hei, wid = alpha.shape
    alphaunfold = torch.nn.Unfold(3, padding=1, stride=1)(alpha.reshape(-1, 1, hei, wid))
    alphaunfold = alphaunfold.max(dim=1)[0]
    return alphaunfold.reshape_as(alpha)


def erode(alpha: torch.Tensor):
    """
    alpha: B x L x H x W
    """
    batchsz, layernum, hei, wid = alpha.shape
    alphaunfold = torch.nn.Unfold(3, padding=1, stride=1)(alpha.reshape(-1, 1, hei, wid))
    alphaunfold = alphaunfold.min(dim=1)[0]
    return alphaunfold.reshape_as(alpha)


def visibility_mask(mpi: torch.Tensor):
    alpha = mpi[0, :, -1]
    render_weight = alpha * torch.cat([torch.cumprod(- torch.flip(alpha, dims=[0]) + 1, dim=0).flip(dims=[0])[1:],
                                       torch.ones_like(alpha[0:1])], dim=0)
    return render_weight


def matplot_mpi(mpi: torch.Tensor, alpha=True, visibility=False, RGBA=False, nrow=8):
    import matplotlib.pyplot as plt
    plt.figure()
    with torch.no_grad():
        if alpha:
            if not visibility:
                img = torchvision.utils.make_grid(mpi[0, :, -1:].detach(), pad_value=1, nrow=nrow)
            else:
                img = visibility_mask(mpi)
                img = torchvision.utils.make_grid(img.unsqueeze(1).detach(), pad_value=1, nrow=nrow)
            img = img[0].cpu().numpy()
        else:
            if RGBA:
                img = torchvision.utils.make_grid(mpi[0].detach(), nrow=nrow)
            else:
                img = torchvision.utils.make_grid(mpi[0, :, :3].detach(), nrow=nrow)
            img = img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.show()


def matplot_img(img):
    import matplotlib.pyplot as plt
    plt.figure()
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        if img.shape[0] == 3 or img.shape[0] == 4:
            img = img.permute(1, 2, 0)
        elif img.shape[0] == 1:
            img = img[0]
        img = img.detach().cpu()
    elif isinstance(img, np.ndarray):
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        elif img.shape[0] == 1:
            img = img[0]
    plt.imshow(img)
    plt.show()


def matplot_net(net):
    import matplotlib.pyplot as plt
    plt.figure()
    if net.dim() != 4:
        assert True, "not a net"
    net = net[0]
    if net.shape[0] == 6:
        net = torch.cat([
            torch.cat([net[0], net[1], net[2]], dim=-1),
            torch.cat([net[3], net[4], net[5]], dim=-1),
        ], dim=-2)
    elif net.shape[0] == 4:
        net = torch.cat([
            torch.cat([net[0], net[1]], dim=-1),
            torch.cat([net[2], net[3]], dim=-1),
        ], dim=-2)
    else:
        assert True, "not a net"
    img = net.detach().cpu().numpy()
    plt.imshow(img)
    plt.show()


def matplot_upmask(upmask, dim=0):
    import matplotlib.pyplot as plt
    bsz, c, hei, wid = upmask.shape
    assert c == 8*8*9, "incorrect channel numebr"
    img = upmask.detach().reshape(bsz, 9, 8, 8, hei, wid)[0, dim]\
        .permute(2, 0, 3, 1).reshape(hei * 8, wid * 8).cpu().numpy()
    plt.imshow(img)
    plt.show()


def matplot_flow(flow: torch.Tensor, maxflow=None):
    import matplotlib.pyplot as plt
    plt.figure()
    vis = flow_to_png_middlebury(flow[0].detach().cpu().numpy(), maxflow=maxflow)
    plt.imshow(vis)
    plt.show()


def matplot_mpf(mpf: torch.Tensor, alphampi=None):
    import matplotlib.pyplot as plt
    plt.figure()
    mpfpad = torchvision.utils.make_grid(mpf[0], nrow=8, pad_value=1)
    mpfvis = flow_to_png_middlebury(mpfpad.cpu().numpy())
    if alphampi is not None:
        mask = visibility_mask(alphampi)
        mask = torchvision.utils.make_grid(mask.unsqueeze(1).detach(), pad_value=1)
        mpfvis = mpfvis.astype(np.float32) * mask[0].unsqueeze(-1).cpu().numpy()
        mpfvis = mpfvis.astype(np.uint8)

    plt.imshow(mpfvis)
    plt.show()
