import numpy as np
import torch
import torch.nn.functional as torchf
import cv2
import os
from typing import *

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


def estimate_disparity_torch(mpi: torch.Tensor, depthes: torch.Tensor, blendweight=None):
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
        blendweight = alpha * torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                                         torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    else:
        blendweight = alpha * blendweight
    disparity = (blendweight * disparities).sum(dim=1)

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


def overcompose(mpi: torch.Tensor, blendweight=None, ret_mask=False)\
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    compose mpi back to front
    mpi: [B, 32, 4, H, W]
    blendweight: [B, 32, H, W] for reduce reduntant computation
    return: image of shape [B, 4, H, W]
        [optional: ] mask of shape [B, H, W], soft mask that
    """
    batchsz, num_plane, _, height, width = mpi.shape
    alpha = mpi[:, :, -1, ...]  # alpha.shape == B x LayerNum x H x W
    if blendweight is None:
        blendweight = alpha * torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                                         torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    else:
        blendweight = alpha * blendweight
    rgb = (mpi[:, :, :3, ...] * blendweight.unsqueeze(2)).sum(dim=1)
    if not ret_mask:
        return rgb
    else:
        return rgb, blendweight.sum(dim=-3)


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
                                 new_grid.reshape(batchsz * planenum, hei, wid, 2))
    return warpped.reshape(batchsz, planenum, cnl, hei, wid)


def get_tardepth_homography(homosref2tar: torch.Tensor, refdepth: torch.Tensor) -> torch.Tensor:
    """
    given depth value in *ref space*, and a homograph that warp 3D point coord from *ref im space* to *tar im space*
    note that here homo is actually a 3D transformation matrix that transform 3D point coordinates
    explanation:
        let P1 be point in ref im space [x1, y1, d1], which has constant depth in our case
            P2 be point in tar im space [x2, y2, d2], x2, y2 is known and d2 is what we want
        P1 = H @ P2
        d1 = [H31, H32] @ [x2, y2]^T + H33 * d2
    """
    pass


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
                                 new_grid.reshape(batchsz * planenum, hei, wid, 2))
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

        so P1 = R @ P2 + d'^-1 t @ n'T @ P2 = (R + t @ n'^T @ R) @ P2
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


def render_newview(mpi: torch.Tensor, srcextrin: torch.Tensor, tarextrin: torch.Tensor, intrin: torch.Tensor,
                   depths: torch.Tensor, ret_mask=False, ret_dispmap=False)\
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
        homos = compute_homography(srcextrin, intrin, tarextrin, intrin,
                                   planenormal, distance)
    if not ret_dispmap:
        mpi_warp = warp_homography(homos, mpi)
        return overcompose(mpi_warp, ret_mask=ret_mask)
    else:
        mpi_warp, mpi_depth = warp_homography_withdepth(homos, mpi, distance)
        disparitys = estimate_disparity_torch(mpi_warp, mpi_depth)
        return overcompose(mpi_warp, ret_mask=ret_mask), disparitys

