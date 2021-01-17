import numpy as np
import torch
import torch.nn.functional as torchf
import torch.nn as nn
from torchvision.transforms import ToTensor
import os
from .RAFT_network import RAFTNet
from ._modules import downflow8

RAFT_path = {
    "small": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/raft-small.pth"),
    "sintel": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/raft-sintel.pth"),
    "kitti": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/raft-kitti.pth")
}


def forward_scatter(flow01: torch.Tensor, content: torch.Tensor, offset=None):
    """
    :param flow01: Bx2xHxWthe target pos in im1
    :param content: BxcxHxW the scatter content
    :return: rangemap of im1
    """
    batchsz, _, hei, wid = flow01.shape
    cnl = content.shape[1]
    if offset is None:
        offsety, offsetx = torch.meshgrid([
            torch.linspace(0, hei - 1, hei),
            torch.linspace(0, wid - 1, wid)
        ])
        offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).type_as(flow01)
    coords = (flow01 + offset).permute(0, 2, 3, 1)
    coords_floor = torch.floor(coords).int()
    coords_offset = coords - coords_floor

    batch_range = torch.arange(batchsz).reshape(batchsz, 1, 1).to(flow01.device)
    idx_batch_offset = (batch_range * hei * wid).repeat([1, hei, wid])

    content_flat = content.permute(0, 2, 3, 1).reshape([-1, cnl])
    coords_floor_flat = coords_floor.reshape([-1, 2])
    coords_offset_flat = coords_offset.reshape([-1, 2])
    idx_batch_offset_flat = idx_batch_offset.reshape([-1])

    idxs_list, weights_list = [], []
    content_list = []
    for di in range(2):
        for dj in range(2):
            idxs_i = coords_floor_flat[:, 0] + di
            idxs_j = coords_floor_flat[:, 1] + dj

            idxs = idx_batch_offset_flat + idxs_j * wid + idxs_i

            mask = torch.bitwise_and(
                torch.bitwise_and(idxs_i >= 0, idxs_i < wid),
                torch.bitwise_and(idxs_j >= 0, idxs_j < hei)).reshape(-1)
            valid_idxs = idxs[mask]
            valid_offsets = coords_offset_flat[mask]
            content_cur = content_flat[mask]

            weights_i = (1. - di) - (-1) ** di * valid_offsets[:, 0]
            weights_j = (1. - dj) - (-1) ** dj * valid_offsets[:, 1]
            weights = weights_i * weights_j

            idxs_list.append(valid_idxs)
            weights_list.append(weights)
            content_list.append(content_cur)

    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    content = torch.cat(content_list, dim=0)

    newcontent = torch.zeros(batchsz * hei * wid, cnl).to(content.device)
    content *= weights.unsqueeze(-1)
    newcontent = newcontent.scatter_add_(0, idxs.unsqueeze(-1).expand(-1, cnl), content)
    denorm = torch.zeros(batchsz * hei * wid).to(content.device)
    denorm = denorm.scatter_add_(0, idxs, weights).unsqueeze(-1) + 0.001
    newcontent = (newcontent / denorm).reshape(batchsz, hei, wid, cnl).permute(0, 3, 1, 2)

    return newcontent.type_as(flow01)


def forward_scatter_withweight(flow01: torch.Tensor, content: torch.Tensor, softmask: torch.Tensor, offset=None):
    """
    :param flow01: Bx2xHxWthe target pos in im1
    :param content: BxcxHxW the scatter content
    :return: rangemap of im1
    """
    batchsz, _, hei, wid = flow01.shape
    content = torch.cat([content, softmask], dim=1)
    cnl = content.shape[1]
    if offset is None:
        offsety, offsetx = torch.meshgrid([
            torch.linspace(0, hei - 1, hei),
            torch.linspace(0, wid - 1, wid)
        ])
        offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).type_as(flow01)
    coords = (flow01 + offset).permute(0, 2, 3, 1)
    coords_floor = torch.floor(coords).int()
    coords_offset = coords - coords_floor

    batch_range = torch.arange(batchsz).reshape(batchsz, 1, 1).to(flow01.device)
    idx_batch_offset = (batch_range * hei * wid).repeat([1, hei, wid])

    content_flat = content.permute(0, 2, 3, 1).reshape([-1, cnl])
    coords_floor_flat = coords_floor.reshape([-1, 2])
    coords_offset_flat = coords_offset.reshape([-1, 2])
    idx_batch_offset_flat = idx_batch_offset.reshape([-1])

    idxs_list, weights_list = [], []
    content_list = []
    for di in range(2):
        for dj in range(2):
            idxs_i = coords_floor_flat[:, 0] + di
            idxs_j = coords_floor_flat[:, 1] + dj

            idxs = idx_batch_offset_flat + idxs_j * wid + idxs_i

            mask = torch.bitwise_and(
                torch.bitwise_and(idxs_i >= 0, idxs_i < wid),
                torch.bitwise_and(idxs_j >= 0, idxs_j < hei)).reshape(-1)
            valid_idxs = idxs[mask]
            valid_offsets = coords_offset_flat[mask]
            content_cur = content_flat[mask]

            weights_i = (1. - di) - (-1) ** di * valid_offsets[:, 0]
            weights_j = (1. - dj) - (-1) ** dj * valid_offsets[:, 1]
            weights = weights_i * weights_j

            idxs_list.append(valid_idxs)
            weights_list.append(weights)
            content_list.append(content_cur)

    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    content = torch.cat(content_list, dim=0)
    weights = weights * content[:, -1]
    content = content[:, :-1]
    cnl -= 1
    newcontent = torch.zeros(batchsz * hei * wid, cnl).to(content.device)
    content = content * weights.unsqueeze(-1)
    newcontent = newcontent.scatter_add_(0, idxs.unsqueeze(-1).expand(-1, cnl), content)
    denorm = torch.zeros(batchsz * hei * wid).to(content.device)
    denorm = denorm.scatter_add_(0, idxs, weights).unsqueeze(-1) + 0.00000001
    newcontent = (newcontent / denorm).reshape(batchsz, hei, wid, cnl).permute(0, 3, 1, 2)

    return newcontent.type_as(flow01)


def forward_scatter_mpi(flow01: torch.Tensor, mpi: torch.Tensor):
    """
    two modifications regarding the forward_scatter:
        1. valid mask = alpha > epsi
        2. final blending not only consider the bilinear weights, but also the alpha as weights
    :param flow01: BxLx2xHxWthe target pos in im1 (or B x 2 x H x W)
    :param mpi: BxLxcxHxW the scatter content
    :return: rangemap of im1
    """
    batchsz, layernum, _, hei, wid = mpi.shape
    if flow01.dim() == 4:
        flow01 = flow01.unsqueeze(1).expand(-1, layernum, -1, -1, -1)
    mpi = mpi.reshape(batchsz * layernum, -1, hei, wid)
    flow01 = flow01.reshape(batchsz * layernum, -1, hei, wid)
    batchsz = batchsz * layernum
    cnl = mpi.shape[1]
    offsety, offsetx = torch.meshgrid([
        torch.linspace(0, hei - 1, hei),
        torch.linspace(0, wid - 1, wid)
    ])
    offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).type_as(flow01)
    coords = (flow01 + offset).permute(0, 2, 3, 1)
    coords_floor = torch.floor(coords).int()
    coords_offset = coords - coords_floor

    batch_range = torch.arange(batchsz).reshape(batchsz, 1, 1).to(flow01.device)
    idx_batch_offset = (batch_range * hei * wid).repeat([1, hei, wid])

    mpi_flat = mpi.permute(0, 2, 3, 1).reshape([-1, cnl])
    coords_floor_flat = coords_floor.reshape([-1, 2])
    coords_offset_flat = coords_offset.reshape([-1, 2])
    idx_batch_offset_flat = idx_batch_offset.reshape([-1])

    idxs_list, weights_list = [], []
    content_list = []
    for di in range(2):
        for dj in range(2):
            idxs_i = coords_floor_flat[:, 0] + di
            idxs_j = coords_floor_flat[:, 1] + dj

            idxs = idx_batch_offset_flat + idxs_j * wid + idxs_i

            mask = torch.bitwise_and(
                torch.bitwise_and(idxs_i >= 0, idxs_i < wid),
                torch.bitwise_and(idxs_j >= 0, idxs_j < hei)).reshape(-1)
            valid_idxs = idxs[mask]
            valid_offsets = coords_offset_flat[mask]
            content_cur = mpi_flat[mask]

            weights_i = (1. - di) - (-1) ** di * valid_offsets[:, 0]
            weights_j = (1. - dj) - (-1) ** dj * valid_offsets[:, 1]
            weights = weights_i * weights_j

            idxs_list.append(valid_idxs)
            weights_list.append(weights)
            content_list.append(content_cur)

    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    mpi = torch.cat(content_list, dim=0)
    alphas = mpi[:, -1]
    weights = alphas * weights

    newmpi = torch.zeros(batchsz * hei * wid, cnl).to(mpi.device)
    mpi = mpi * weights.unsqueeze(-1)
    newmpi = newmpi.scatter_add_(0, idxs.unsqueeze(-1).expand(-1, cnl), mpi)
    denorm = torch.zeros(batchsz * hei * wid).to(mpi.device)
    denorm = denorm.scatter_add_(0, idxs, weights).unsqueeze(-1) + 0.0001
    newmpi = (newmpi / denorm).reshape(batchsz, hei, wid, cnl).permute(0, 3, 1, 2)

    return newmpi.reshape(-1, layernum, cnl, hei, wid)


def forward_scatter_mpi_HRLR(flow01: torch.Tensor, mpi: torch.Tensor):
    """
    modification regarding the forward_scatter_mpi:
        first scatter to higher resolution (x2) then maxpooling to downsample
    :param flow01: BxLx2xHxWthe target pos in im1 (or B x 2 x H x W)
    :param mpi: BxLxcxHxW the scatter content
    :return: rangemap of im1
    """
    batchsz, layernum, _, hei, wid = mpi.shape
    if flow01.dim() == 4:
        flow01 = flow01.unsqueeze(1).expand(-1, layernum, -1, -1, -1)
    flow01 = flow01 * 2
    mpi = mpi.reshape(batchsz * layernum, -1, hei, wid)
    flow01 = flow01.reshape(batchsz * layernum, -1, hei, wid)
    batchsz = batchsz * layernum
    cnl = mpi.shape[1]
    offsety, offsetx = torch.meshgrid([
        torch.linspace(0, (hei - 1) * 2, hei),
        torch.linspace(0, (wid - 1) * 2, wid)
    ])
    offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).type_as(flow01)
    coords = (flow01 + offset).permute(0, 2, 3, 1)
    coords_floor = torch.floor(coords).int()
    coords_offset = coords - coords_floor

    batch_range = torch.arange(batchsz).reshape(batchsz, 1, 1).to(flow01.device)
    idx_batch_offset = (batch_range * hei * wid * 4).repeat([1, hei, wid])

    mpi_flat = mpi.permute(0, 2, 3, 1).reshape([-1, cnl])
    coords_floor_flat = coords_floor.reshape([-1, 2])
    coords_offset_flat = coords_offset.reshape([-1, 2])
    idx_batch_offset_flat = idx_batch_offset.reshape([-1])

    idxs_list, weights_list = [], []
    content_list = []
    for di in range(2):
        for dj in range(2):
            idxs_i = coords_floor_flat[:, 0] + di
            idxs_j = coords_floor_flat[:, 1] + dj

            idxs = idx_batch_offset_flat + idxs_j * wid * 2 + idxs_i

            mask = torch.bitwise_and(
                torch.bitwise_and(idxs_i >= 0, idxs_i < wid * 2 - 1),
                torch.bitwise_and(idxs_j >= 0, idxs_j < hei * 2 - 1)).reshape(-1)
            valid_idxs = idxs[mask]
            valid_offsets = coords_offset_flat[mask]
            content_cur = mpi_flat[mask]

            weights_i = (1. - di) - (-1) ** di * valid_offsets[:, 0]
            weights_j = (1. - dj) - (-1) ** dj * valid_offsets[:, 1]
            weights = weights_i * weights_j

            idxs_list.append(valid_idxs)
            weights_list.append(weights)
            content_list.append(content_cur)

    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    mpi = torch.cat(content_list, dim=0)
    alphas = mpi[:, -1]
    weights = alphas * weights

    newmpi = torch.zeros(batchsz * hei * wid * 4, cnl).to(mpi.device)
    denorm = torch.zeros(batchsz * hei * wid * 4).to(mpi.device)
    mpi = mpi * weights.unsqueeze(-1)
    newmpi = newmpi.scatter_add_(0, idxs.unsqueeze(-1).expand(-1, cnl), mpi)
    denorm = denorm.scatter_add_(0, idxs, weights) + 0.0001

    denorm, maxidx = denorm.reshape(batchsz, hei, 2, wid, 2)\
        .permute(0, 1, 3, 2, 4).reshape(batchsz, hei, wid, 4).max(dim=-1, keepdim=True)
    newmpi = newmpi.reshape(batchsz, hei, 2, wid, 2, cnl)\
        .permute(0, 1, 3, 2, 4, 5).reshape(batchsz, hei, wid, 4, cnl)\
        .gather(-2, maxidx.unsqueeze(-1).expand(-1, -1, -1, -1, cnl)).squeeze(-2)
    newmpi = (newmpi / denorm).permute(0, 3, 1, 2)

    return newmpi.reshape(-1, layernum, cnl, hei, wid)


def forward_scatter_mpi_nearest(flow01: torch.Tensor, mpi: torch.Tensor):
    """
    modifications regarding the forward_scatter_mpi:
        splat to 4 pixel -> splat to nearest 1 pixel
    :param flow01: BxLx2xHxWthe target pos in im1 (or B x 2 x H x W)
    :param mpi: BxLxcxHxW the scatter content
    :return: rangemap of im1
    """
    batchsz, layernum, _, hei, wid = mpi.shape
    if flow01.dim() == 4:
        flow01 = flow01.unsqueeze(1).expand(-1, layernum, -1, -1, -1)
    mpi = mpi.reshape(batchsz * layernum, -1, hei, wid)
    flow01 = flow01.reshape(batchsz * layernum, -1, hei, wid)
    batchsz = batchsz * layernum
    cnl = mpi.shape[1]
    offsety, offsetx = torch.meshgrid([
        torch.linspace(0, hei - 1, hei),
        torch.linspace(0, wid - 1, wid)
    ])
    offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).type_as(flow01)
    coords = (flow01 + offset).permute(0, 2, 3, 1)
    coords_floor = torch.floor(coords).int()

    batch_range = torch.arange(batchsz).reshape(batchsz, 1, 1).to(flow01.device)
    idx_batch_offset = (batch_range * hei * wid).repeat([1, hei, wid])

    mpi_flat = mpi.permute(0, 2, 3, 1).reshape([-1, cnl])
    coords_floor_flat = coords_floor.reshape([-1, 2])
    idx_batch_offset_flat = idx_batch_offset.reshape([-1])

    idxs_i = coords_floor_flat[:, 0]
    idxs_j = coords_floor_flat[:, 1]
    idxs = idx_batch_offset_flat + idxs_j * wid + idxs_i

    mask = torch.bitwise_and(
        torch.bitwise_and(idxs_i >= 0, idxs_i < wid),
        torch.bitwise_and(idxs_j >= 0, idxs_j < hei)).reshape(-1)
    idxs = idxs[mask]
    mpi = mpi_flat[mask]

    # weights = mpi[:, -1]
    # mpi = mpi * weights.unsqueeze(-1)

    weights = torch.ones_like(mpi[:, -1])

    newmpi = torch.zeros(batchsz * hei * wid, cnl).to(mpi.device)
    denorm = torch.zeros(batchsz * hei * wid).to(mpi.device)

    newmpi = newmpi.scatter_add_(0, idxs.unsqueeze(-1).expand(-1, cnl), mpi)
    denorm = denorm.scatter_add_(0, idxs, weights).unsqueeze(-1) + 0.0001
    newmpi = (newmpi / denorm).reshape(batchsz, hei, wid, cnl).permute(0, 3, 1, 2)

    return newmpi.reshape(-1, layernum, cnl, hei, wid)


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
        im1_warpped = torchf.grid_sample(im1, flow, align_corners=True)
        return im1_warpped


def upsample_flow(content, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, C, H, W = content.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = torchf.unfold(8 * content, [3, 3], padding=1)
    up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, C, 8 * H, 8 * W)


FLOW_IDX, FLOWMASK_IDX, FLOWNET_IDX = 0, 1, 2


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


def flow_to_png_middlebury(flow: np.ndarray, maxflow=None):
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
    if maxflow is not None:
        maxrad = maxflow

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)
