import numpy
import torch
from models.mpi_utils import render_newview, make_depths, matplot_img
from models.flow_utils import forward_scatter

hei, wid = 360, 640

source_pose = torch.tensor(
    [[1.0, 0.0, 0.0, 0],
     [0.0, 1.0, 0.0, 0],
     [0.0, 0.0, 1.0, 0]]
).unsqueeze(0)
intrin = torch.tensor(
    [[wid / 2, 0.0, wid / 2],
     [0.0, hei / 2, hei / 2],
     [0.0, 0.0, 1.0]]
).unsqueeze(0)

target_pose1 = torch.tensor(
    [[[0.9844, 0.0060, -0.1760, 0.1770],
      [0.0000, 0.9994, 0.0340, -0.0340],
      [0.1761, -0.0335, 0.9838, 0.1600]]]
)

target_pose2 = torch.tensor(
    [[[0.9946, 0.0010, 0.1038, -0.1040],
      [0.0000, 0.9999, -0.0100, 0.0100],
      [-0.1038, 0.0099, 0.9945, -0.0800]]]
)

target_pose3 = torch.tensor(
    [[[0.9835, -0.0036, -0.1810, 0.1820],
      [0.0000, 0.9998, -0.0200, 0.0200],
      [0.1810, 0.0197, 0.9833, 0.0000]]]
)

target_pose4 = torch.tensor(
    [[[0.9980, -0.0062, -0.0627, 0.0630],
      [0.0000, 0.9952, -0.0978, 0.0980],
      [0.0630, 0.0976, 0.9932, 0.0000]]]
)


def renderto(mpi, tarpose, focal=1):
    mpi = mpi.cpu()
    depthes = make_depths(32).type_as(mpi)
    tarintrin = intrin.clone()
    tarintrin[0, 0, 0] *= focal
    tarintrin[0, 1, 1] *= focal
    view = render_newview(mpi, source_pose, tarpose, intrin, intrin, depthes)
    return view


def disoccmask(disparity, tarpose, focal=1.):
    if disparity.dim() == 4:
        disparity = disparity[0, 0]
    elif disparity.dim() == 3:
        disparity = disparity[0]

    hei, wid = disparity.shape
    y_coord, x_coord = torch.meshgrid([torch.linspace(-1, 1, hei), torch.linspace(1, 1, wid)])
    ones = torch.ones_like(x_coord)
    coords = torch.stack([x_coord, y_coord, -torch.reciprocal(disparity), ones])

    bottom = torch.tensor([0, 0, 0, 1.]).type_as(tarpose).reshape(1, 1, 4)
    tarpose = torch.cat([tarpose, bottom], dim=1)
    srcpose = torch.cat([source_pose, bottom], dim=1)
    srcintrin = torch.eye(4).type_as(tarpose).reshape(1, 4, 4) * 0.5
    srcintrin[0, 0, 2] = 0.5
    srcintrin[0, 1, 2] = 0.5
    srcintrin[0, 2, 2] = 1
    srcintrin[0, 3, 3] = 1

    tarintrin = srcintrin.clone()
    tarintrin[0, 0, 0] *= focal
    tarintrin[0, 1, 1] *= focal

    coordsnew = tarintrin @ tarpose @ srcpose @ srcintrin @ coords.reshape(4, -1)
    coordsnew = coordsnew.reshape(4, hei, wid)[:3]
    coordsnew = (coordsnew / coordsnew[2:3])[:2]
    flow = (coordsnew - coords[:2]).unsqueeze(0)
    flow = torch.tensor([wid, hei]).type_as(coordsnew).reshape(2, 1, 1) * (flow + 1) / 2
    mask = forward_scatter(flow, ones.unsqueeze(0).unsqueeze(0))
    return mask


if __name__ == "__main__":
    disparity = torch.ones(1, 40, 50).type(torch.float)
    mask = disoccmask(disparity, target_pose1, 1.16)
