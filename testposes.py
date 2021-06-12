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

target_posemmp = torch.tensor(
    [[[0.9919, -0.0054, -0.1265, 0.1270],
      [0.0000, 0.9991, -0.0430, 0.0430],
      [0.1267, 0.0426, 0.9910, 0.0400]]]
)

target_pose5 = torch.tensor(
    [[[0.9944, 0.0107, 0.1053, -0.1060],
      [0.0000, 0.9949, -0.1008, 0.1010],
      [-0.1058, 0.1003, 0.9893, 0.0000]]]
)

target_pose_blackswan = torch.tensor(
    [[[0.9989, 0.0058, 0.0456, -0.0460],
      [0.0000, 0.9921, -0.1257, 0.1260],
      [-0.0460, 0.1255, 0.9910, 0.0000]]]
)

target_pose_bmx_bumps = torch.tensor(
    [[[0.9958, 0.0028, 0.0918, -0.0920],
      [0.0000, 0.9996, -0.0300, 0.0300],
      [-0.0919, 0.0299, 0.9953, 0.0000]]]
)

target_pose_camel = torch.tensor(
    [[[0.9942, -0.0018, -0.1078, 0.1080],
      [0.0000, 0.9999, -0.0170, 0.0170],
      [0.1078, 0.0169, 0.9940, 0.0000]]]
)

target_pose_carroundabout = torch.tensor(
    [[[0.9930, 0.0081, 0.1174, -0.1180],
      [0.0000, 0.9976, -0.0689, 0.0690],
      [-0.1177, 0.0685, 0.9907, 0.0000]]]
)
target_pose_soapbox = torch.tensor(
    [[[0.9950, -0.0072, -0.0996, 0.1000],
      [0.0000, 0.9974, -0.0719, 0.0720],
      [0.0998, 0.0716, 0.9924, 0.0000]]]
)

target_posedx = lambda dx: torch.tensor(
    [[[1, 0, 0, -dx],
      [0, 1, 0, 0],
      [0, 0, 1, 0.]]]
).type(torch.float32)

target_pose_forvis1 = torch.tensor(
    [[[0.9852, -0.0034, 0.1711, -0.1720],
      [0.0000, 0.9998, 0.0200, -0.0200],
      [-0.1712, -0.0197, 0.9850, 0.0000]]]
)


def renderto(mpi, tarpose, focal=1, device='cpu'):
    mpi = mpi.to(device)
    tarpose = tarpose.to(device)
    depthes = make_depths(32).type_as(mpi)
    tarintrin = intrin.clone().type_as(mpi)
    tarintrin[0, 0, 0] *= focal
    tarintrin[0, 1, 1] *= focal
    view = render_newview(mpi, source_pose.clone().type_as(mpi), tarpose, tarintrin, tarintrin, depthes)
    return view


def estimate_dx(mpi, refim, tarim, flow_estim, flowcache):
    if refim.dim() == 3:
        refim = refim.unsequeeze(0)
    if tarim.dim() == 3:
        tarim = tarim.unsequeeze(0)
    flowgt = flowcache.estimate_flow(refim, tarim)


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
