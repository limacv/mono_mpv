import numpy as np
import torch


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


def estimate_disparity_torch(mpi: torch.Tensor, min_depth=1, max_depth=100):
    """Compute disparity map from a set of MPI layers.
    mpi: tensor of shape B x LayerNum x 4 x H x W
    return: tensor of shape B x H x W
    """
    assert(mpi.dim() == 5)
    batchsz, num_plane, _, height, width = mpi.shape
    disparities = torch.linspace(1. / max_depth, 1. / min_depth, num_plane)
    disparities = disparities.reshape(1, -1, 1, 1).type_as(mpi)

    alpha = mpi[:, :, -1, ...]  # alpha.shape == B x LayerNum x H x W
    alpha = alpha * torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                               torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    disparity = (alpha * disparities).sum(dim=1)

    return disparity
