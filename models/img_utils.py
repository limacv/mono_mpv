from skimage import metrics
import torch
from .lpips.lpips import LPIPS

photometric = {
    "mse": metrics.mean_squared_error,
    "ssim": metrics.structural_similarity,
    "psnr": metrics.peak_signal_noise_ratio,
    "lpips": LPIPS()
}


def compute_img_metric(im1t: torch.Tensor, im2t: torch.Tensor,
                       metric="mse", margin=0.05):
    if metric not in photometric.keys():
        raise RuntimeError(f"img_utils:: metric {metric} not recognized")

    if im1t.dim() == 3:
        im1t = im1t.unsqueeze(0)
        im2t = im2t.unsqueeze(0)
    im1t = im1t.detach().cpu()
    im2t = im2t.detach().cpu()
    im1 = im1t.permute(0, 2, 3, 1).numpy()
    im2 = im2t.permute(0, 2, 3, 1).numpy()
    batchsz, hei, wid, _ = im1.shape
    if margin > 0:
        marginh = int(hei * margin) + 1
        marginw = int(wid * margin) + 1
        im1 = im1[:, marginh:hei-marginh, marginw:wid-marginw]
        im2 = im2[:, marginh:hei-marginh, marginw:wid-marginw]
    values = []
    for i in range(batchsz):
        if metric in ["mse", "psnr"]:
            value = photometric[metric](
                im1[i], im2[i]
            )
        elif metric in ["ssim"]:
            value = photometric[metric](
                im1[i], im2[i],
                multichannel=True
            )
        elif metric in ["lpips"]:
            value = photometric[metric](
                im1t[i:i+1], im2t[i:i+1]
            )
        else:
            raise NotImplementedError
        values.append(value)

    return sum(values) / len(values)
