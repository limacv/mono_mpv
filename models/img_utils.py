from skimage import metrics
import torch

photometric = {
    "mse": metrics.mean_squared_error,
    "ssim": metrics.structural_similarity,
    "psnr": metrics.peak_signal_noise_ratio
}


def compute_img_metric(im1: torch.Tensor, im2: torch.Tensor,
                       metric="mse", margin=0.05):
    if metric not in photometric.keys():
        raise RuntimeError(f"img_utils:: metric {metric} not recognized")

    if im1.dim() == 3:
        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
    im1 = im1.permute(0, 2, 3, 1).detach().cpu().numpy()
    im2 = im2.permute(0, 2, 3, 1).detach().cpu().numpy()
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
        else:
            raise NotImplementedError
        values.append(value)

    return sum(values) / len(values)
