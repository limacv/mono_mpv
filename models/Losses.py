import torch
import torch.nn as nn
import torch.nn.functional as torchf
from typing import List, Sequence, Union, Tuple, Dict
from ._util_modules import downsampleflow_as, remove_hsv_v

'''
implement all kinds of losses
'''
const_use_sobel = True
history_data_term_loss = {}  # store history [max, min]


class UnsuTriFrameLoss:
    """
    Unsupervised loss function for three image input
    """

    def __init__(self, weight_scheme: dict):
        super().__init__()
        # cfgs
        self.level_weights = weight_scheme.pop("level_weight", [1.])
        self.smooth_to_level = weight_scheme.pop("smooth_to_level", 0)
        self.flow_to_level = weight_scheme.pop("flow_to_level", 0)
        self.hsv_at_finest = weight_scheme.pop("hsv_at_finest", False)

        assert isinstance(self.level_weights, Sequence), "UnsuTriFrameLoss:: level_weight should be list if specified"
        self.offset = [None] * len(self.level_weights)  # used to transfer dx, dy to x, y Tensor: 1x2xHxW
        self.weights = [weight_scheme.copy() for _w in self.level_weights]
        for i, _ws in enumerate(self.weights):
            if i > self.smooth_to_level:
                _ws.pop("smooth_loss", None)
            if i > self.flow_to_level:
                _ws.pop("flow_loss", None)

    def __call__(self, im1s: Sequence[torch.Tensor], im2s: Sequence[torch.Tensor], im3s: Sequence[torch.Tensor],
                 f12s: Sequence[torch.Tensor], f23s: Sequence[torch.Tensor], f13s: Sequence[torch.Tensor],
                 o12s: Sequence[torch.Tensor], o23s: Sequence[torch.Tensor], o13s: Sequence[torch.Tensor]) \
            -> Tuple[torch.Tensor, Dict]:
        """
        imXs: image/feature pyramid, [0] is the finest and should be the original image
        fXYs: flows pyramid, [0] is the finest level
        oXY: occlusions from the finest level
        :return: loss, dict_of_loss (the former used for backwards, latter used for visualization)
        """
        assert len(f12s) == len(f23s) == len(f13s)
        assert len(im1s) == len(im2s) == len(im3s)
        if len(self.weights) > len(f12s):
            raise ValueError(f"UnsuTriFrameLoss:: too much level weights, the flow only has {len(f12s)} levels")

        def process_one_level(_weights: dict, level: int, _im1, _im2, _im3, _f12, _f23, _f13,
                              _o12=None, _o13=None, _o23=None):
            # preprocess some dataset
            # ----------------------
            # compute offset for the class
            batch_sz, _, hei, wid = _f12.shape
            if self.offset[level] is None or self.offset[level].shape[-3:] != _f12.shape[-3:]:
                offsety, offsetx = torch.meshgrid([
                    torch.linspace(0, hei - 1, hei),
                    torch.linspace(0, wid - 1, wid)
                ])
                self.offset[level] = torch.stack([offsetx, offsety], 0).unsqueeze(0).type(_f12.type()).cuda()

            def computegrid(_flow):
                _grid = (_flow + self.offset[level]).permute(0, 2, 3, 1)  # Bx2xHxW -> BxHxWx2
                _grid = _grid * torch.tensor((2. / (wid - 1), 2. / (hei - 1))).cuda() - 1
                return _grid

            # the sampling grid
            _f12_grid = computegrid(_f12)
            _f23_grid = computegrid(_f23)
            _f13_grid = computegrid(_f13)
            # the occlusion cache
            if _o12 is None:  # fake occlusion map
                _o12 = _o23 = _o13 = torch.tensor(0.).type_as(_f12)
                _no12 = _no13 = _no23 = torch.tensor(1.).type_as(_f12)
                _no12_sum = _no23_sum = _no13_sum = torch.tensor(_f12[:, 0:1, :, :].nelement()).type_as(_f12)
            else:
                _no12, _no13, _no23 = 1. - _o12, 1. - _o13, 1. - _o23
                _no12_sum, _no23_sum, _no13_sum = torch.sum(_no12), torch.sum(_no23), torch.sum(_no13)
            # prewarp images
            _im2_warpto1 = torchf.grid_sample(_im2, _f12_grid, 'bilinear', padding_mode='border')
            _im3_warpto1 = torchf.grid_sample(_im3, _f13_grid, 'bilinear', padding_mode='border')
            _im3_warpto2 = torchf.grid_sample(_im3, _f23_grid, 'bilinear', padding_mode='border')

            # start to compute all the loss
            # -----------------------------
            _final_loss, _loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}

            if "census_loss" in _weights.keys() and not _weights["census_loss"] == 0.:
                _f12_loss = ternary_loss(_im1, _im2_warpto1) * _no12
                _f23_loss = ternary_loss(_im2, _im3_warpto2) * _no23
                _f13_loss = ternary_loss(_im1, _im3_warpto1) * _no13
                _census_loss = torch.sum(_f12_loss) / _no12_sum \
                              + torch.sum(_f23_loss) / _no23_sum \
                              + torch.sum(_f13_loss) / _no13_sum
                _census_loss /= 3.
                _final_loss += _census_loss * _weights["census_loss"]
                _loss_dict["census_loss"] = float(_census_loss)

            if "photo_loss" in _weights.keys() and not _weights["photo_loss"] == 0.:
                _f12_loss = photometric_loss(_im1, _im2_warpto1) * _no12
                _f23_loss = photometric_loss(_im2, _im3_warpto2) * _no23
                _f13_loss = photometric_loss(_im1, _im3_warpto1) * _no13
                _photo_loss = torch.sum(_f12_loss) / _no12_sum \
                              + torch.sum(_f23_loss) / _no23_sum \
                              + torch.sum(_f13_loss) / _no13_sum
                _photo_loss /= 3.
                _final_loss += _photo_loss * _weights["photo_loss"]
                _loss_dict["photo_loss"] = float(_photo_loss)

            if "ssim_loss" in _weights.keys() and not _weights["ssim_loss"] == 0.:
                _f12_loss = ssim_loss(_im1, _im2_warpto1) * _no12
                _f23_loss = ssim_loss(_im2, _im3_warpto2) * _no23
                _f13_loss = ssim_loss(_im1, _im3_warpto1) * _no13
                _ssim_loss = torch.sum(_f12_loss) / _no12_sum \
                              + torch.sum(_f23_loss) / _no23_sum \
                              + torch.sum(_f13_loss) / _no13_sum
                _ssim_loss /= 3.
                _final_loss += _ssim_loss * _weights["ssim_loss"]
                _loss_dict["ssim_loss"] = float(_ssim_loss)

            if "smooth_loss" in _weights.keys() and not _weights["smooth_loss"] == 0.:
                smooth_order = _weights["smooth_order"] if "smooth_order" in _weights.keys() else 1
                smooth_normalize = _weights["smooth_normalize"] \
                    if "smooth_normalize" in _weights.keys() else 'x_dx'
                _smt1_loss = smooth_grad(_f12, _im1, order=smooth_order, normalize=smooth_normalize).mean()
                _smt2_loss = smooth_grad(_f23, _im2, order=smooth_order, normalize=smooth_normalize).mean()
                _smt3_loss = smooth_grad(_f13, _im1, order=smooth_order, normalize=smooth_normalize).mean()
                _smth_loss = (_smt1_loss + _smt2_loss + _smt3_loss) / 3.
                _final_loss += _smth_loss * _weights["smooth_loss"]
                _loss_dict["smooth_loss"] = float(_smth_loss)

            if "occ_loss" in _weights.keys() and not _weights["smooth_loss"] == 0.:
                # `O1(x) * || O2(x) – O3(x) ||
                _occ_loss = _no12 * - torch.log2(_o13 * _o23 + _no13 * _no23)
                _occ_loss = torch.sum(_occ_loss) / _no12_sum

                _final_loss += _occ_loss * _weights["occ_loss"]
                _loss_dict["occ_loss"] = float(_occ_loss)

            if "flow_loss" in _weights.keys() and not _weights["flow_loss"] == 0.:
                # get the f23 in frame1 space
                # for flow outof border, the occlusion detection step should detect it as occluded,
                # so the padding mode isn't really matter
                _f23_in1 = torchf.grid_sample(_f23, _f12_grid, 'bilinear', padding_mode='border')

                _f123 = _f12 + _f23_in1
                _no123 = _no12 * _no13
                # no23_in1 = torchf.grid_sample(no23, f12_grid, 'bilinear', padding_mode='border')
                # no123_in1 = no12 * no23_in1
                # ----------------------------
                # check which one is better
                with torch.no_grad():
                    _f123_grid = computegrid(_f123.detach())
                    _im3_warpto2to1 = torchf.grid_sample(_im3, _f123_grid, 'bilinear', padding_mode='border')
                    diff_f13 = photometric_loss(_im1, _im3_warpto1.detach())
                    diff_f123 = photometric_loss(_im1, _im3_warpto2to1.detach())
                    mask_f13_good = (diff_f13 < diff_f123).type_as(_f12)
                    mask_f123_good = - mask_f13_good + 1.

                _flow_loss = torch.norm(_f123.detach() - _f13, dim=1, keepdim=True) * mask_f123_good \
                             + torch.norm(_f123 - _f13.detach(), dim=1, keepdim=True) * mask_f13_good
                _flow_loss *= _no123
                # this is for compansate when no12 and no13 == None
                _no123_sum = torch.sum(_no123) if _no123.dim() > 0 else _no12_sum
                # ----------------------------
                # ||F1(x) + F2(F(x)+x) – F3(x)|| * `O1(x) * `O2(x)
                # flow_loss = torch.norm(f12 + f123 - f13, dim=1, keepdim=True) * no123
                _flow_loss = torch.sum(_flow_loss) / _no123_sum

                # adjust the flow loss based on the photo loss
                # dataterm_key = "photo_loss" if "photo_loss" in loss_dict.keys() else "census_loss"
                # if dataterm_key not in loss_dict.keys():
                #     raise RuntimeError("UnsuTriFrameLoss:: Should specify at least on dataterm loss")

                # flow_loss_activation = 1.
                # global history_data_term_loss  # a dict store history info
                # if not history_data_term_loss:  # initialization
                #     history_data_term_loss['smth'] = 0
                #     history_data_term_loss['min'] = history_data_term_loss['max'] = loss_dict[dataterm_key]
                # else:
                #     history_data_term_loss['smth'] = 0.5*history_data_term_loss['smth']+0.5*loss_dict[dataterm_key]
                #     history_data_term_loss['max'] = max(history_data_term_loss['max'], history_data_term_loss['smth'])
                #     history_data_term_loss['min'] = min(history_data_term_loss['min'], history_data_term_loss['smth'])
                #     min_thr, max_thr = history_data_term_loss['min'], history_data_term_loss['max']
                #     cur_val = history_data_term_loss['smth']
                #     flow_loss_activation = (max_thr - cur_val) / (max_thr - min_thr + 0.01)

                # To prevent the naive solution that flow==0
                _final_loss += _flow_loss * _weights["flow_loss"]  # * flow_loss_activation
                _loss_dict["flow_loss"] = float(_flow_loss)
                # loss_dict["flow_activate"] = flow_loss_activation

            return _final_loss, _loss_dict

        # Call process_one_level
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        for level, weights in enumerate(self.weights):

            f12, f23, f13 = f12s[level], f23s[level], f13s[level]
            try:
                img_idx = [im.shape[-2:] for im in im1s].index(f12.shape[-2:])
            except ValueError:
                raise ValueError(f"UnsuTriFrameLoss::Warning:: flow of shape:{f12.shape} "
                                 f"doesn't have correspond feature pyramid")
            if level < len(o12s):
                o12, o23, o13 = o12s[level], o23s[level], o13s[level]
            else:
                o12, o23, o13 = None, None, None
            # preprocess some dataset
            im1, im2, im3 = im1s[img_idx].detach(), im2s[img_idx].detach(), im3s[img_idx].detach()
            if self.hsv_at_finest and im1.shape[1] == 3:
                im1 = remove_hsv_v(im1, srgb=True)
                im2 = remove_hsv_v(im2, srgb=True)
                im3 = remove_hsv_v(im3, srgb=True)
            # kick off loss computation
            final_loss_level, loss_dict_level = process_one_level(weights, level, im1, im2, im3,
                                                                  f12, f23, f13, o12, o13, o23)
            final_loss += final_loss_level * self.level_weights[level]

            # test, please delete afterwards
            # loss_str = ", ".join([f"{k}:{v:.3f}" for k, v in loss_dict_level.items()])
            # print(f"level {level}: {loss_str}")

            if level == 0:  # only show the first level loss since it's most intuitive
                loss_dict.update(loss_dict_level)
        return final_loss, loss_dict


class SuBiFrameLoss:
    """
    Unsupervised loss function for three image input
    """

    def __init__(self, weight_scheme: dict):
        super().__init__()
        self.weights = weight_scheme

    def __call__(self, estimated_flow01: Sequence[torch.Tensor], estimated_flow10: Sequence[torch.Tensor],
                 gt_flow01, gt_flow10,
                 estimated_occ=None, gt_occ01=None, gt_occ10=None) -> Tuple[torch.Tensor, Dict]:
        """
        estimated_flow: flow from different level, where [0] is the finest resolution,

        :return: loss, dict_of_loss
        """
        assert len(estimated_flow01) == len(estimated_flow10)
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        if "su_epe_loss" in self.weights.keys():
            # check level_weight availablity
            if "level_weight" not in self.weights:
                self.weights["level_weight"] = [1.] + [0.] * (len(estimated_flow01) - 1)
                print(f"Warning::SuBiFrameLoss:: level_weight not specified, "
                      f"use default weights: {self.weights['level_weight']}")
            elif len(self.weights["level_weight"]) != len(estimated_flow01):
                raise ValueError(f"SuBiFrameLoss:: cfg['level_weight']: {self.weights['level_weight']} "
                                 f"is incompatiable with output level number {len(estimated_flow01)}")

            epe_loss = torch.tensor(0., dtype=torch.float32).cuda()
            for level, (e_flow01, e_flow10, level_weight) in \
                    enumerate(zip(estimated_flow01, estimated_flow10, self.weights["level_weight"])):
                gt_flow01_level = downsampleflow_as(e_flow01, gt_flow01)
                gt_flow10_level = downsampleflow_as(e_flow10, gt_flow10)
                epe_loss01_level = torch.norm(gt_flow01_level - e_flow01, dim=1).mean()
                epe_loss10_level = torch.norm(gt_flow10_level - e_flow10, dim=1).mean()
                epe_loss_level = (epe_loss01_level + epe_loss10_level) / 2.
                epe_loss += epe_loss_level * level_weight
                if level == 0:
                    loss_dict["epe_loss_level0"] = epe_loss_level.detach()

            final_loss += epe_loss
            loss_dict["epe_loss_weighted"] = epe_loss.detach()

        # TODO: implement occ loss if necessary
        if "su_occ_loss" in self.weights.keys():
            raise NotImplementedError

        return final_loss, loss_dict


def photometric_loss(im, im_warp, order=1):
    # scale = 3. / im.shape[1]  # normalize the loss to as if there were three channels
    diff = torch.abs(im - im_warp) + 0.0001  # why?
    diff = torch.sum(diff, dim=1, keepdim=True) / 3.
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
    # torch.tensor((
    #     (0.015026, 0.028569, 0.035391, 0.028569, 0.015026),
    #     (0.028569, 0.054318, 0.067288, 0.054318, 0.028569),
    #     (0.035391, 0.067288, 0.083355, 0.067288, 0.035391),
    #     (0.028569, 0.054318, 0.067288, 0.054318, 0.028569),
    #     (0.015026, 0.028569, 0.035391, 0.028569, 0.015026)
    # )).reshape((1, 1, 5, 5)).type(torch.float32)


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


def smooth_grad(flo, image, alpha=50, order=1, normalize='None'):
    """
    :param flo:
    :param image:
    :param alpha: typically alpha = 10 ~ 50
    :param normalize: 'dx' 'x' 'None'
    :return:
    """
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), dim=1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), dim=1, keepdim=True) * alpha)

    dx, dy = gradient(flo, order=order)
    if normalize == 'dx_min':  # normalize on dx, prevent large dx
        dx = torch.min(dx.abs(), torch.tensor(20.).type_as(dx))
        dy = torch.min(dy.abs(), torch.tensor(20.).type_as(dy))
    elif normalize == "dx_sqrt":
        dx = torch.pow(dx.abs() / 20., 0.5) * 20.
        dy = torch.pow(dy.abs() / 20., 0.5) * 20.
    elif normalize == 'dx_softmin':
        dx, dy = dx ** 2, dy ** 2
        dx = dx / (dx + 66.7) * 20.
        dy = dy / (dy + 66.7) * 20.
    elif normalize == 'x' or normalize == 'x_dx':
        with torch.no_grad():
            smt_kernel = torch.tensor(((1. / 9., 1. / 9., 1. / 9.),
                                       (1. / 9., 1. / 9., 1. / 9.),
                                       (1. / 9., 1. / 9., 1. / 9.))).reshape((1, 1, 3, 3)).type_as(dx)
            flow_mag = torch.norm(flo, dim=1, keepdim=True)
            flow_mag = torchf.conv2d(torchf.pad(flow_mag, [1, 1, 1, 1], 'replicate'), smt_kernel)
            denominator = torch.sqrt((flow_mag / 10.) ** 2 + 1.).detach()
        dx = dx.abs() / denominator
        dy = dy.abs() / denominator
        if normalize == 'x_dx':
            dx, dy = dx ** 2, dy ** 2
            dx = dx / (dx + 66.7) * 20.
            dy = dy / (dy + 66.7) * 20.

    elif normalize == 'None':
        dx = dx.abs()
        dy = dy.abs()
    else:
        raise ValueError(f"Losses::smooth_grad: unrecognized token: {normalize}")

    loss_x = weights_x * dx
    loss_y = weights_y * dy
    return torch.sum(loss_x + loss_y, dim=1, keepdim=True) / 4.  # 2 for du, dv, 2 for loss_x, loss_y


# test loss
if __name__ == "__main__":
    import sys

    sys.path.append('../')
    from dataset.sintel_seq import SintelSeq

    dataset = SintelSeq("D:\\MSI_NB\\source\\dataset\\Sintel\\" + "training\\final\\temple_2")
    idx = 26
    lossfn = UnsuTriFrameLoss({
        "flow_loss": 1,
        "occ_loss": 2,
        "photo_loss": 1,
        "smooth_loss": 1,
        "ssim_loss": 1,
        "smooth_order": 1,
        "smooth_normalize": 'x_dx'
    })
    # lossfn = SuBiFrameLoss({
    #     "su_epe_loss": 1,
    #     "level_weight": [1, 2]
    # })
    frame1, frame2, frame3 = dataset.get_tensorframe(idx), dataset.get_tensorframe(idx + 1), dataset.get_tensorframe(
        idx + 0)
    frame1 = frame1.unsqueeze(0).cuda()
    frame2 = frame2.unsqueeze(0).cuda()
    frame3 = frame3.unsqueeze(0).cuda()
    flow1, flow2, flow3 = dataset.get_gtflow(idx), dataset.get_gtflow(idx + 1), dataset.get_gtflow(idx)
    flow1 = torch.tensor(flow1).permute(2, 0, 1).unsqueeze(0).cuda()
    flow2 = torch.tensor(flow2).permute(2, 0, 1).unsqueeze(0).cuda()
    flow3 = torch.tensor(flow3).permute(2, 0, 1).unsqueeze(0).cuda()
    occ1, occ2, occ3 = dataset.get_gtocc(idx), dataset.get_gtocc(idx + 1), dataset.get_gtocc(idx)
    occ1 = torch.tensor(occ1).unsqueeze(0).unsqueeze(0).cuda()
    occ2 = torch.tensor(occ2).unsqueeze(0).unsqueeze(0).cuda()
    occ3 = torch.tensor(occ3).unsqueeze(0).unsqueeze(0).cuda()
    loss, loss_dict = lossfn([frame1], [frame2], [frame3], [flow1], [flow2], [flow3], [occ1], [occ2], [occ3])
    # loss, loss_dict = lossfn(frame1, frame2, frame3, flow1, flow2, flow3, occ1, occ2, occ3)
    print(loss)
    print(loss_dict)
