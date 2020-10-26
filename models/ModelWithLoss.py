import torch
import torch.nn as nn
import torch.nn.functional as torchf
from typing import List, Callable
import numpy as np
import sys
from .Losses import *
sys.path.append("../")
from torchvision.transforms import ToTensor
import cv2
from ._util_modules import initialize_msra, OccEstimator

'''
This class is a model container provide interface between
                                            1. dataset <-> model
                                            2. model <-> loss
'''
visualize_epemap_max_epe = 5


class ModelandLossBase:
    def __init__(self, model: nn.Module, occ_cfg=None):
        """
        the model should satisfy:
            1. take two Bx3xHxW Tensors as input
            2. output:
                2.1. when training: return flowlist and occmaplist, where [0] is the finest resolution
                2.2. when evaluating: flow, occ_map, with flow.shape=Bx2xHxW and occ_map.shape=Bx1xHxW, occ_map = None
                     if the model don't estimate occ_map
            3. the occ_map should between (0, 1), while flow is the displacement in image space
            4.[optional] if the model implement encode_only(), then the input should take feature and image as input
        """
        torch.set_default_tensor_type(torch.FloatTensor)
        self.model = model
        self.model.train()
        self.model.cuda()
        self.valid_visualize = True

        if occ_cfg is None:
            occ_cfg = dict()
        occ_threshold = occ_cfg["occ_thresh"] if "occ_thresh" in occ_cfg.keys() else 3.5
        self.occ_estimator = OccEstimator(occ_threshold)

    def __call__(self, *args):  # convenient call train_forward
        args = [item.cuda() for item in args]
        return self.train_forward(*args)

    def train_forward(self, *args) -> Tuple[torch.Tensor, Dict]:
        """
        Abstract call for training forward process
        return the final loss to be backward() and loss dict contains all the loss term
        """
        raise NotImplementedError

    def valid_forward(self, frame0s: torch.Tensor, frame1s: torch.Tensor,
                      gtflows: torch.Tensor, gtoccs: torch.Tensor,
                      output_res=True) -> Tuple[Dict[str, float], Union[torch.Tensor, None]]:
        """
        valid using ground truth, using EPE metric, also optionally output display for first flow
        :param frame0s and frame1s: Tensor of Bx3xHxW ~(0, 1)
        :param gtflows: gt flows: Tensor of Bx2xHxW or 2xHxW or HxWx2 or BxHxWx2
        :param gtoccs: gt occs: Tensor of Bx1xHxW or BxHxW
        :return: EPE from GT
        """
        def checkdim(ten: torch.Tensor):
            if ten.dim() == 3:
                ten = ten.unsqueeze(0)
            elif ten.dim() == 2:
                ten = ten.unsqueeze(0).unsqueeze(0)
            elif ten.dim() != 4:
                raise ValueError(f"ModelBase.valid_forward::input shape {ten.shape} wrong")
            return ten

        def checkshape(ten: torch.Tensor, desire_cnl=3):  # input should be dim=4
            if ten.shape[-1] == desire_cnl:
                return ten.permute(0, 3, 1, 2)
            else:
                return ten

        frame0s = checkshape(checkdim(frame0s))
        frame1s = checkshape(checkdim(frame1s))
        gtflows = checkshape(checkdim(gtflows), desire_cnl=2)
        gtoccs = checkshape(checkdim(gtoccs), desire_cnl=1)

        gtoccs = gtoccs.type(torch.uint8)

        gtnoccs = torch.bitwise_xor(gtoccs, 0x01)
        # now all input should be in form of B x C x H x W
        if not (frame0s.shape[0] == frame1s.shape[0] == gtflows.shape[0] == gtoccs.shape[0]):
            raise RuntimeError(f"ModelBase.valid_forward::input batch size not matched")

        frame0s, frame1s = frame0s.cuda(), frame1s.cuda()
        with torch.no_grad():
            eflows, eoccs = self.infer_forward(frame0s, frame1s, occ_out=True)

        eflows = eflows.cpu()
        eoccs = (eoccs > 0.5).type(torch.uint8).cpu()
        val_dict = {}
        # compute error and record
        epe_all = torch.norm(eflows - gtflows, dim=1, keepdim=True)
        epe_wocc = epe_all * gtnoccs
        val_dict["val_EPE_all"] = float(torch.mean(epe_all))
        val_dict["val_EPE_nocc"] = float(torch.sum(epe_wocc)) / float(torch.sum(gtnoccs))

        occ_gt_or_e = torch.bitwise_or(eoccs, gtoccs)
        occ_gt_and_e = torch.bitwise_and(eoccs, gtoccs)
        val_dict["val_OCCiou_i"] = float(torch.sum(occ_gt_and_e)) / 1000.  # 1000 means nothing, just to make
        val_dict["val_OCCiou_u"] = float(torch.sum(occ_gt_or_e)) / 1000.  # it more readable

        # output the first flow result if desired
        if output_res:
            display_flow, display_occ = flow_to_png_middlebury(eflows[0].numpy()), eoccs[0, 0].numpy()
            epe_map = epe_all[0, 0].numpy()
            epe_map = np.minimum(epe_map, visualize_epemap_max_epe)
            epe_map = (epe_map * (255 / visualize_epemap_max_epe)).astype(np.uint8)
            epe_map = cv2.applyColorMap(epe_map, cv2.COLORMAP_JET)
            epe_map = cv2.cvtColor(epe_map, cv2.COLOR_BGR2RGB)

            display_occ = cv2.cvtColor(display_occ * 255, cv2.COLOR_GRAY2RGB)
            display_image = ToTensor()(np.vstack([display_flow, epe_map, display_occ]))
        else:
            display_image = None

        return val_dict, display_image

    def infer_forward(self, im1, im2, occ_out=False, cuda=True):
        """
        :param occ_out: if occ_out is False and the model don't estimate occ, return flow, None
                        else: occ will be estimated by two pass
        """
        assert im1.shape == im2.shape
        if cuda:
            im1, im2 = im1.cuda(), im2.cuda()
        if im1.dim() == 3:
            im1, im2 = im1.unsqueeze(0), im2.unsqueeze(0)
        if im1.shape[-1] == 3:
            im1, im2 = im1.permute(0, 3, 1, 2), im2.permute(0, 3, 1, 2)

        self.model.eval()
        with torch.no_grad():
            flow, occ = self.model(im1, im2)
            if occ is None and occ_out:
                flowb, _ = self.model(im2, im1)
                occ = self.occ_estimator.estimate(flowb)

        self.model.train()
        return flow, occ

    def initial_weights(self):
        initialize_msra(self.model)


class ModelandUnsuLoss(ModelandLossBase):
    def __init__(self, model: nn.Module, loss_cfg: dict, occ_cfg: dict):
        """
        the model should satisfy:
            1. take two Bx3xHxW Tensors as input
            2. output:
                2.1. when training: return flowlist and occmaplist, where [0] is the finest resolution
                2.2. when evaluating: flow, occ_map, with flow.shape=Bx2xHxW and occ_map.shape=Bx1xHxW, occ_map = None
                     if the model don't estimate occ_map
            3. the occ_map should between (0, 1), while flow is the displacement in image space
            4.[optional] if the model implement encode_only(), then the input should take feature and image as input
        """
        super().__init__(model, occ_cfg)
        self.loss = UnsuTriFrameLoss(loss_cfg)
        # Configuration
        self.stop_grad_at_occ = occ_cfg["stop_grad_at_occ"] if "stop_grad_at_occ" in occ_cfg.keys() else True
        self.soft_occ_map = occ_cfg["soft_occ_map"] if "soft_occ_map" in occ_cfg.keys() else False
        self.estimate_occ = occ_cfg["estimate_occ"] if "estimate_occ" in occ_cfg.keys() else True
        self.occ_down_level = loss_cfg["occ_down_level"] if "occ_down_level" in loss_cfg.keys() else 0

    def train_forward(self, im1: torch.Tensor, im2: torch.Tensor, im3: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        forward the model and compute loss
        :return the final loss
        this implementation implicitly batch size * 2
        """
        assert im1.shape == im2.shape == im3.shape and len(im1.shape) == 4
        if hasattr(self.model, "encode_only"):
            feat1 = self.model.encode_only(im1)
            feat2 = self.model.encode_only(im2)
            feat3 = self.model.encode_only(im3)
            f12, o12 = self.model(feat1, feat2)
            f23, o23 = self.model(feat2, feat3)
            f13, o13 = self.model(feat1, feat3)
            with torch.no_grad():
                f21, o21 = self.model(feat2, feat1)
                f32, o32 = self.model(feat3, feat2)
                f31, o31 = self.model(feat3, feat1)
        else:
            raise NotImplementedError("the model should have encode_only function to produce feature encoding")
            # f12, o12 = self.model(im1, im2)
            # f23, o23 = self.model(im2, im3)
            # f13, o13 = self.model(im1, im3)
            #
            # f21, o21 = self.model(im2, im1)
            # f32, o32 = self.model(im3, im2)
            # f31, o31 = self.model(im3, im1)

        if isinstance(f12, torch.Tensor):
            f12, f23, f13 = [f12], [f23], [f13]
            f21, f32, f31 = [f21], [f32], [f31]

        if not self.estimate_occ:
            o12 = o23 = o13 = o21 = o32 = o31 = torch.zeros_like(f12[0][:, 0:1, :, :])
        elif o12 is None:  # and self.estimate_occ
            def nogradifneed(_f):
                return _f[0].detach() if self.stop_grad_at_occ else _f[0]
            o12 = self.occ_estimator.estimate(nogradifneed(f21))
            o23 = self.occ_estimator.estimate(nogradifneed(f32))
            o13 = self.occ_estimator.estimate(nogradifneed(f31))

            # o21 = self.occ_estimator.estimate(nogradifneed(f12))
            # o32 = self.occ_estimator.estimate(nogradifneed(f23))
            # o31 = self.occ_estimator.estimate(nogradifneed(f13))

        if not self.soft_occ_map:
            o12 = (o12 > 0.5).type_as(f12[0]).detach()
            o23 = (o23 > 0.5).type_as(f12[0]).detach()
            o13 = (o13 > 0.5).type_as(f12[0]).detach()
            # o21 = (o21 > 0.5).type_as(f12[0]).detach()
            # o32 = (o32 > 0.5).type_as(f12[0]).detach()
            # o31 = (o31 > 0.5).type_as(f12[0]).detach()

        # make occlusion a pyramid
        def make_pyramid(_occ):
            if not isinstance(_occ, Sequence):
                _pyramid = [_occ]
                for _f in f12[1:]:
                    if len(_pyramid) > self.occ_down_level:
                        break
                    _pyramid.append(torchf.adaptive_avg_pool2d(_pyramid[-1], [_f.shape[-2], _f.shape[-1]]))
                return _pyramid

        assert 0 <= self.occ_down_level < len(f12), \
            f"occ_down_level: {self.occ_down_level} exceed the flow_level: {len(f12)}"
        o12 = make_pyramid(o12)
        o23 = make_pyramid(o23)
        o13 = make_pyramid(o13)
        # o21 = make_pyramid(o21)
        # o32 = make_pyramid(o32)
        # o31 = make_pyramid(o31)
        # -----------------------------
        # compute loss
        # -----------------------------
        loss1, loss_dict1 = self.loss(feat1[::-1], feat2[::-1], feat3[::-1], f12, f23, f13, o12, o23, o13)
        return loss1, loss_dict1
        # loss2, loss_dict2 = self.loss(feat3[::-1], feat2[::-1], feat1[::-1], f32, f21, f31, o32, o21, o31)
        # return (loss1 + loss2) / 2., {k: (loss_dict1[k] + loss_dict2[k]) / 2. for k in loss_dict1.keys()}


class ModelandSuLoss(ModelandLossBase):
    def __init__(self, model: nn.Module, loss_cfg: dict):
        super(ModelandSuLoss, self).__init__(model)
        self.loss = SuBiFrameLoss(loss_cfg)

    def train_forward(self, im1: torch.Tensor, im2: torch.Tensor,
                      gtflow12: torch.Tensor, gtflow21: torch.Tensor,
                      gtocc12: torch.Tensor, gtocc21: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if hasattr(self.model, "encode_only"):
            feat1 = self.model.encode_only(im1)
            feat2 = self.model.encode_only(im2)
            f12, o12 = self.model(feat1, feat2)
            f21, o21 = self.model(feat2, feat1)
        else:
            f12, o12 = self.model(im1, im2)
            f21, o21 = self.model(im2, im1)

        if isinstance(f12, torch.Tensor):
            f12, f21 = [f12], [f21]

        return self.loss(f12, f21, gtflow12, gtflow21, o12, gtocc12, gtocc21)
