import torch
import torch.nn as nn
import torch.nn.functional as torchf
from typing import List, Callable
import numpy as np
import sys
from .loss_utils import *
from torchvision.transforms import ToTensor, Resize
from torchvision.utils import make_grid
from .mpi_utils import *
from .geo_utils import RandomAffine
import cv2
from .mpi_network import *
from .mpv_network import *
from .mpifuse_network import *
from .mpi_flow_network import *
from .flow_utils import *
from .hourglass import Hourglass
from collections import deque
from .img_utils import *

'''
This class is a model container provide interface between
                                            1. dataset <-> model
                                            2. model <-> loss
'''

SCALE_EPS = 0
SCALE_SCALE = 1.03
Sigma_Denorm = 10


class ModelandSVLoss(nn.Module):
    """
    SV stand for single view
    """

    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        assert "loss_weights" in cfg.keys(), "ModelAndLossBase: didn't find 'loss_weights' in cfg"
        self.loss_weight = cfg["loss_weights"].copy()

        torch.set_default_tensor_type(torch.FloatTensor)

        self.model = model
        self.model.train()

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        self.scale_mode = self.loss_weight.pop("scale_mode", "norm")  # first / fix / adaptive

        self.scheduler = ParamScheduler([5e4, 10e4], [0.5, 1])
        self.smth_scheduler = ParamScheduler(
            milestones=[2e3, 4e3],
            values=[0, 1]
        )

        self.flow_estim = RAFTNet(False).cuda()
        self.flow_estim.eval()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)
        self.offset = None
        self.offset_bhw2 = None

    def prepareoffset(self, wid, hei):
        if self.offset is None or self.offset.shape[-2:] != (hei, wid):
            offsety, offsetx = torch.meshgrid([
                torch.linspace(0, hei - 1, hei),
                torch.linspace(0, wid - 1, wid)
            ])
            self.offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).float().cuda()
            self.offset_bhw2 = self.offset.permute(0, 2, 3, 1).contiguous()

    def warp_disp(self, flowf, flowb, disp):
        flowf_warp = warp_flow(flowf, flowb, offset=self.offset_bhw2, pad_mode="border")
        disp_warp = warp_flow(
            content=disp.unsqueeze(1),
            flow=-flowf_warp,
            offset=self.offset_bhw2,
            pad_mode="border"
        )
        return disp_warp

    def infer_forward(self, im: torch.Tensor, mode='pad_constant'):
        with torch.no_grad():
            if im.dim() == 3:
                im = im.unsqueeze(0)

            im = im.cuda()
            batchsz, cnl, hei, wid = im.shape
            if mode == 'resize':
                hei_new, wid_new = 512, 512 + 128 * 3
                img_padded = Resize([hei_new, wid_new])(im)
            elif 'pad' in mode:
                hei_new, wid_new = ((hei - 1) // 128 + 1) * 128, ((wid - 1) // 128 + 1) * 128
                padding = [
                    (wid_new - wid) // 2,
                    (wid_new - wid + 1) // 2,
                    (hei_new - hei) // 2,
                    (hei_new - hei + 1) // 2
                ]
                mode = mode[4:] if "pad_" in mode else "constant"
                img_padded = torchf.pad(im, padding, mode)
            else:
                raise NotImplementedError

            self.model.eval()
            netout = self.model(img_padded)

            # depad
            if mode == 'resize':
                netout = Resize([hei_new, wid_new])(netout)
            else:
                netout = netout[..., padding[2]: hei_new - padding[3], padding[0]: wid_new - padding[1]]
            mpi = netout2mpi(netout, im)
            self.model.train()

        return mpi

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        """
        basically same as train_forward
        only evaluate L1 loss
        also output displacement heat map & newview image
        """
        args = [_t.cuda() for _t in args]
        self.eval()
        refims, tarims, refextrins, tarextrin, intrin, pt2ds, ptzs_gts = args
        batchsz, framenum, _, heiori, widori = refims.shape
        self.prepareoffset(widori, heiori)
        layernum = self.model.num_layers
        bfnum_1 = batchsz * (framenum - 1)
        bfnum_2 = batchsz * (framenum - 2)
        with torch.no_grad():
            flows_f, flows_fma, _ = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                                    refims[:, 1:].reshape(bfnum_1, 3, heiori, widori),
                                                    ret_upmask=True)
            flows_b, flows_bma, _ = self.flow_estim(refims[:, 1:-1].reshape(bfnum_2, 3, heiori, widori),
                                                    refims[:, :-2].reshape(bfnum_2, 3, heiori, widori),
                                                    ret_upmask=True)
            flows_f = upsample_flow(flows_f, flows_fma)
            flows_b = upsample_flow(flows_b, flows_bma)
            flows_f = flows_f.reshape(batchsz, framenum - 1, 2, heiori, widori)
            flows_b = flows_b.reshape(batchsz, framenum - 2, 2, heiori, widori)
            depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

            netout0 = self.model(refims[:, 0])
            mpi0, bw0 = netout2mpi(netout0, refims[:, 0], ret_blendw=True)
            disp0 = estimate_disparity_torch(mpi0, depth, blendweight=bw0)

            disp_warp = self.warp_disp(flows_f[:, 0], flows_b[:, 0], disp0)

            disp_warp_list = [disp_warp]
            disp_list = []
            mpi_list = []

            for frameidx in range(1, framenum - 1):
                netout = self.model(refims[:, frameidx])
                mpi, bw = netout2mpi(netout, refims[:, frameidx], ret_blendw=True)
                disp = estimate_disparity_torch(mpi, depth, blendweight=bw)

                disp_list.append(disp)
                mpi_list.append(mpi)

                if frameidx >= framenum - 2:
                    break

                disp_warp = self.warp_disp(flows_f[:, frameidx], flows_b[:, frameidx], disp)
                disp_warp_list.append(disp_warp)

            ptdisp_es0 = torchf.grid_sample(disp0.unsqueeze(1), pt2ds[:, 0:1], align_corners=True).reshape(batchsz, -1)
            scale = torch.exp(torch.log(ptdisp_es0 * ptzs_gts[:, 0]).mean(dim=-1, keepdim=True))
            depth = depth * scale.reshape(-1, 1)
            # render target view
            tarviews = [
                render_newview(
                    mpi=mpi_list[i],
                    srcextrin=refextrins[:, i + 1],
                    tarextrin=tarextrin,
                    srcintrin=intrin,
                    tarintrin=intrin,
                    depths=depth,
                    ret_mask=False
                )
                for i in range(len(mpi_list))
            ]

        val_dict = {}
        # Photometric metric====================
        tarviews = torch.stack(tarviews, dim=1)
        targts = tarims.unsqueeze(1).expand_as(tarviews)
        metrics = ["psnr", "ssim"]
        for metric in metrics:
            error = compute_img_metric(
                targts.reshape(-1, 3, heiori, widori),
                tarviews.reshape(-1, 3, heiori, widori),
                metric=metric
            )
            val_dict[f"val_{metric}"] = error

        # depth difference===========================
        disp_es = [
            torchf.grid_sample(disp_list[i].unsqueeze(1),
                               pt2ds[:, i:i + 1],
                               align_corners=True).reshape(batchsz, -1)
            for i in range(len(disp_list))
        ]
        disp_es = torch.stack(disp_es, dim=1)
        z_gt = ptzs_gts[:, 1:-1]
        thresh = np.log(1.25) ** 2
        diff = (disp_es / scale.reshape(batchsz, -1, 1) - 1 / z_gt).abs().mean()
        diff_scale = torch.log(disp_es * z_gt / scale.reshape(-1, 1))
        diff_scale = torch.pow(diff_scale, 2)
        inliner_pct = (diff_scale < thresh).type(torch.float32).sum() / diff_scale.nelement()
        diff_scale = diff_scale.mean()
        val_dict["val_d_mae"] = diff
        val_dict["val_d_msle"] = diff_scale
        val_dict["val_d_goodpct"] = inliner_pct

        # depth temporal consistency loss==============
        temporal = torch.tensor(0.).type_as(refims)
        temporal_scale = torch.tensor(0.).type_as(refims)
        for disp, disp_warp in zip(disp_list, disp_warp_list):
            temporal += ((torch.abs(disp.unsqueeze(1) - disp_warp)) / scale.reshape(batchsz, 1, 1, 1)).mean()
            # print(f"disp_warp: min {disp_warp.min()}, max {disp_warp.max()}")
            # print(f"disp_warp: min {disp.min()}, max {disp.max()}")
            disp_warp = torch.max(disp_warp, torch.tensor(0.000001).type_as(disp_warp))
            temporal_scale += torch.pow(torch.log(disp_warp / disp), 2).mean()
        temporal = temporal / len(disp_list)
        temporal_scale /= len(disp_list)
        val_dict["val_t_mae"] = temporal
        val_dict["val_t_msle"] = temporal_scale

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            val_dict["vis_disp"] = draw_dense_disp(disp_list[1], 1)
            val_dict["vis_bgimg"] = (netout0[0, -3:] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_img"] = (refims[0, -2] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, refextrins, tarextrins, intrin, pt2dses, ptzs_gts = args
        batchsz, framenum, _, heiori, widori = refims.shape
        refim = refims.squeeze(0)
        tarim = tarims.expand_as(refim)
        refextrin = refextrins.squeeze(0)
        tarextrin = tarextrins.expand_as(refextrin)
        pt2ds = pt2dses.squeeze(0)
        ptzs_gt = ptzs_gts.squeeze(0)
        intrin = intrin.repeat(framenum, 1, 1)

        batchsz, _, heiori, widori = refim.shape
        netout = self.model(refim)

        # compute mpi from netout
        # scheduling the s_bg
        step = kwargs["step"] if "step" in kwargs else 0
        bg_pct = self.scheduler.get_value(step)
        mpi, blend_weight = netout2mpi(netout, refim, bg_pct=bg_pct, ret_blendw=True)

        # estimate depth map and sample sparse point depth
        depth = make_depths(32).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
        disparity = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)
        ptdis_e = torchf.grid_sample(disparity.unsqueeze(1), pt2ds.unsqueeze(1), align_corners=True) \
            .squeeze(1).squeeze(1)

        # with torch.no_grad():  # compute scale
        if self.scale_mode == "norm":
            scale = torch.exp(torch.log(ptdis_e * ptzs_gt).mean(dim=-1, keepdim=True))
            scale = scale * SCALE_SCALE + SCALE_EPS
        elif self.scale_mode == "fix":
            scale = torch.exp(torch.log(0.7 *
                                        torch.kthvalue(ptzs_gt, dim=-1,
                                                       k=int(0.05 * ptzs_gt.shape[-1]))[0]))
        elif self.scale_mode == "adaptive":
            scale_tar = torch.exp(torch.log(0.7 *
                                            torch.kthvalue(ptzs_gt, dim=-1,
                                                           k=int(0.05 * ptzs_gt.shape[-1]))[0]))
            scale = torch.exp(torch.log(ptdis_e * ptzs_gt).mean(dim=-1, keepdim=True))
            scale_tar = scale_tar.reshape_as(scale)
            scale = scale * 0.6 + scale_tar * 0.4
        else:
            raise RuntimeError(f"ModelandSVLoss::unrecognized scale mode {self.scale_mode}")

        depth *= scale
        # render target view
        tarview, tarmask = render_newview(mpi, refextrin, tarextrin, intrin, intrin, depth, True)
        # sparsedepthgt = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt)
        # sparsedepth1 = draw_sparse_depth(refim, pt2ds, ptdis_e / scale)
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        loss_dict["s_bg"] = torch.tensor(bg_pct).type_as(scale.detach())
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = self.photo_loss(tarview, tarim)
            l1_loss = l1_loss.mean()
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = l1_loss.detach()

        if "smooth_loss" in self.loss_weight.keys():
            smth_schedule = self.smth_scheduler.get_value(step)
            smth_loss = smooth_grad(disparity, refim)
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"] * smth_schedule)
            loss_dict["smth"] = smth_loss.detach()

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(ptdis_e * ptzs_gt / scale)
            diff = torch.pow(diff, 2).mean()
            final_loss += (diff * self.loss_weight["depth_loss"])
            loss_dict["depth"] = diff.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class ModelandDispLoss(nn.Module):
    """
    single view semi-dense disparity map
    """

    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        self.loss_weight = cfg["loss_weights"].copy()
        torch.set_default_tensor_type(torch.FloatTensor)

        self.model = model.cuda()
        self.model.train()
        self.layernum = self.model.num_layers

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "vgg")
        self.scale_mode = self.loss_weight.pop("scale_mode", "norm")  # first / fix / adaptive

        # used for backward warp
        self.flow_estim = RAFTNet(False).cuda()
        self.flow_estim.eval()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)
        self.offset = None
        self.offset_bhw2 = None

        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        self.scheduler = ParamScheduler([3e4, 5e4], [0.5, 1])
        self.affine_perturbation = RandomAffine(
            degrees=(-1, 1),
            translate=(-2, 2),
            scale=(0.97, 1.03),
            shear=(-1, 1, -1, 1),
        )

    def infer_forward(self, im: torch.Tensor, ret_cfg=None):
        with torch.no_grad():
            if im.dim() == 3:
                im = im.unsqueeze(0)
            im = im.cuda()
            self.model.eval()
            netout = self.model(im)
            mpi = netout2mpi(netout, im)
            self.model.train()

        return mpi

    def prepareoffset(self, wid, hei):
        if self.offset is None or self.offset.shape[-2:] != (hei, wid):
            offsety, offsetx = torch.meshgrid([
                torch.linspace(0, hei - 1, hei),
                torch.linspace(0, wid - 1, wid)
            ])
            self.offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).float().cuda()
            self.offset_bhw2 = self.offset.permute(0, 2, 3, 1).contiguous()

    def warp_disp(self, flowf, flowb, disp):
        flowf_warp = warp_flow(flowf, flowb, offset=self.offset_bhw2, pad_mode="border")
        disp_warp = warp_flow(
            content=disp.unsqueeze(1),
            flow=-flowf_warp,
            offset=self.offset_bhw2,
            pad_mode="border"
        )
        return disp_warp

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        self.eval()
        refims, tarims, disp_gts, certainty_maps, isleft = args
        shift_gt = isleft[:, 1]
        isleft = isleft[:, 0]
        certainty_norm = certainty_maps.sum(dim=[-1, -2])
        batchsz, framenum, _, heiori, widori = refims.shape
        self.prepareoffset(widori, heiori)
        layernum = self.model.num_layers
        bfnum_1 = batchsz * (framenum - 1)
        bfnum_2 = batchsz * (framenum - 2)
        with torch.no_grad():
            flows_f, flows_fma, _ = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                                    refims[:, 1:].reshape(bfnum_1, 3, heiori, widori),
                                                    ret_upmask=True)
            flows_b, flows_bma, _ = self.flow_estim(refims[:, 1:-1].reshape(bfnum_2, 3, heiori, widori),
                                                    refims[:, :-2].reshape(bfnum_2, 3, heiori, widori),
                                                    ret_upmask=True)
            flows_f = upsample_flow(flows_f, flows_fma)
            flows_b = upsample_flow(flows_b, flows_bma)
            flows_f = flows_f.reshape(batchsz, framenum - 1, 2, heiori, widori)
            flows_b = flows_b.reshape(batchsz, framenum - 2, 2, heiori, widori)
            depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

            netout0 = self.model(refims[:, 0])
            mpi0, bw0 = netout2mpi(netout0, refims[:, 0], ret_blendw=True)
            disp0 = estimate_disparity_torch(mpi0, depth, blendweight=bw0)

            disp_warp = self.warp_disp(flows_f[:, 0], flows_b[:, 0], disp0)

            disp_warp_list = [disp_warp]
            disp_list = []
            mpi_list = []

            for frameidx in range(1, framenum - 1):
                netout = self.model(refims[:, frameidx])
                mpi, bw = netout2mpi(netout, refims[:, frameidx], ret_blendw=True)
                disp = estimate_disparity_torch(mpi, depth, blendweight=bw)

                disp_list.append(disp)
                mpi_list.append(mpi)

                if frameidx >= framenum - 2:
                    break

                disp_warp = self.warp_disp(flows_f[:, frameidx], flows_b[:, frameidx], disp)
                disp_warp_list.append(disp_warp)

            disp_diff0 = disp0 / (torch.abs(disp_gts[:, 0] - shift_gt.reshape(batchsz, 1, 1)) + 0.0001)
            # estimate disparity to ground truth
            scale = torch.exp((torch.log(disp_diff0) * certainty_maps[:, 0]).sum(dim=[-1, -2]) / certainty_norm[:, 0])
            disparities = torch.reciprocal(depth * scale.reshape(-1, 1) * -isleft.reshape(-1, 1)) \
                          + shift_gt.reshape(-1, 1)
            # render target view
            tarviews = [
                shift_newview(mpi_, -disparities)
                for mpi_ in mpi_list
            ]

        val_dict = {}
        # Photometric metric====================
        tarviews = torch.stack(tarviews, dim=1)
        targts = tarims[:, 1:-1]
        metrics = ["psnr", "ssim"]
        for metric in metrics:
            error = compute_img_metric(
                targts.reshape(-1, 3, heiori, widori),
                tarviews.reshape(-1, 3, heiori, widori),
                metric=metric
            )
            val_dict[f"val_{metric}"] = error

        # depth difference===========================
        disp = torch.stack(disp_list, dim=1) / scale.reshape(batchsz, 1, 1, 1)
        disp_gt = (disp_gts[:, 1:-1] - shift_gt.reshape(batchsz, 1, 1, 1)) \
                  * -isleft.reshape(batchsz, 1, 1, 1)
        mask = certainty_maps[:, 1:-1]
        mask_norm = certainty_norm[:, 1:-1].sum()

        thresh = np.log(1.25) ** 2  # log(1.25) ** 2
        diff = ((disp - disp_gt).abs() * mask).sum() / mask_norm
        diff_scale = torch.pow(torch.log((disp_gt / disp).abs() + 0.00001), 2)
        inliner_pct = ((diff_scale < thresh) * mask).sum() / mask_norm
        diff_scale = (diff_scale * mask).sum() / mask_norm
        val_dict["val_d_mae"] = diff
        val_dict["val_d_msle"] = diff_scale
        val_dict["val_d_goodpct"] = inliner_pct

        # depth temporal consistency loss==============
        temporal = torch.tensor(0.).type_as(refims)
        temporal_scale = torch.tensor(0.).type_as(refims)
        for disp, disp_warp in zip(disp_list, disp_warp_list):
            disp_warp /= scale.reshape(batchsz, 1, 1, 1)
            temporal += (torch.abs(disp.unsqueeze(1) - disp_warp)).mean()
            # print(f"disp_warp: min {disp_warp.min()}, max {disp_warp.max()}")
            # print(f"disp_warp: min {disp.min()}, max {disp.max()}")
            disp_warp = torch.max(disp_warp, torch.tensor(0.000001).type_as(disp_warp))
            temporal_scale += torch.pow(torch.log(disp_warp / disp), 2).mean()
        temporal /= len(disp_list)
        temporal_scale /= len(disp_list)
        val_dict["val_t_mae"] = temporal
        val_dict["val_t_msle"] = temporal_scale

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            val_dict["vis_dispgt"] = draw_dense_disp(disp_gt[:, 1], 1)
            val_dict["vis_disp"] = draw_dense_disp(disp_list[1], 1)
            val_dict["vis_bgimg"] = (netout0[0, -3:] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_img"] = (refims[0, -2] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        refim = refims.squeeze(0)
        tarim = tarims.squeeze(0)
        disp_gt = disp_gts.squeeze(0)
        certainty_map = certainty_maps.squeeze(0)
        framenum = refim.shape[0]
        isleft = isleft.repeat(framenum, 1)

        # refim, tarim, disp_gt, certainty_map, isleft = args
        shift_gt = isleft[:, 1]
        isleft = isleft[:, 0]
        certainty_norm = certainty_map.sum(dim=[-1, -2])

        batchsz, _, heiori, widori = refim.shape
        netout = self.model(refim)
        # compute mpi from netout
        # scheduling the s_bg
        step = kwargs["step"] if "step" in kwargs else 0
        bg_pct = self.scheduler.get_value(step)
        mpi, blend_weight = netout2mpi(netout, refim, bg_pct=bg_pct, ret_blendw=True)
        # estimate depth map and sample sparse point depth
        depth = make_depths(self.layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
        disp_hat = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)

        disp_diff = disp_hat / (torch.abs(disp_gt - shift_gt.reshape(batchsz, 1, 1)) + 0.000001)

        if self.scale_mode == "norm":
            scale = torch.exp((torch.log(disp_diff) * certainty_map).sum(dim=[-1, -2], keepdim=True)
                              / certainty_norm.reshape(batchsz, 1, 1))
            scale = scale * SCALE_SCALE + SCALE_EPS
        elif self.scale_mode == "fix":
            dispgt = torch.abs(disp_gt - shift_gt.reshape(batchsz, 1, 1))
            mask = certainty_map > 0
            kthval = [torch.kthvalue(dispgt[i][mask[i]], int(0.95 * mask[i].sum()))[0]
                      for i in range(mask.shape[0])]
            kthval = torch.stack(kthval, dim=0)
            scale = torch.exp(torch.log(0.7 / kthval)).reshape(batchsz, 1, 1)
        elif self.scale_mode == "adaptive":
            dispgt = torch.abs(disp_gt - shift_gt.reshape(batchsz, 1, 1))
            mask = certainty_map > 0
            kthval = [torch.kthvalue(dispgt[i][mask[i]], int(0.95 * mask[i].sum()))[0]
                      for i in range(mask.shape[0])]
            kthval = torch.stack(kthval, dim=0)
            scale_tar = torch.exp(torch.log(0.7 / kthval)).reshape(batchsz, 1, 1)

            scale = torch.exp((torch.log(disp_diff) * certainty_map).sum(dim=[-1, -2], keepdim=True)
                              / certainty_norm.reshape(batchsz, 1, 1))
            scale = 0.6 * scale + 0.4 * scale_tar
        else:
            raise RuntimeError(f"ModelandDispLoss::unrecognized scale mode {self.scale_mode}")

        disparities = torch.reciprocal(depth * scale.reshape(-1, 1) * -isleft.reshape(-1, 1)) \
                      + shift_gt.reshape(-1, 1)
        # render target view
        tarview = shift_newview(mpi, -disparities, ret_mask=False)
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        loss_dict["fg_bg"] = torch.tensor(bg_pct).type_as(scale.detach())
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = self.photo_loss(tarview, tarim)
            l1_loss = l1_loss.mean()
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = l1_loss.detach()

        if "smooth_loss" in self.loss_weight.keys():
            smth_loss = smooth_grad(disp_hat, refim)
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"])
            loss_dict["smth"] = smth_loss.detach()

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(disp_diff / scale.reshape(-1, 1, 1))
            diff = (torch.pow(diff, 2) * certainty_map).mean()
            final_loss += (diff * self.loss_weight["depth_loss"])
            loss_dict["depth"] = diff.detach()

        if "sparse_loss" in self.loss_weight.keys():
            raise NotImplementedError(f"ModelandDispLoss:: sparse_loss not implement")

        if "temporal_loss" in self.loss_weight.keys():
            self.affine_perturbation.random_generate(refim)
            img_pert = self.affine_perturbation.apply(refim)
            net_out_gt = self.affine_perturbation.apply(netout)
            net_out_pert = self.model(img_pert)
            temporal_loss = torch.sqrt(torch.pow(net_out_gt - net_out_pert, 2).mean())
            alpha = self.loss_weight["temporal_loss"]
            final_loss += temporal_loss * alpha / (1 - alpha)
            loss_dict["temporal"] = temporal_loss

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class ModelandLossJoint(nn.Module):
    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__()
        self.pipeline_disp = ModelandDispLoss(models, cfg)
        cfg["loss_weights"]["depth_loss"] /= 5
        self.pipeline_sv = ModelandSVLoss(models, cfg)
        del self.pipeline_sv.flow_estim
        del self.pipeline_sv.photo_loss
        self.pipeline_sv.flow_estim = self.pipeline_disp.flow_estim
        self.pipeline_sv.photo_loss = self.pipeline_disp.photo_loss

    def valid_forward(self, *args, **kwargs):
        if len(args) == 5:
            return self.pipeline_disp.valid_forward(*args, **kwargs)
        elif len(args) == 7:
            return self.pipeline_sv.valid_forward(*args, **kwargs)
        else:
            raise RuntimeError()

    def forward(self, *args, **kwargs):
        if len(args) == 5:
            return self.pipeline_disp.forward(*args, **kwargs)
        elif len(args) == 7:
            return self.pipeline_sv.forward(*args, **kwargs)
        else:
            raise RuntimeError()


class PipelineV2(nn.Module):
    """
    The entire pipeline using backward warping
    """

    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__()
        self.loss_weight = cfg["loss_weights"].copy()
        torch.set_default_tensor_type(torch.FloatTensor)

        assert (isinstance(models, nn.ModuleDict))
        models.train()
        models = models.cuda()
        self.mpimodel = models["MPI"]
        self.sfmodel = models["SceneFlow"] if "SceneFlow" in models.keys() else None
        self.afmodel = models["AppearanceFlow"]

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        # self.splat_func = forward_scatter_withweight

        # adjustable config
        self.aflow_fusefgpct = self.loss_weight.pop("aflow_fusefgpct", False)
        self.flownet_dropout = self.loss_weight.pop("flownet_dropout", 0)  # pct of drop the net.
        self.aflow_includeself = self.loss_weight.pop("aflow_includeself", False)
        self.scale_scaling = self.loss_weight.pop("scale_scaling", SCALE_SCALE)
        self.scale_mode = self.loss_weight.pop("scale_mode", "adaptive")  # first / mean / random
        self.depth_loss_mode = self.loss_weight.pop("depth_loss_mode",
                                                    "fine")  # fine, coarse, direct_fine, direct_coarse
        self.tempnewview_mode = self.loss_weight.pop("tempnewview_mode",  # biflow / imwarp
                                                     "imwarp")
        self.upmask_magaware = self.loss_weight.pop("upmask_magaware", False)

        print(f"PipelineV2 activated config:\n"
              f"aflow_includeself: {self.aflow_includeself}\n"
              f"flownet_dropout: {self.flownet_dropout}\n"
              f"scale_scaling: {self.scale_scaling}\n"
              f"scale_mode: {self.scale_mode}\n"
              f"tempnewview_mode: {self.tempnewview_mode}\n"
              f"aflow_fusefgpct: {self.aflow_fusefgpct}\n"
              f"depth_loss_mode: {self.depth_loss_mode}\n"
              f"")

        # optical flow estimator
        if not hasattr(self, "flow_estim"):
            self.flow_estim = RAFTNet(False).cuda()
            self.flow_estim.eval()
            state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.flow_estim.load_state_dict(state_dict)
            for param in self.flow_estim.parameters():
                param.requires_grad = False

        # self.flow_loss_ord = self.loss_weight.pop("flow_smth_ord", 2)
        # self.flow_loss_consider_weight = self.loss_weight.pop("flow_smth_bw", False)

        # used for backward warp
        self.offset = None
        self.offset_bhw2 = None
        self.offset_d8 = None
        self.offset_bhw2_d8 = None

        self.bgfgfuse_scheduler = ParamScheduler(
            milestones=[2e3, 4e3],
            values=[0.5, 1]
        ) if self.aflow_fusefgpct else None

        upmask_warmup_val = self.loss_weight["mask_warmup"] if "mask_warmup" in self.loss_weight.keys() else 0
        upmask_warmup_milstone = self.loss_weight.pop("mask_warmup_milestone", [5e3, 8e3])
        self.upmask_warmup_scheduler = ParamScheduler(
            milestones=upmask_warmup_milstone,
            values=[upmask_warmup_val, 0]
        )
        bgflow_warmup_val = self.loss_weight["bgflow_warmup"] if "bgflow_warmup" in self.loss_weight.keys() else 0
        bgflow_warmup_milstone = self.loss_weight.pop("bgflow_warmup_milestone", [5e3, 8e3])
        self.bgflow_warmup_scheduler = ParamScheduler(
            milestones=bgflow_warmup_milstone,
            values=[bgflow_warmup_val, 0]
        )
        net_warmup_val = self.loss_weight["net_warmup"] if "net_warmup" in self.loss_weight.keys() else 0
        net_warmup_milestone = self.loss_weight.pop("net_warmup_milestone", [5e3, 8e3])

        self.net_warmup_scheduler = ParamScheduler(
            milestones=net_warmup_milestone,
            values=[net_warmup_val, 0]
        )
        self.smth_scheduler = ParamScheduler(
            milestones=[4e3, 6e3],
            values=[0, 1]
        )

        # used for inference
        self.neighbor_sz = 1
        self.max_win_sz = self.neighbor_sz * 2 + 1
        self.img_window = deque()
        self.upmask_window = deque()
        self.flowf_win, self.flowb_win = deque(), deque()
        self.flowfnet_win, self.flowbnet_win = deque(), deque()
        self.flowfmask_win, self.flowbmask_win = deque(), deque()
        self.net_warp = None
        self.lastmpi = None

    def infer_forward(self, img: torch.Tensor, ret_cfg: str):
        """
        new frames will be pushed into a queue, and will only output if len(queue)==self.maxwinsz
        """
        self.eval()
        if "restart" in ret_cfg:
            self.img_window.clear()
            self.flowb_win.clear()
            self.flowf_win.clear()
            self.flowbmask_win.clear()
            self.flowfmask_win.clear()
            self.flowfnet_win.clear()
            self.flowbnet_win.clear()
            self.upmask_window.clear()
            self.net_warp = None
            self.lastmpi = None

        if "nolast" in ret_cfg:
            self.net_warp = None
            self.lastmpi = None

        if img.dim() == 3:
            img = img.unsqueeze(0)
        bsz, _, hei, wid = img.shape
        self.prepareoffset(wid, hei)
        self.img_window.append(img)
        if len(self.img_window) > 1:
            flowf, flowf_upmask, flowf_net = self.flow_estim(self.img_window[-2],
                                                             self.img_window[-1],
                                                             ret_upmask=True)
            flowb, flowb_upmask, flowb_net = self.flow_estim(self.img_window[-1],
                                                             self.img_window[-2],
                                                             ret_upmask=True)
            self.flowf_win.append(flowf)
            self.flowfnet_win.append(flowf_net)
            self.flowfmask_win.append(flowf_upmask)
            self.flowb_win.append(flowb)
            self.flowbnet_win.append(flowb_net)
            self.flowbmask_win.append(flowb_upmask)

        if len(self.img_window) > self.max_win_sz:
            self.img_window.popleft()
            self.flowf_win.popleft(), self.flowfnet_win.popleft(), self.flowfmask_win.popleft()
            self.flowb_win.popleft(), self.flowbnet_win.popleft(), self.flowbmask_win.popleft()

        elif len(self.img_window) < self.max_win_sz:
            return None

        alpha, net, upmask = self.forwardmpi(
            None,
            self.img_window[1],
            [self.flowbnet_win[0], self.flowfnet_win[1]],
            self.net_warp
        )
        depth = make_depths(alpha.shape[1]).type_as(alpha).unsqueeze(0).repeat(1, 1)
        disp, blend_weight = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)

        img_list = [self.img_window[0], self.img_window[2]]
        flow_list = [self.flowb_win[0], self.flowf_win[1]]
        flownet_list = [self.flowbnet_win[0], self.flowfnet_win[1]]
        flowmask_list = [self.flowbmask_win[0], self.flowfmask_win[1]]
        flowb_list = [self.flowf_win[0], self.flowb_win[1]]
        flownetb_list = [self.flowfnet_win[0], self.flowbnet_win[1]]
        flowmaskb_list = [self.flowfmask_win[0], self.flowbmask_win[1]]

        if self.aflow_includeself or "aflow_includeself" in ret_cfg:
            img_list.append(self.img_window[1])
            with torch.no_grad():
                flowself, flownetself, flowmaskself = \
                    self.flow_estim(self.img_window[1], self.img_window[1])
            flow_list.append(flowself)
            flownet_list.append(flownetself)
            flowmask_list.append(flowmaskself)

        imbg = self.forwardaf(disp, net, upmask, self.img_window[1],
                              flows_list=flow_list,
                              imgs_list=img_list,
                              flow_nets_list=flownet_list,
                              flow_upmasks_list=flowmask_list,
                              flowsb_list=flowb_list,
                              flowb_nets_list=flownetb_list,
                              flowb_upmasks_list=flowmaskb_list)
        mpi = alpha2mpi(alpha, self.img_window[1], imbg, blend_weight=blend_weight)

        # update last frame info
        self.lastmpi = mpi
        self.net_warp = self.forwardsf(self.flowf_win[1], self.flowb_win[1], net)
        if "ret_net" in ret_cfg:
            ret_net = learned_upsample(net, upmask)
            ret_net = torch.cat([ret_net, self.img_window[1], imbg], dim=1)
            return mpi, ret_net
        return mpi

    def forward_multiframes(self, refims, step=None, intermediates=None):
        batchsz, framenum, _, heiori, widori = refims.shape
        layernum = self.mpimodel.num_layers
        self.prepareoffset(widori, heiori)
        bfnum_1 = batchsz * (framenum - 1)
        # All are [B x Frame [x cnl] x H x W]
        with torch.no_grad():
            # the flow in /8 resolution
            flows_f, flows_f_upmask, flows_f_net = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                                                   refims[:, 1:].reshape(bfnum_1, 3, heiori, widori),
                                                                   ret_upmask=True)
            flows_b, flows_b_upmask, flows_b_net = self.flow_estim(refims[:, 1:].reshape(bfnum_1, 3, heiori, widori),
                                                                   refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                                                   ret_upmask=True)
            flows_f = flows_f.reshape(batchsz, framenum - 1, 2, heiori // 8, widori // 8)
            flows_f_upmask = flows_f_upmask.reshape(batchsz, framenum - 1, -1, heiori // 8, widori // 8)
            flows_f_net = flows_f_net.reshape(batchsz, framenum - 1, -1, heiori // 8, widori // 8)
            flows_b = flows_b.reshape(batchsz, framenum - 1, 2, heiori // 8, widori // 8)
            flows_b_upmask = flows_b_upmask.reshape(batchsz, framenum - 1, -1, heiori // 8, widori // 8)
            flows_b_net = flows_b_net.reshape(batchsz, framenum - 1, -1, heiori // 8, widori // 8)

        mpi_out = []
        net_out = []
        upmask_out = []
        disp_out = []
        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        if intermediates is not None:
            if "disp0" in intermediates.keys():
                with torch.no_grad():
                    # flows_f0, flows_f_upmask0, flows_f_net0 = \
                    #     self.flow_estim(refims[:, 0], refims[:, 1], ret_upmask=True)
                    alpha, net, upmask = self.forwardmpi(
                        None,
                        refims[:, 0],
                        [flows_f_net[:, 0]],
                        None
                    )
                    disp0 = estimate_disparity_torch(alpha.unsqueeze(2), depth)
                    intermediates["disp0"] = disp0

                if "net_warp" in intermediates.keys():
                    net_warp = self.forwardsf(flows_f[:, 0], flows_b[:, 0], net)
                    intermediates["net_warp"].append(net_warp)

                if "disp_warp" in intermediates.keys():
                    disp0_warp = self.flowfwarp_warp(upsample_flow(flows_f[:, 0], flows_f_upmask[:, 0]),
                                                     upsample_flow(flows_b[:, 0], flows_b_upmask[:, 0]),
                                                     disp0.unsqueeze(1))
                    intermediates["disp_warp"].append(disp0_warp)

            if "flows_f_upmask" in intermediates.keys():
                intermediates["flows_f_upmask"] = flows_f_upmask
            if "flows_f" in intermediates.keys():
                intermediates["flows_f"] = flows_f
            if "flows_b_upmask" in intermediates.keys():
                intermediates["flows_b_upmask"] = flows_b_upmask
            if "flows_b" in intermediates.keys():
                intermediates["flows_b"] = flows_b

        for frameidx in range(1, framenum - 1):
            alpha, net, upmask = self.forwardmpi(
                intermediates,
                refims[:, frameidx],
                [flows_b_net[:, frameidx - 1], flows_f_net[:, frameidx]],
                None
            )
            disp, blend_weight = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)

            flow_list = [flows_b[:, frameidx - 1], flows_f[:, frameidx]]
            flow_net_list = [flows_b_net[:, frameidx - 1], flows_f_net[:, frameidx]]
            flow_mask_list = [flows_b_upmask[:, frameidx - 1], flows_f_upmask[:, frameidx]]
            flowb_list = [flows_f[:, frameidx - 1], flows_b[:, frameidx]]
            flowb_net_list = [flows_f_net[:, frameidx - 1], flows_b_net[:, frameidx]]
            flowb_mask_list = [flows_f_upmask[:, frameidx - 1], flows_b_upmask[:, frameidx]]
            image_list = [refims[:, frameidx - 1], refims[:, frameidx + 1]]
            if self.aflow_includeself:
                image_list.append(refims[:, frameidx])
                with torch.no_grad():
                    flow_self, flow_net_self, flow_upmask_self = \
                        self.flow_estim(refims[:, frameidx], refims[:, frameidx])
                    flow_list.append(flow_self)
                    flow_net_list.append(flow_net_self)
                    flow_mask_list.append(flow_upmask_self)

            imbg = self.forwardaf(disp, net, upmask, refims[:, frameidx],
                                  flows_list=flow_list,
                                  imgs_list=image_list,
                                  flow_nets_list=flow_net_list,
                                  flow_upmasks_list=flow_mask_list,
                                  flowsb_list=flowb_list,
                                  flowb_nets_list=flowb_net_list,
                                  flowb_upmasks_list=flowb_mask_list, intermediate=intermediates)
            if self.bgfgfuse_scheduler is not None and step is not None:
                fg_pct = self.bgfgfuse_scheduler.get_value(step)
                imfg = imbg * (1. - fg_pct) + refims[:, frameidx] * fg_pct
            else:
                imfg = refims[:, frameidx]
            mpi = alpha2mpi(alpha, imfg, imbg, blend_weight=blend_weight)
            mpi_out.append(mpi)
            net_out.append(net)
            upmask_out.append(upmask)
            disp_out.append(disp)

            if frameidx >= framenum - 2:
                break

            if intermediates is not None:
                if "net_warp" in intermediates.keys():
                    net_warp = self.forwardsf(
                        flowf=flows_f[:, frameidx],
                        flowb=flows_b[:, frameidx],
                        net=net
                    )
                    intermediates["net_warp"].append(net_warp)

                if "disp_warp" in intermediates.keys():
                    disp_warp = self.flowfwarp_warp(
                        upsample_flow(flows_f[:, frameidx], flows_f_upmask[:, frameidx]),
                        upsample_flow(flows_b[:, frameidx], flows_b_upmask[:, frameidx]),
                        disp.unsqueeze(1))
                    intermediates["disp_warp"].append(disp_warp)

                if "imbg" in intermediates.keys():
                    intermediates["imbg"] = imbg

        return mpi_out, net_out, upmask_out, disp_out

    @torch.no_grad()
    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        self.eval()
        refims, tarims, disp_gts, certainty_maps, isleft = args
        shift_gt = isleft[:, 1]
        isleft = isleft[:, 0]
        certainty_norm = certainty_maps.sum(dim=[-1, -2])
        batchsz, framenum, cnl, heiori, widori = refims.shape
        layernum = self.mpimodel.num_layers
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        intermediate = {"disp0": None, "disp_warp": [], "imbg": None}
        mpi_list, net_list, upmask_list, disp_list = self.forward_multiframes(refims, None, intermediate)
        disp0 = intermediate["disp0"]
        disp_warp_list = intermediate["disp_warp"]
        imbg = intermediate["imbg"]

        disp_diff0 = disp0 / (torch.abs(disp_gts[:, 0] - shift_gt.reshape(batchsz, 1, 1)) + 0.0001)
        # estimate disparity to ground truth
        scale = torch.exp((torch.log(disp_diff0) * certainty_maps[:, 0]).sum(dim=[-1, -2]) / certainty_norm[:, 0])
        disparities = torch.reciprocal(depth * scale.reshape(-1, 1) * -isleft.reshape(-1, 1)) \
                      + shift_gt.reshape(-1, 1)
        # render target view
        tarviews = [
            shift_newview(mpi_, -disparities)
            for mpi_ in mpi_list
        ]

        val_dict = {}
        # Photometric metric====================
        tarviews = torch.stack(tarviews, dim=1)
        targts = tarims[:, 1:-1]
        metrics = ["psnr", "ssim"]
        for metric in metrics:
            error = compute_img_metric(
                targts.reshape(-1, 3, heiori, widori),
                tarviews.reshape(-1, 3, heiori, widori),
                metric=metric
            )
            val_dict[f"val_{metric}"] = error

        # depth difference===========================
        disp = torch.stack(disp_list, dim=1) / scale.reshape(batchsz, 1, 1, 1)
        disp_gt = (disp_gts[:, 1:-1] - shift_gt.reshape(batchsz, 1, 1, 1)) \
                  * -isleft.reshape(batchsz, 1, 1, 1)
        mask = certainty_maps[:, 1:-1]
        mask_norm = certainty_norm[:, 1:-1].sum()

        thresh = np.log(1.25) ** 2  # log(1.25) ** 2
        diff = ((disp - disp_gt).abs() * mask).sum() / mask_norm
        diff_scale = torch.pow(torch.log((disp_gt / disp).abs() + 0.00001), 2)
        inliner_pct = ((diff_scale < thresh) * mask).sum() / mask_norm
        diff_scale = (diff_scale * mask).sum() / mask_norm
        val_dict["val_d_mae"] = diff
        val_dict["val_d_msle"] = diff_scale
        val_dict["val_d_goodpct"] = inliner_pct

        # depth temporal consistency loss==============
        temporal = torch.tensor(0.).type_as(refims)
        temporal_scale = torch.tensor(0.).type_as(refims)
        for disp, disp_warp in zip(disp_list, disp_warp_list):
            disp_warp /= scale.reshape(batchsz, 1, 1, 1)
            temporal += (torch.abs(disp.unsqueeze(1) - disp_warp)).mean()
            # print(f"disp_warp: min {disp_warp.min()}, max {disp_warp.max()}")
            # print(f"disp_warp: min {disp.min()}, max {disp.max()}")
            disp_warp = torch.max(disp_warp, torch.tensor(0.000001).type_as(disp_warp))
            disp = torch.max(disp, torch.tensor(0.000001).type_as(disp))
            temporal_scale += torch.pow(torch.log(disp / disp_warp), 2).mean()
        temporal /= len(disp_list)
        temporal_scale /= len(disp_list)
        val_dict["val_t_mae"] = temporal
        val_dict["val_t_msle"] = temporal_scale

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            val_dict["vis_dispgt"] = draw_dense_disp(disp_gt[:, 1], 1)
            val_dict["vis_disp"] = draw_dense_disp(disp_list[1], 1)
            val_dict["vis_bgimg"] = (imbg[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_img"] = (refims[0, -2] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()

        return val_dict

    def prepareoffset(self, wid, hei):
        if self.offset is None or self.offset.shape[-2:] != (hei, wid):
            offsety, offsetx = torch.meshgrid([
                torch.linspace(0, hei - 1, hei),
                torch.linspace(0, wid - 1, wid)
            ])
            self.offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).float().cuda()
            self.offset_bhw2 = self.offset.permute(0, 2, 3, 1).contiguous()

            wid //= 8
            hei //= 8
            offsety, offsetx = torch.meshgrid([
                torch.linspace(0, hei - 1, hei),
                torch.linspace(0, wid - 1, wid)
            ])
            self.offset_d8 = torch.stack([offsetx, offsety], 0).unsqueeze(0).float().cuda()
            self.offset_bhw2_d8 = self.offset_d8.permute(0, 2, 3, 1).contiguous()

    def forwardmpi(self, tmpcfg, img, flow_net_list: list, net_warp=None):
        """
        Now this only return alpha channles (like depth estimation)
        """
        batchsz, cnl, hei, wid = img.shape

        # if net_warp is None and (not hasattr(self.mpimodel, "recurrent") or self.mpimodel.recurrent):
        #     net_cnl = self.mpimodel.outcnl
        #     net_warp = torch.zeros(batchsz, net_cnl, hei // 8, wid // 8).type_as(img)

        # randomly drop one flow net if training
        if len(flow_net_list) == 1:
            with torch.no_grad():
                selfflow, _, flow_net = self.flow_estim(img, img, ret_upmask=True)
                flow_net_list.append(flow_net)
        if self.training:  # random switch and drop out
            if np.random.randint(0, 2) == 0:  # randomly switch
                flow_net_list[0], flow_net_list[1] = flow_net_list[1], flow_net_list[0]
            if isinstance(self, PipelineV2SV) and np.random.rand() < self.flownet_dropout:  # random drop
                flow_net_zero = torch.zeros_like(flow_net_list[0])
                flow_net_list = [flow_net_zero, flow_net_zero]

        flow_net = torch.cat(flow_net_list, dim=1)
        net, upmask = self.mpimodel(img, flow_net, net_warp)

        alpha, net_up = self.net2alpha(net, upmask, ret_netup=True)
        if tmpcfg is not None and "net_up" in tmpcfg.keys():
            tmpcfg["net_up"].append(net_up)
        return alpha, net, upmask

    def net2alpha(self, net, upmask, ret_netup=False):
        layernum = self.mpimodel.num_layers
        if upmask is None:
            netout = net
        else:
            netout = learned_upsample(net, upmask)
        batchsz, outcnl, hei, wid = netout.shape
        if isinstance(self.mpimodel, MPI_V5Nset):
            sigma, disp, thick = torch.split(netout, self.mpimodel.num_set, dim=1)
            x = make_depths(layernum).reshape(1, 1, layernum, 1, 1).type_as(netout)
            disp_min, disp_max = 1 / default_d_far, 1 / default_d_near
            disp = disp * (disp_max - disp_min) + disp_min
            alpha = self.multiparams2alpha(x,
                                           disp.unsqueeze(2),
                                           thick.unsqueeze(2),
                                           sigma.unsqueeze(2))
        elif isinstance(self.mpimodel, (MPI_V6Nset, MPI_AB_nonet, MPI_AB_up)):
            sigma, disp, thick = torch.split(netout, self.mpimodel.num_set, dim=1)
            x = torch.reciprocal(make_depths(layernum)).reshape(1, 1, layernum, 1, 1).type_as(netout)
            disp_min, disp_max = 1 / default_d_far, 1 / default_d_near
            disp = disp * (disp_max - disp_min) + disp_min
            alpha = self.nsets2alpha(x,
                                     disp.unsqueeze(2),
                                     thick.unsqueeze(2),
                                     sigma.unsqueeze(2))

        elif isinstance(self.mpimodel, MPI_AB_alpha):
            alpha = netout
        else:
            raise RuntimeError(f"Pipelinev2::incompatiable mpimodel: {type(self.mpimodel)}")

        alpha = torch.cat([torch.ones([batchsz, 1, hei, wid]).type_as(alpha), alpha], dim=1)
        if ret_netup:
            return alpha, netout
        else:
            return alpha

    def volume2alpha(self, x: torch.Tensor, depth, thick, sigma=4):
        t = x - depth + thick
        t = torch.relu(t)
        t = torch.min(t, thick)
        dt = t[:, 1:] - t[:, :-1]
        return - torch.exp(-dt * sigma * Sigma_Denorm) + 1

    def volume2alphaindepth(self, x, disp, thickindisp, sigma=1):
        thickindisp = torch.min(disp - 1. / default_d_far, thickindisp)
        depth0 = torch.reciprocal(disp)
        t = x - depth0
        t = torch.min(torch.relu(t), thickindisp * torch.reciprocal(disp - thickindisp) * depth0)
        dt = t[:, :-1] - t[:, 1:]
        return - torch.exp(-dt * sigma * Sigma_Denorm) + 1

    @staticmethod
    def multiparams2alpha(x, ds, th, sig):
        th = torch.min(ds - 1. / default_d_far, th)
        de = torch.reciprocal(ds)
        y = x - de
        y = torch.min(torch.relu(y), th * torch.reciprocal(ds - th) * de)
        dy = y[:, :, :-1] - y[:, :, 1:]
        expo = -(dy * sig).sum(dim=1)
        return -torch.exp(expo * Sigma_Denorm) + 1

    @staticmethod
    def nsets2alpha(x, ds, th, sig):
        th = torch.min(ds - 1. / default_d_far, th)
        t = x - ds + th
        t = torch.relu(t)
        t = torch.min(t, th)
        dt = t[:, :, 1:] - t[:, :, :-1]

        expo = -(dt * sig).sum(dim=1)
        return -torch.exp(expo * 20) + 1

    def params2alpha(self, x, ds1, ds2, t1, t2, s1, s2):
        t1 = torch.min(ds1 - 1. / default_d_far, t1)
        t2 = torch.min(ds2 - 1. / default_d_far, t2)
        de1 = torch.reciprocal(ds1)
        de2 = torch.reciprocal(ds2)
        x1 = x - de1
        x2 = x - de2
        x1 = torch.min(torch.relu(x1), t1 * torch.reciprocal(ds1 - t1) * de1)
        x2 = torch.min(torch.relu(x2), t2 * torch.reciprocal(ds2 - t2) * de2)
        dx1 = x1[:, :-1] - x1[:, 1:]
        dx2 = x2[:, :-1] - x2[:, 1:]
        return -torch.exp(-(dx1 * s1 + dx2 * s2) * 10) + 1

    def square2alpha(self, x, depth, thick, scale=1):
        denorm = torch.tensor(self.mpimodel.num_layers - 1).type_as(x)
        n = denorm * scale * (x - depth + thick)
        n = torch.max(n, torch.zeros(1).type_as(n))
        n = torch.min(n, denorm * scale * thick)
        return n[:, 1:] - n[:, :-1]

    def flowfwarp_warp(self, flowf, flowb, content):
        """
        :param flowf: t0 -> t1
        :param flowb: t1 -> t0
        :param content: t0
        :return: t1
        """
        if flowf.shape == self.offset.shape:
            offset = self.offset_bhw2
        else:
            offset = self.offset_bhw2_d8
        flowf_warp = warp_flow(flowf, flowb, offset=offset, pad_mode="border")
        # scheme2: warp using -flowf_warp
        content_warp = warp_flow(
            content=content,
            flow=-flowf_warp,
            offset=offset,
            pad_mode="border"
        )
        return content_warp

    def warp_and_occ(self, flowf, flowb, content):
        """
        :param flowf: t0 -> t1
        :param flowb: t1 -> t0
        :param content: t0
        :return: t1 and occ_t1
        """
        if flowf.shape == self.offset.shape:
            offset = self.offset_bhw2
        else:
            offset = self.offset_bhw2_d8
        content_warp = warp_flow(
            content=content,
            flow=flowb,
            offset=offset
        )
        occ = warp_flow(flowf, flowb, offset=offset) + flowb
        occ_norm = torch.norm(occ, dim=1, keepdim=True)
        return content_warp, occ_norm

    def forwardsf(self, flowf, flowb, net, intcfg=None):
        if self.sfmodel is None:
            net_last = net
        else:
            sflow = self.sfmodel(flowf, net)
            net_last = net + sflow
            raise NotImplementedError(f"sfmodel not implement")

        net_warp = self.flowfwarp_warp(flowf, flowb, net_last)
        return net_warp

    def forwardaf(self, disparity, net, upmask, curim,
                  flows_list: list, imgs_list: list, flow_nets_list: list, flow_upmasks_list: list,
                  flowsb_list: list, flowb_nets_list: list, flowb_upmasks_list: list,
                  intermediate=None):
        """
        Guarantee the flows[0] and imgs_list[0] to be the last frame
        """
        assert len(flows_list) == len(imgs_list) == len(flow_nets_list) == len(flow_upmasks_list)
        if isinstance(self.afmodel, AFNet):  # special deal
            assert len(flows_list) == 2
            bglist, malist = [], []
            for img, flow, flowb, flowupma, flowbupma in \
                    zip(imgs_list, flows_list, flowsb_list, flow_upmasks_list, flowb_upmasks_list):
                with torch.no_grad():
                    # 0:curim  1:img  flow:0->1 flowb:1->0
                    flow = upsample_flow(flow, flowupma) if flowupma is not None else flow
                    flowb = upsample_flow(flowb, flowbupma) if flowbupma is not None else flowb
                    # compute
                    # occ1 = torch.norm(warp_flow(flow, flowb, self.offset_bhw2) + flowb, dim=1, keepdim=True)
                    # occ1 = occ1.clamp_max(2)
                    rangemap1 = forward_scatter(flow, torch.ones_like(img[:, 0:1]), self.offset)
                    occ1 = - erode(rangemap1) + 1
                    rangemap0 = forward_scatter(flowb, occ1, self.offset)
                    flowbg = forward_scatter_withweight(flowb, -flowb, occ1, self.offset)
                    for i in range(3):
                        flowbg = -warp_flow(flowb, flowbg, offset=self.offset_bhw2)
                    imbg = warp_flow(img, flowbg, offset=self.offset_bhw2)
                    bglist.append(imbg)
                    malist.append(rangemap0)

            net_up = learned_upsample(net, upmask) if upmask is not None else net
            inp = torch.cat([net_up, curim] + bglist + malist, dim=1)
            bg = self.afmodel(inp)
            if intermediate is not None and "bg_supervise" in intermediate.keys():
                intermediate["bg_supervise"].append([bg, malist[0], bglist[0]])
                intermediate["bg_supervise"].append([bg, malist[1], bglist[1]])
            return bg

        masks, bgs = [], []
        # lastmask = None
        for flow, img, flow_net, flow_upmask in zip(flows_list, imgs_list, flow_nets_list, flow_upmasks_list):
            if isinstance(self.afmodel, ASPFNetWithMaskOut):
                flow = upsample_flow(flow, flow_upmask) if flow_upmask is not None else flow
                bgflow, mask = self.afmodel(flow, disparity)
                bgflow = flow + bgflow
            elif isinstance(self.afmodel, (AFNet_HR_netflowin, AFNet_HR_netflowinbig)):
                flow = upsample_flow(flow, flow_upmask) if flow_upmask is not None else flow
                net_up = learned_upsample(net, upmask) if upmask is not None else net
                bgflow, mask = self.afmodel(net_up, flow)
            elif isinstance(self.afmodel, AFNet_HR_netflowimgin):
                flow = upsample_flow(flow, flow_upmask)
                net_up = learned_upsample(net, upmask)
                bgflow, mask = self.afmodel(net_up, flow, curim)
            elif isinstance(self.afmodel, AFNet_AB_svdbg):
                net_up = learned_upsample(net, upmask)
                bg = self.afmodel(net_up, curim)
                bgs = [bg]
                masks = [torch.ones_like(bg)]
                break
            else:
                raise NotImplementedError("AFModel incompatiable")
            # if isinstance(self.afmodel, AFNet_HR_netflowin):
            #     bgflow, mask = self.afmodel(flow, disparity)
            # else:
            #     bgflow, mask = self.afmodel(flow, disparity, lastmask)
            #     lastmask = mask if lastmask is None else mask + lastmask

            bg = warp_flow(img, bgflow, offset=self.offset_bhw2)
            if intermediate is not None and "bgflow_fgflow" in intermediate.keys():
                if flow.shape[-2:] == curim.shape[-2:]:
                    intermediate["bgflow_fgflow"].append((bgflow, flow))
                else:
                    intermediate["bgflow_fgflow"].append((bgflow, upsample_flow(flow, flow_upmask)))

            masks.append(mask)
            bgs.append(bg)

        bgs = torch.stack(bgs)
        masks = torch.stack(masks)
        weight = torchf.softmax(masks, dim=0)  # masks / masks.sum(dim=0, keepdim=True)
        bg = (bgs * weight).sum(dim=0)

        return bg

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        self.train()
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        # remove first frame's gt
        tarims = tarims[:, 1:]
        disp_gts = disp_gts[:, 1:]
        shift_gt = isleft[:, 1]
        isleft = isleft[:, 0]
        certainty_maps = certainty_maps[:, 1:]
        certainty_norm = certainty_maps.sum(dim=[-1, -2])
        layernum = self.mpimodel.num_layers
        batchsz, framenum, cnl, heiori, widori = refims.shape
        step = kwargs["step"]
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        intermediates = {
            "flows_b_upmask": None,
            "flows_f_upmask": None,
            "flows_f": None,
            "flows_b": None,
            "bgflow_fgflow": [],
            "net_up": [],
            "bg_supervise": []
        }
        mpi_out, net_out, upmask_out, disp_out = self.forward_multiframes(refims, step, intermediates)
        flows_f_upmask = intermediates["flows_f_upmask"]
        flows_f = intermediates["flows_f"]
        flows_b_upmask = intermediates["flows_b_upmask"]
        flows_b = intermediates["flows_b"]
        # scale and shift invariant
        # with torch.no_grad():  # compute scale
        disp_diffs = [
            disp_out[i] / (torch.abs(disp_gts[:, i] - shift_gt.reshape(batchsz, 1, 1)) + 0.000001)
            for i in range(framenum - 2)
        ]
        if self.scale_mode == "first":
            scale = torch.exp((torch.log(disp_diffs[0]) * certainty_maps[:, 0]).sum(dim=[-1, -2])
                              / certainty_norm[:, 0])
            scale = scale * self.scale_scaling + SCALE_EPS
        elif self.scale_mode == "mean":
            scale = [
                torch.exp((torch.log(disp_diffs[i]) * certainty_maps[:, i]).sum(dim=[-1, -2]) / certainty_norm[:, i])
                for i in range(framenum - 2)
            ]
            scale = torch.stack(scale, dim=1).mean(dim=1)
            scale = scale * self.scale_scaling + SCALE_EPS
        elif self.scale_mode == "random":
            i = np.random.randint(0, framenum - 2)
            scale = torch.exp((torch.log(disp_diffs[i]) * certainty_maps[:, i]).sum(dim=[-1, -2])
                              / certainty_norm[:, i])
            scale = scale * self.scale_scaling + SCALE_EPS
        elif self.scale_mode == "fix":
            dispgt = torch.abs(disp_gts[:, 0] - shift_gt.reshape(batchsz, 1, 1))
            mask = certainty_maps[:, 0] > 0
            kthval = torch.kthvalue(dispgt[mask], int(0.95 * certainty_maps[:, 0].sum()))[0]
            scale = torch.exp(torch.log(0.7 / kthval))

        elif self.scale_mode == "adaptive":
            dispgt = torch.abs(disp_gts[:, 0] - shift_gt.reshape(batchsz, 1, 1))
            mask = certainty_maps[:, 0] > 0
            kthval = torch.kthvalue(dispgt[mask], int(0.95 * certainty_maps[:, 0].sum()))[0]
            scale_tar = torch.exp(torch.log(0.7 / kthval))
            i = np.random.randint(0, framenum - 2)
            scale = torch.exp((torch.log(disp_diffs[i]) * certainty_maps[:, i]).sum(dim=[-1, -2])
                              / certainty_norm[:, i])
            scale = scale * 0.6 + scale_tar * 0.4
        else:
            raise RuntimeError(f"PipelineV2::unrecognized scale mode {self.scale_mode}")

        # disparity in ground truth space
        disparities = torch.reciprocal(depth * scale.reshape(-1, 1) * -isleft.reshape(-1, 1)) + shift_gt.reshape(-1, 1)
        # render target view
        tarviews = [
            shift_newview(mpi_, -disparities, ret_mask=False)
            for mpi_ in mpi_out
        ]
        return self.compute_loss(
            False, refims, tarims,
            net_out, upmask_out, mpi_out, tarviews, disp_out, scale,
            flows_f, flows_b, flows_f_upmask, flows_b_upmask,
            disp_diffs, certainty_maps,
            None, None,
            intermediates, step
        )

    def compute_loss(self, issparse: bool, refims, tarims,
                     net_out, upmask_out, mpi_out, tarviews, disp_out, scale,
                     flowsf, flowsb, flowsf_upmask, flowsb_upmask,
                     disp_diffs, certainty_maps,  # if issparse is False
                     ptdis_es, ptzs_gts,  # if issparse is True
                     intermediates, step,
                     ):
        batchsz, framenum, _, heiori, widori = refims.shape
        layernum = self.mpimodel.num_layers
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        if self.bgfgfuse_scheduler is not None:
            fg_pct = self.bgfgfuse_scheduler.get_value(step)
            loss_dict["fg_pct"] = torch.tensor(fg_pct).type_as(final_loss)
        # MPI loss
        for mpiframeidx in range(len(mpi_out)):
            if "pixel_loss" in self.loss_weight.keys():
                tarim = tarims if issparse else tarims[:, mpiframeidx]
                l1_loss = self.photo_loss(tarviews[mpiframeidx], tarim)
                l1_loss = l1_loss.mean()
                final_loss += (l1_loss * self.loss_weight["pixel_loss"])
                if "pixel" not in loss_dict.keys():
                    loss_dict["pixel"] = l1_loss.detach()
                else:
                    loss_dict["pixel"] += l1_loss.detach()

            smth_schedule = self.smth_scheduler.get_value(step)
            if "smooth_loss" in self.loss_weight.keys():
                raise DeprecationWarning("PipelineV2::smooth_Loss is deprecated now, will do nothing")

            if "net_smth_loss" in self.loss_weight.keys():
                if isinstance(self.mpimodel, MPI_AB_alpha):
                    net = disp_out[mpiframeidx].unsqueeze(1)
                    net_smth_loss = smooth_grad(net, refims[:, mpiframeidx])
                    net_smth_loss = net_smth_loss.mean()
                else:
                    net = intermediates["net_up"][mpiframeidx]
                    num_cnl = net.shape[1]
                    bias = torch.ones(1, num_cnl, 1, 1).type_as(net)
                    bias[:, :(num_cnl // 3)] *= 0.5
                    net_smth_loss = smooth_grad(
                        net * bias,
                        refims[:, mpiframeidx + 1]
                    )
                    net_smth_loss = net_smth_loss.mean()

                final_loss += (net_smth_loss * self.loss_weight["net_smth_loss"] * smth_schedule)
                if "netsmth" not in loss_dict.keys():
                    loss_dict["netsmth"] = net_smth_loss.detach()
                else:
                    loss_dict["netsmth"] += net_smth_loss.detach()

            if "net_smth_loss_fg" in self.loss_weight.keys():
                raise DeprecationWarning("PipelineV2::net_smth_loss_fg is deprecated now, will do nothing")
                # net = intermediates["net_up"][mpiframeidx]
                # layerfg = net[:, :3] * torch.tensor([0.2, 1, 1]).reshape(1, 3, 1, 1).type_as(net)
                # fg_smth_loss = smooth_grad(
                #     layerfg,
                #     refims[:, mpiframeidx + 1]
                # )
                # fg_smth_loss = fg_smth_loss.mean()
                #
                # final_loss += (fg_smth_loss * self.loss_weight["net_smth_loss_fg"] * smth_schedule)
                # if "netfgsmth" not in loss_dict.keys():
                #     loss_dict["netfgsmth"] = fg_smth_loss.detach()
                # else:
                #     loss_dict["netfgsmth"] += fg_smth_loss.detach()

            if "net_smth_loss_bg" in self.loss_weight.keys():
                raise DeprecationWarning("PipelineV2::net_smth_loss_bg is deprecated now, will do nothing")
                # net = intermediates["net_up"][mpiframeidx]
                # layerbg = net[:, 3:] * torch.tensor([0.2, 1, 1]).reshape(1, 3, 1, 1).type_as(net)
                # bg_smth_loss = smooth_grad(
                #     layerbg,
                #     refims[:, mpiframeidx + 1]
                # )
                # bg_smth_loss = bg_smth_loss.mean()
                #
                # final_loss += (bg_smth_loss * self.loss_weight["net_smth_loss_bg"] * smth_schedule)
                # if "netbgsmth" not in loss_dict.keys():
                #     loss_dict["netbgsmth"] = bg_smth_loss.detach()
                # else:
                #     loss_dict["netbgsmth"] += bg_smth_loss.detach()

            if "depth_loss" in self.loss_weight.keys():
                if issparse:
                    diff = torch.log(ptdis_es[mpiframeidx] * ptzs_gts[:, mpiframeidx] / scale.reshape(-1, 1))
                    diff = torch.pow(diff, 2).mean()
                else:
                    diff = torch.log(disp_diffs[mpiframeidx] / scale.reshape(-1, 1, 1))
                    diff = (torch.pow(diff, 2) * certainty_maps[:, mpiframeidx]).mean()
                final_loss += (diff * self.loss_weight["depth_loss"])
                if "depth" not in loss_dict.keys():
                    loss_dict["depth"] = diff.detach()
                else:
                    loss_dict["depth"] += diff.detach()

        if "tempdepth_loss" in self.loss_weight.keys():
            raise DeprecationWarning(f"PipelineV2::tempdepth loss is no more used")

        if "bgflowsmth_loss" in self.loss_weight.keys():
            raise DeprecationWarning(f"PipelineV2:: bgflow_smooth loss is no use")

        if "tempnewview_loss" in self.loss_weight.keys():
            if issparse:
                tarviews = torch.stack(tarviews, dim=1)
                meanview = tarviews.mean(dim=1, keepdim=True)
                diff = (tarviews - meanview).abs().mean()
            else:
                bfnum = batchsz * (framenum - 3)
                tarviews = torch.stack(tarviews, dim=1)
                tarviews0 = tarviews[:, :-1].reshape(bfnum, 3, heiori, widori)
                tarviews1 = tarviews[:, 1:].reshape(bfnum, 3, heiori, widori)
                tarviewgts0 = tarims[:, :-2].reshape(bfnum, 3, heiori, widori)
                tarviewgts1 = tarims[:, 1:-1].reshape(bfnum, 3, heiori, widori)
                with torch.no_grad():
                    nvflows_b, nvflows_b_upmask, _ = self.flow_estim(tarviewgts1, tarviewgts0, ret_upmask=True)
                    nvflows_b = upsample_flow(nvflows_b, nvflows_b_upmask)

                if self.tempnewview_mode == "imwarp":
                    tarviewgts1_warp = warp_flow(tarviewgts0, nvflows_b, offset=self.offset_bhw2)
                    weight = (tarviewgts1_warp - tarviewgts1).abs().mean(dim=1, keepdim=True)
                    weight = torch.clamp(torch.exp(-weight * 10 + 0.5), 0, 1)
                elif self.tempnewview_mode == "biflow":
                    with torch.no_grad():
                        nvflows_f, nvflows_f_upmask, _ = self.flow_estim(tarviewgts0, tarviewgts1, ret_upmask=True)
                        nvflows_f = upsample_flow(nvflows_f, nvflows_f_upmask)
                        occ_b = warp_flow(nvflows_f, nvflows_b, offset=self.offset_bhw2) + nvflows_b
                        weight = torch.norm(occ_b, dim=1, keepdim=True)
                        weight = torch.clamp(torch.exp(-weight * 2 + 0.5), 0, 1)
                else:
                    raise RuntimeError(f"PipelineV2::unrecognized tempnewview_mode: {self.tempnewview_mode}")

                tarviews1_warp = warp_flow(tarviews0, nvflows_b, offset=self.offset_bhw2)
                diff = ((tarviews1_warp - tarviews1).abs() * weight).mean()

            final_loss += (self.loss_weight["tempnewview_loss"] * diff)
            loss_dict["temp_nv"] = diff.detach()

        # Warmup
        if "mask_warmup" in self.loss_weight.keys():
            mask_loss = torch.tensor(0.).type_as(refims)
            for i in range(len(upmask_out)):
                if np.random.randint(0, 2) == 0:
                    flow = flowsf[:, i + 1]
                    flowmask = flowsf_upmask[:, i + 1]
                else:
                    flow = flowsb[:, i]
                    flowmask = flowsb_upmask[:, i]

                if self.upmask_magaware:
                    with torch.no_grad():
                        weightx, weighty = gradient(flow)
                        weight = (weightx.abs() + weighty.abs()).sum(dim=1, keepdim=True)
                        weight = torch.clamp(weight, 0, 1)
                else:
                    weight = 1

                diff = (upmask_out[i] - flowmask).abs().mean(dim=1) * weight
                mask_loss += diff.mean()

            mask_loss /= len(upmask_out)
            weight = self.upmask_warmup_scheduler.get_value(step)
            final_loss += (weight * mask_loss)
            loss_dict["mask_warmup"] = mask_loss.detach()

        if "bgflow_warmup" in self.loss_weight.keys():
            bgflow_suloss = torch.tensor(0.).type_as(refims)
            for bgflow, fgflow in intermediates["bgflow_fgflow"]:
                epe = torch.norm(bgflow - fgflow, dim=1)
                bgflow_suloss += epe.mean()
            bgflow_suloss /= len(intermediates["bgflow_fgflow"])
            weight = self.bgflow_warmup_scheduler.get_value(step)
            final_loss += (weight * bgflow_suloss)
            loss_dict["bgflow_warmup"] = bgflow_suloss.detach()

        if "bg_supervision" in self.loss_weight.keys():
            bg_suloss = torch.tensor(0.).type_as(refims)
            for tar, ma, im in intermediates["bg_supervise"]:
                bg_suloss += ((tar - im).abs() * ma)[..., 10:-10, 10:-10].mean()

            final_loss += (self.loss_weight["bg_supervision"] * bg_suloss)
            loss_dict["bg_su"] = bg_suloss.detach()

        if "net_warmup" in self.loss_weight.keys():
            raise DeprecationWarning("PipelineV2::The net_warmup is deprecated")

        if "net_std" in self.loss_weight.keys():
            raise DeprecationWarning("PipelineV2::The net_std is deprecated")
            # disp = net_out.reshape(batchsz * (framenum - 2), 4, -1)[:, 1]
            # disp_std = torch.std(disp, dim=-1, unbiased=False).mean()
            # net_std = - disp_std + 0.5
            # final_loss += (net_std * self.loss_weight["net_std"])
            # loss_dict["net_std"] = net_std.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class PipelineV2SV(PipelineV2):
    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__(models, cfg)

    @torch.no_grad()
    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        self.eval()
        refims, tarims, refextrins, tarextrin, intrin, pt2ds, ptzs_gts = args
        batchsz, framenum, cnl, heiori, widori = refims.shape
        layernum = self.mpimodel.num_layers
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        intermediate = {"disp0": None, "disp_warp": [], "imbg": None}
        mpi_list, net_list, upmask_list, disp_list = self.forward_multiframes(refims, None, intermediate)
        disp0 = intermediate["disp0"]
        disp_warp_list = intermediate["disp_warp"]
        imbg = intermediate["imbg"]

        ptdisp_es0 = torchf.grid_sample(disp0.unsqueeze(1), pt2ds[:, 0:1], align_corners=True).reshape(batchsz, -1)
        scale = torch.exp(torch.log(ptdisp_es0 * ptzs_gts[:, 0]).mean(dim=-1, keepdim=True))
        depth = depth * scale.reshape(-1, 1)

        # render target view
        tarviews = [
            render_newview(
                mpi=mpi_list[i],
                srcextrin=refextrins[:, i + 1],
                tarextrin=tarextrin,
                srcintrin=intrin,
                tarintrin=intrin,
                depths=depth,
                ret_mask=False
            )
            for i in range(len(mpi_list))
        ]

        val_dict = {}
        # Photometric metric====================
        tarviews = torch.stack(tarviews, dim=1)
        targts = tarims.unsqueeze(1).expand_as(tarviews)
        metrics = ["psnr", "ssim"]
        for metric in metrics:
            error = compute_img_metric(
                targts.reshape(-1, 3, heiori, widori),
                tarviews.reshape(-1, 3, heiori, widori),
                metric=metric
            )
            val_dict[f"val_{metric}"] = error

        disp_es = [
            torchf.grid_sample(disp_list[i].unsqueeze(1),
                               pt2ds[:, i + 1:i + 2],
                               align_corners=True).reshape(batchsz, -1)
            for i in range(len(disp_list))
        ]
        disp_es = torch.stack(disp_es, dim=1)
        z_gt = ptzs_gts[:, 1:-1]
        thresh = np.log(1.25) ** 2
        diff = (disp_es / scale.reshape(batchsz, -1, 1) - 1 / z_gt).abs().mean()
        diff_scale = torch.log(disp_es * z_gt / scale.reshape(-1, 1))
        diff_scale = torch.pow(diff_scale, 2)
        inliner_pct = (diff_scale < thresh).type(torch.float32).sum() / diff_scale.nelement()
        diff_scale = diff_scale.mean()
        val_dict["val_d_mae"] = diff
        val_dict["val_d_msle"] = diff_scale
        val_dict["val_d_goodpct"] = inliner_pct

        # depth temporal consistency loss==============
        temporal = torch.tensor(0.).type_as(refims)
        temporal_scale = torch.tensor(0.).type_as(refims)
        for disp, disp_warp in zip(disp_list, disp_warp_list):
            temporal += ((torch.abs(disp.unsqueeze(1) - disp_warp)) / scale.reshape(batchsz, 1, 1, 1)).mean()
            # print(f"disp_warp: min {disp_warp.min()}, max {disp_warp.max()}")
            # print(f"disp_warp: min {disp.min()}, max {disp.max()}")
            disp_warp = torch.max(disp_warp, torch.tensor(0.000001).type_as(disp_warp))
            disp = torch.max(disp, torch.tensor(0.000001).type_as(disp))
            temporal_scale += torch.pow(torch.log(disp / disp_warp), 2).mean()
        temporal = temporal / len(disp_list)
        temporal_scale /= len(disp_list)
        val_dict["val_t_mae"] = temporal
        val_dict["val_t_msle"] = temporal_scale

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            val_dict["vis_disp"] = draw_dense_disp(disp_list[1], 1)
            val_dict["vis_bgimg"] = (imbg[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_img"] = (refims[0, -2] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        self.train()
        args = [_t.cuda() for _t in args]
        refims, tarims, refextrins, tarextrin, intrin, pt2ds, ptzs_gts = args
        batchsz, framenum, _, heiori, widori = refims.shape
        layernum = self.mpimodel.num_layers
        # remove first frame's gt
        refextrins = refextrins[:, 1:]
        pt2ds = pt2ds[:, 1:]
        ptzs_gts = ptzs_gts[:, 1:]
        step = kwargs["step"]
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        intermediates = {
            "flows_b_upmask": None,
            "flows_f_upmask": None,
            "flows_f": None,
            "flows_b": None,
            "bgflow_fgflow": [],
            "net_up": [],
            "bg_supervise": []
        }
        mpi_out, net_out, upmask_out, disp_out = self.forward_multiframes(refims, step, intermediates)
        flows_f_upmask = intermediates["flows_f_upmask"]
        flows_f = intermediates["flows_f"]
        flows_b_upmask = intermediates["flows_b_upmask"]
        flows_b = intermediates["flows_b"]
        if self.depth_loss_mode == "fine":  # in original resolution
            ptdis_es = [
                torchf.grid_sample(disp_out[i].unsqueeze(1), pt2ds[:, i:i + 1], align_corners=True).reshape(batchsz, -1)
                for i in range(len(disp_out))
            ]
        elif self.depth_loss_mode == "coarse":
            alpha_coarse = [self.net2alpha(net_coarse, None) for net_coarse in net_out]
            disp_coarse = [estimate_disparity_torch(alpha.unsqueeze(2), depth) for alpha in alpha_coarse]
            ptdis_es = [
                torchf.grid_sample(disp_coarse[i].unsqueeze(1),
                                   pt2ds[:, i:i + 1], align_corners=True, mode='nearest').reshape(batchsz, -1)
                for i in range(len(disp_coarse))
            ]
        elif self.depth_loss_mode == "direct_fine":
            raise NotImplementedError(f"not implemented")
        elif self.depth_loss_mode == "direct_coarse":
            raise NotImplementedError(f"not implemented")
        else:
            raise RuntimeError(f"PipelineV2SV::depth_loss_mode = {self.depth_loss_mode} not recognized")

        # currently use first frame to compute scale and shift
        if self.scale_mode == "first":
            scale = torch.exp(torch.log(ptdis_es[0] * ptzs_gts[:, 0]).mean(dim=-1, keepdim=True))
            scale = scale * self.scale_scaling + SCALE_EPS
        elif self.scale_mode == "mean":
            scale = [
                torch.exp(torch.log(ptdis_es[i] * ptzs_gts[:, i]).mean(dim=-1, keepdim=True))
                for i in range(framenum - 2)
            ]
            scale = torch.stack(scale, dim=1).mean(dim=1)
            scale = scale * self.scale_scaling + SCALE_EPS
        elif self.scale_mode == "random":
            i = np.random.randint(0, framenum - 2)
            scale = torch.exp(torch.log(ptdis_es[i] * ptzs_gts[:, i]).mean(dim=-1, keepdim=True))
            scale = scale * self.scale_scaling + SCALE_EPS
        elif self.scale_mode == "fix":
            scale = torch.exp(torch.log(0.7 *
                                        torch.kthvalue(ptzs_gts[:, 0],
                                                       int(0.05 * ptzs_gts[:, 0].nelement()))[0]))
        elif self.scale_mode == "adaptive":
            i = np.random.randint(0, framenum - 2)
            scale_tar = torch.exp(torch.log(0.7 *
                                            torch.kthvalue(ptzs_gts[:, 0],
                                                           int(0.05 * ptzs_gts[:, 0].nelement()))[0]))
            scale = torch.exp(torch.log(ptdis_es[i] * ptzs_gts[:, i]).mean(dim=-1, keepdim=True))
            scale = scale * 0.6 + scale_tar * 0.4
        else:
            raise RuntimeError(f"PipelineSVV2::unrecognized scale mode {self.scale_mode}")

        depth = depth * scale.reshape(-1, 1)
        # render target view
        tarviews = [
            render_newview(
                mpi=mpi_out[i],
                srcextrin=refextrins[:, i],
                tarextrin=tarextrin,
                srcintrin=intrin,
                tarintrin=intrin,
                depths=depth,
                ret_mask=False
            )
            for i in range(len(mpi_out))
        ]

        return self.compute_loss(
            True, refims, tarims,
            net_out, upmask_out, mpi_out, tarviews, disp_out, scale,
            flows_f, flows_b, flows_f_upmask, flows_b_upmask,
            None, None,
            ptdis_es, ptzs_gts,
            intermediates, step
        )


class PipelineJoint(nn.Module):
    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__()
        self.pipeline_disp = PipelineV2(models, cfg)
        cfg["loss_weights"]["depth_loss"] /= 5
        self.pipeline_sv = PipelineV2SV(models, cfg)
        del self.pipeline_sv.flow_estim
        del self.pipeline_sv.photo_loss
        self.pipeline_sv.flow_estim = self.pipeline_disp.flow_estim
        self.pipeline_sv.photo_loss = self.pipeline_disp.photo_loss

    def valid_forward(self, *args, **kwargs):
        if len(args) == 5:
            return self.pipeline_disp.valid_forward(*args, **kwargs)
        elif len(args) == 7:
            return self.pipeline_sv.valid_forward(*args, **kwargs)
        else:
            raise RuntimeError()

    def forward(self, *args, **kwargs):
        if len(args) == 5:
            return self.pipeline_disp.forward(*args, **kwargs)
        elif len(args) == 7:
            return self.pipeline_sv.forward(*args, **kwargs)
        else:
            raise RuntimeError()


class PipelineV3(PipelineV2):
    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__(models, cfg)
        self.img_window = []
        self.flow_cache = {}
        self.kf0_cache = KeyFrameInfo()
        self.kf1_cache = KeyFrameInfo()
        self.kf_lock = False

    def estimate_flow(self, idx0, idx1):
        with torch.no_grad():
            if idx0 < 0:
                idx0 = len(self.img_window) + idx0
            if idx1 < 0:
                idx1 = len(self.img_window) + idx1
            if idx0 == idx1:
                return torch.zeros_like(self.offset)
            key = f"{idx0}_{idx1}"
            if key not in self.flow_cache.keys():
                flow = self.flow_estim(self.img_window[idx0], self.img_window[idx1])
                self.flow_cache[key] = flow
                return flow
            else:
                return self.flow_cache[key]

    def update_window(self, img: torch.Tensor):
        """
        update img_window, and detect keyframe
        return: is keyframe detected
        """
        if self.kf_lock:
            print("start new key frame segment")
            self.img_window = self.img_window[-1:]
            self.flow_cache.clear()
            self.kf0_cache.copyfrom(self.kf1_cache)
            self.kf1_cache.update(None, None, None)
            self.kf_lock = False

        batchsz, cnl, hei, wid = img.shape
        self.prepareoffset(wid, hei)
        self.img_window.append(img)
        if len(self.img_window) < 2:
            return False

        # detect keyframes
        flowf = self.estimate_flow(0, len(self.img_window) - 1)
        flowb = self.estimate_flow(len(self.img_window) - 1, 0)

        occmask = warp_flow(flowb, flowf, offset=self.offset_bhw2_d8) + flowf
        occmask = (torch.norm(occmask, dim=1)).type(torch.float32)
        if occmask.max() > 0.1 * max(hei, wid) / 8:
            self.kf_lock = True
            return True
        else:
            return False

    def infer_forward(self, idx, ret_cfg: str):
        """
        estimate the mpi in index idx (self.img_window)
        """
        self.eval()
        if not self.kf_lock:
            print("Warning! haven't lock key frames, force produce result based on current status")
            self.kf_lock = True

        if idx == len(self.img_window) - 1:
            return None
        with torch.no_grad():
            depth = make_depths(self.mpimodel.num_layers).type_as(self.img_window[0]).unsqueeze(0).repeat(1, 1)
            # keyframe need special deal
            if self.kf0_cache.isinvalid():
                flow_list = [self.estimate_flow(0, i) for i in [len(self.img_window) // 2, -1]]
                img_list = [self.img_window[i] for i in [len(self.img_window) // 2, -1]]
                net_warp = None
                for i in range(3):
                    alpha, net, upmask = self.forwardmpi(None, self.img_window[0], flow_list, net_warp=net_warp)
                disp, blend_weight = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)
                disp = disp.unsqueeze(1)
                imbg = self.forwardaf(disp, net, upmask, flow_list, img_list)
                mpi = alpha2mpi(alpha, self.img_window[0], imbg, blend_weight=blend_weight)
                self.kf0_cache.update(net, upmask, mpi)

            if self.kf1_cache.isinvalid():
                net_warp = self.flowfwarp_warp(self.estimate_flow(0, -1), self.estimate_flow(-1, 0),
                                               self.kf0_cache.net)
                flow_list = [self.estimate_flow(-1, i) for i in [0, -len(self.img_window) // 2]]
                img_list = [self.img_window[i] for i in [0, -len(self.img_window) // 2]]
                alpha, net, upmask = self.forwardmpi(None, self.img_window[-1], flow_list, net_warp)
                disp, blend_weight = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)
                imbg = self.forwardaf(disp, net, upmask, flow_list, img_list)
                mpi = alpha2mpi(alpha, self.img_window[-1], imbg, blend_weight=blend_weight)
                self.kf1_cache.update(net, upmask, mpi)

            if idx == 0:
                return self.kf0_cache.mpi

            net_warp0 = self.flowfwarp_warp(
                flowf=self.estimate_flow(0, idx),
                flowb=self.estimate_flow(idx, 0),
                content=self.kf0_cache.net
            )
            weight0 = (len(self.img_window) - 1 - idx) / (len(self.img_window) - 1)
            net_warp1 = self.flowfwarp_warp(
                flowf=self.estimate_flow(-1, idx),
                flowb=self.estimate_flow(idx, -1),
                content=self.kf1_cache.net
            )
            weight1 = idx / (len(self.img_window) - 1)
            net = net_warp0 * weight0 + net_warp1 * weight1
            flow_list = [self.estimate_flow(idx, 0), self.estimate_flow(idx, -1)]
            img_list = [self.img_window[0], self.img_window[-1]]
            _, _, upmask = self.forwardmpi(None, self.img_window[idx], flow_list, net)
            alpha = self.net2alpha(net, upmask)

            disp, blend_weight = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)
            imbg = self.forwardaf(disp, net, upmask, flow_list, img_list)
            mpi = alpha2mpi(alpha, self.img_window[idx], imbg, blend_weight=blend_weight)
            return mpi


class PipelineFiltering(PipelineV2):
    """
    mirror pading the first and last frames
    """

    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__(models, cfg)
        self.img_window = deque()  # level0 material
        self.net_window = deque()  # level1 material
        self.upmask_window = deque()
        self.flow_cache = dict()

        self.flowskip = cfg.pop("flowskip", 2)  # level0 bootstrap len // 2
        self.winsz = cfg.pop("winsz", 7)  # level0 bootstrap len
        self.mididx = self.winsz // 2
        self.defer_num = self.mididx + self.flowskip
        self.filter_weight = torch.tensor([0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.1])

    def clear(self):
        self.img_window.clear()
        self.net_window.clear()
        self.upmask_window.clear()
        self.flow_cache.clear()

    def estimate_flow(self, idx0, idx1, neednet=False):
        with torch.no_grad():
            if idx0 < 0:
                idx0 = len(self.img_window) + idx0
                assert idx0 >= 0
            if idx1 < 0:
                idx1 = len(self.img_window) + idx1
                assert idx1 >= 0

            key = (idx0, idx1)
            if key not in self.flow_cache.keys() or neednet:
                flowdown, flowmask, flownet = self.flow_estim(self.img_window[idx0],
                                                              self.img_window[idx1],
                                                              ret_upmask=True)
                self.flow_cache[key] = (flowdown, upsample_flow(flowdown, flowmask))
                return self.flow_cache[key], flownet
            else:
                return self.flow_cache[key], None

    def _1update_one_net(self, net, upmask, ret_cfg=""):  # update level1 material / pipeline1 action / update net
        self.net_window.append(net)
        self.upmask_window.append(upmask)
        if len(self.net_window) > self.winsz:  # free memory in last level
            self.img_window.popleft()
            self.net_window.popleft()
            self.upmask_window.popleft()
            # update flow_cache
            self.flow_cache = {(idx0 - 1, idx1 - 1): v
                               for (idx0, idx1), v in self.flow_cache.items()
                               if idx0 > 0 and idx1 > 0}

        if len(self.net_window) <= self.mididx:  # bootstrap in level1
            return None
        else:
            mididx = len(self.net_window) - self.mididx - 1
            net_warp_list = []
            net_warp_weight = []
            for i in range(mididx - self.winsz // 2, mididx + self.winsz // 2 + 1):
                i = -i if i < 0 else i  # pading mode in level0

                if i == mididx:
                    net_warp_list.append(self.net_window[i])
                    net_warp_weight.append(torch.ones_like(net_warp_weight[-1]))
                else:
                    net_warp, occ_norm = self.warp_and_occ(self.estimate_flow(i, mididx)[0][0],
                                                           self.estimate_flow(mididx, i)[0][0],
                                                           self.net_window[i])
                    net_warp_weight.append((occ_norm < 0.25).type(torch.float32))
                    net_warp_list.append(net_warp)
            net_warps = torch.cat(net_warp_list, dim=0)
            weights = torch.cat(net_warp_weight, dim=0)
            self.filter_weight = self.filter_weight.type_as(net_warps).reshape(-1, 1, 1, 1)
            weights = self.filter_weight * weights
            net_filtered = (weights * net_warps).sum(dim=0, keepdim=True) / weights.sum(dim=0, keepdim=True)
            upmask = self.upmask_window[mididx]
            alpha, net_up = self.net2alpha(net_filtered, upmask, ret_netup=True)

            depth = make_depths(self.mpimodel.num_layers).type_as(self.img_window[0]).unsqueeze(0).repeat(1, 1)
            disp, bw = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)

            if "selfonly" in ret_cfg:
                flow_list = [self.estimate_flow(mididx, mididx)[0][1]]
                flowb_list = flow_list
                img_list = [self.img_window[mididx]]
            else:
                idx0, idx1 = mididx - 1, mididx + 1
                idx0 = mididx if idx0 < 0 else idx0
                flow_list = [self.estimate_flow(mididx, idx0)[0][1],
                             self.estimate_flow(mididx, idx1)[0][1]]
                flowb_list = [self.estimate_flow(idx0, mididx)[0][1],
                              self.estimate_flow(idx1, mididx)[0][1]]
                img_list = [self.img_window[idx0], self.img_window[idx1]]

            flownet_list = [None] * len(flow_list)
            # [self.estimate_flow(self.mididx, idx0)[FLOWNET_IDX],
            # self.estimate_flow(self.mididx, idx1)[FLOWNET_IDX]]
            flowmask_list = [None] * len(flow_list)
            # [self.estimate_flow(self.mididx, idx0)[FLOWMASK_IDX],
            #  self.estimate_flow(self.mididx, idx1)[FLOWMASK_IDX]]
            imbg = self.forwardaf(disp, net_filtered, upmask, self.img_window[mididx],
                                  flow_list, img_list, flownet_list, flowmask_list,
                                  flowb_list, flownet_list, flowmask_list)
            return net_up, alpha, self.img_window[mididx], imbg, bw

    def _0update_one_frame(self, img, ret_cfg=""):  # update level0 material / pipeline0 action / update net
        """return whether is ready for output one frame"""
        batchsz, cnl, hei, wid = img.shape
        self.prepareoffset(wid, hei)
        self.img_window.append(img)
        if len(self.img_window) <= self.flowskip:  # bootstrap in level0
            return None
        else:
            idx0 = len(self.img_window) - 1 - self.flowskip
            idx1 = len(self.img_window) - 1 - 2 * self.flowskip
            if idx1 < 0:
                idx1 = - idx1  # pading mode in level0
            idx2 = len(self.img_window) - 1
            flow_net_list = [self.estimate_flow(idx0, idx1, True)[-1],
                             self.estimate_flow(idx0, idx2, True)[-1]]
            alpha, net, upmask = self.forwardmpi(None, self.img_window[idx0], flow_net_list)
            return net, upmask

    def infer_forward(self, img, ret_cfg: str):
        """
        estimate the mpi in index idx (self.img_window)
        """
        self.eval()

        ret0 = self._0update_one_frame(img, ret_cfg)
        if ret0 is not None:
            ret1 = self._1update_one_net(*ret0, ret_cfg)
            if ret1 is not None:
                net_up, alpha, imfg, imbg, bw = ret1
                if "hardbw" in ret_cfg:
                    order = 6
                    bw = torch.where(bw < 0.5, 0.5 * (2 * bw) ** order, 1 - 0.5 * (2 - 2 * bw) ** order)
                mpi = alpha2mpi(alpha, imfg, imbg, blend_weight=bw)
                if "ret_net" in ret_cfg:
                    ret_net = torch.cat([net_up, self.img_window[self.mididx], imbg], dim=1)
                    return mpi, ret_net
                else:
                    return mpi
        return None

    def infer_multiple(self, imgs: list, ret_cfg="", device=torch.device('cpu')):
        self.eval()
        mpis = []
        for img in imgs:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.cuda()
            mpi = self.infer_forward(img, ret_cfg)
            if mpi is not None:
                mpis.append(mpi.to(device))

        pad_img = [self.img_window[-i] for i in range(-self.defer_num, 0)][::-1]
        for img in pad_img:
            mpi = self.infer_forward(img, ret_cfg)
            if mpi is not None:
                mpis.append(mpi.to(device))
        self.clear()
        return mpis


class KeyFrameInfo:
    def __init__(self):
        self.net = None
        self.upmask = None
        self.mpi = None

    def update(self, net, upmask, mpi):
        self.net, self.upmask, self.mpi = net, upmask, mpi

    def copyfrom(self, other: "KeyFrameInfo"):
        if other is None:
            self.net = None
            self.upmask = None
            self.mpi = None
        self.net = other.net
        self.upmask = other.upmask
        self.mpi = other.mpi

    def isinvalid(self):
        return self.net is None
