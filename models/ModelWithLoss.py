import torch
import torch.nn as nn
import torch.nn.functional as torchf
from typing import List, Callable
import numpy as np
import sys
from .loss_utils import *
from torchvision.transforms import ToTensor, Resize
from .mpi_utils import *
from .geo_utils import RandomAffine
import cv2
from .mpi_network import *
from .mpv_network import *
from .mpifuse_network import *
from .mpi_flow_network import *
from .flow_utils import *
from collections import deque
from .img_utils import *
from scipy.ndimage import gaussian_filter1d

'''
This class is a model container provide interface between
                                            1. dataset <-> model
                                            2. model <-> loss
                                            
The following methods should be implemented:
# if the pipeline needs to be trained
1. forward(self, *args, **kwargs): forward method, args for training data, kwargs for settings
2. valid_forward(self, *args, **kwargs): validate during training. args for data, kwargs for settings

# if the pipeline needs to be runed on the real world video:
1. infer_forward(self, img, ret_cfg): each time get an image, return result or None if several frame needed (pipelined)

# if the pipelien needs to be evaluated with main_evaluation_xxx.py:
1. infer_multiple(self, img_list, ret_cfg, *): get several images and return a result with same length
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
        self.model.eval()
        with torch.no_grad():
            if im.dim() == 3:
                im = im.unsqueeze(0)
            im = im.cuda()
            netout = self.model(im)
            mpi = netout2mpi(netout, im)
        return mpi

    def infer_multiple(self, imgs: list, ret_cfg="", device=torch.device('cpu')):
        self.eval()
        # ret_cfg += "ret_net"

        mpis = []
        # disps = []
        for img in imgs:
            mpi = self.infer_forward(img, ret_cfg)
            mpis.append(mpi.to(device))
        return mpis

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
        self.upmask_magaware = self.loss_weight.pop("upmask_magaware", True)
        self.upmask_lr = self.loss_weight.pop("upmask_lr", True)
        self.aflow_contextaware = self.loss_weight.pop("aflow_contextaware", False)
        self.aflow_selfsu = self.loss_weight.pop("aflow_selfsu", False)

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
            milestones=[1e3, 2e3],
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
        naive forward
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
            intermediates["blendweight"] = None
            alpha, net, upmask = self.forwardmpi(
                intermediates,
                refims[:, frameidx],
                [flows_b_net[:, frameidx - 1], flows_f_net[:, frameidx]],
                None
            )
            blend_weight = intermediates.pop("blendweight")

            netups = intermediates.get("net_up", [])
            net_up = netups[-1] if len(netups) > 0 else learned_upsample(net, upmask)
            if isinstance(self.mpimodel, MPILDI_AB_alpha):
                disp = estimate_disparity_torch(alpha.unsqueeze(2), depth)
            else:
                disp = self.net2disparity(net_up).squeeze(1)

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
        # temporal = torch.tensor(0.).type_as(refims)
        # temporal_scale = torch.tensor(0.).type_as(refims)
        # for disp, disp_warp in zip(disp_list, disp_warp_list):
        #     disp_warp /= scale.reshape(batchsz, 1, 1, 1)
        #     temporal += (torch.abs(disp.unsqueeze(1) - disp_warp)).mean()
        #     # print(f"disp_warp: min {disp_warp.min()}, max {disp_warp.max()}")
        #     # print(f"disp_warp: min {disp.min()}, max {disp.max()}")
        #     disp_warp = torch.max(disp_warp, torch.tensor(0.000001).type_as(disp_warp))
        #     disp = torch.max(disp, torch.tensor(0.000001).type_as(disp))
        #     temporal_scale += torch.pow(torch.log(disp / disp_warp), 2).mean()
        # temporal /= len(disp_list)
        # temporal_scale /= len(disp_list)
        # val_dict["val_t_mae"] = temporal
        # val_dict["val_t_msle"] = temporal_scale

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            netvis = net_list[1][:, :4] * torch.tensor([1, 1, 10, 10.]).type_as(refims).reshape(1, 4, 1, 1)
            netvis = torch.cat(netvis.split(1, dim=1), dim=-1)
            val_dict["vis_net"] = draw_dense_disp(netvis[:, 0], 1)
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

        alpha, net_up, blendweight = self.net2alpha(net, upmask, ret_bw=True)
        if tmpcfg is not None and "net_up" in tmpcfg.keys():
            tmpcfg["net_up"].append(net_up)
        if tmpcfg is not None and "blendweight" in tmpcfg.keys():
            tmpcfg["blendweight"] = blendweight
        return alpha, net, upmask

    @staticmethod
    def net2disparity(net, denorm=5*20):
        assert net.shape[1] == 4
        fg, bg, fgt, _ = net.split(1, dim=1)
        _alpha = torch.exp(-fgt * denorm)
        return fg * (- _alpha + 1) + bg * _alpha

    def net2alpha(self, net, upmask, ret_bw=False):
        layernum = self.mpimodel.num_layers
        if upmask is None:
            netout = net
        else:
            netout = learned_upsample(net, upmask)
        batchsz, outcnl, hei, wid = netout.shape
        ones = torch.ones([batchsz, 1, hei, wid]).type_as(net)
        zeros = torch.zeros([batchsz, 1, hei, wid]).type_as(net)

        if isinstance(self.mpimodel, (MPI_V6Nset, MPI_AB_up)):
            sigma, disp, thick = torch.split(netout, self.mpimodel.num_set, dim=1)
            x = torch.reciprocal(make_depths(layernum)).reshape(1, 1, layernum, 1, 1).type_as(netout)
            disp_min, disp_max = 1 / default_d_far, 1 / default_d_near
            disp = disp * (disp_max - disp_min) + disp_min
            alpha = self.nsets2alpha(x,
                                     disp.unsqueeze(2),
                                     thick.unsqueeze(2),
                                     sigma.unsqueeze(2))

            # compute blend weight
            fg, bg = disp.split(1, dim=1)
            tfg, tbg = thick.split(1, dim=1)
            turning = ((fg - tfg) + bg) / 2.
            x_o = x[:, 0]
            blend_weight = torch.relu(x_o[:, 1:] - torch.max(turning, x_o[:, :-1])) \
                           / (x_o[0, 2, 0, 0] - x_o[0, 1, 0, 0])
            blend_weight = torch.cat([zeros, blend_weight], dim=1)
            blend_weight = blend_weight.clamp(0, 1)
        elif isinstance(self.mpimodel, (MPI_LDI, MPI_LDI_res, MPI_LDIdiv, MPI_LDIbig, MPILDI_AB_nonet)):
            disp, thick = torch.split(netout, self.mpimodel.num_set, dim=1)
            x = torch.reciprocal(make_depths(layernum)).reshape(1, 1, layernum, 1, 1).type_as(netout)
            disp_min, disp_max = 1 / default_d_far, 1 / default_d_near
            disp = disp * (disp_max - disp_min) + disp_min
            thick[:, 1] = 1
            # if not self.training:
            #     thick[:, 0] = torch.where(thick[:, 0] < 0.2 / 31, zeros, thick[:, 0])
            netout = torch.cat([disp, thick], dim=1)
            alpha = self.nsets2alpha(x,
                                     disp.unsqueeze(2),
                                     thick.unsqueeze(2),
                                     5)
            # compute blend weight
            # fg, bg = disp.split(1, dim=1)
            # tfg, tbg = thick.split(1, dim=1)
            # turning = ((fg - tfg) + bg) / 2.
            # x_o = x[:, 0]
            # blend_weight = torch.relu(x_o[:, 1:] - torch.max(turning, x_o[:, :-1])) \
            #                / (x_o[0, 2, 0, 0] - x_o[0, 1, 0, 0])
            # blend_weight = torch.cat([zeros, blend_weight], dim=1)
            # blend_weight = blend_weight.clamp(0, 1)
            # compute blend weight
            fg, bg = disp.split(1, dim=1)
            tfg, tbg = thick.split(1, dim=1)

            # scheme1
            turning = (dilate(dilate(fg)) + bg) / 2.
            scaling = - torch.exp(-dilate(dilate(tfg)) * 100) + 1

            # scheme2
            # turning = (fg + bg) / 2.
            # scaling = - torch.exp(-tfg * 100) + 1

            x_o = x[:, 0]
            blend_weight = torch.relu(torch.min(turning, x_o[:, 1:]) - x_o[:, :-1]) \
                           / (x_o[0, 2, 0, 0] - x_o[0, 1, 0, 0])
            blend_weight = - blend_weight * scaling + 1
            blend_weight = torch.cat([zeros, blend_weight], dim=1)
            blend_weight = blend_weight.clamp(0, 1)
        elif isinstance(self.mpimodel, MPILDI_AB_alpha):
            alpha = netout
            blend_weight = None
        else:
            raise RuntimeError(f"Pipelinev2::incompatiable mpimodel: {type(self.mpimodel)}")

        if not self.training:
            alpha = dilate(erode(alpha))
        alpha = torch.cat([ones, alpha], dim=1)
        alpha = torch.clamp(alpha, 0, 1)

        if ret_bw:
            return alpha, netout, blend_weight
        else:
            return alpha, netout

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
    def set2alpha(x, ds, th, sig):
        th = torch.min(ds - 1. / default_d_far, th)
        t = x - ds + th
        t = torch.relu(t)
        t = torch.min(t, th)
        dt = t[:, 1:] - t[:, :-1]
        return -torch.exp(-dt * sig * 20) + 1

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
        # =========================
        # new pipeline
        # =========================
        if isinstance(self.afmodel, (AFNet, InPaintNet, AFNetBig)):
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
                    rangemap0 = forward_scatter_withweight(flowb, occ1, occ1, self.offset)
                    rangemap0 = erode(rangemap0)
                    flowbg = forward_scatter_withweight(flowb, -flowb, occ1, self.offset)
                    for i in range(5):
                        flowbg = media_filter(flowbg)
                        flowbg = -warp_flow(flowb, flowbg, offset=self.offset_bhw2)
                    imbg = warp_flow(img, flowbg, offset=self.offset_bhw2)
                    bglist.append(imbg)
                    malist.append(rangemap0)

            eps = 0.3
            net_up = learned_upsample(net, upmask) if upmask is not None else net
            if self.aflow_contextaware:
                with torch.no_grad():
                    fg = net[:, 0:1].detach()
                    fge = erode(erode(fg))
                    diff = learned_upsample(fg - fge, upmask.detach())
                    context = dilate(diff, dilate=2) < 4. / 32.
                    for ma in malist:
                        context *= (ma < eps)
                    imgin = curim * context
            else:
                imgin = curim

            if isinstance(self.afmodel, AFNetBig):
                # sort based on the number of pixel in mask from small to big
                pixelnum = [float((ma > eps).sum()) for ma in malist]
                idxs = np.argsort(pixelnum)
                malist = [malist[i] for i in idxs]
                bglist = [bglist[i] for i in idxs]

                frame = curim.clone()

                accmask_yes = malist[0] > eps

                mask_maybe = dilate(dilate(malist[0]), dilate=2) > eps
                mask = mask_maybe.expand(-1, 3, -1, -1)
                frame[mask] = bglist[0][mask]
                for bgctx, mactx in zip(bglist[1:], malist[1:]):
                    mask_maybe = dilate(dilate(mactx), dilate=2) > eps
                    mask = torch.logical_and(torch.logical_not(accmask_yes), mask_maybe).expand(-1, 3, -1, -1)
                    frame[mask] = bgctx[mask]

                    accmask_yes = torch.logical_or(accmask_yes, mactx > 0.1)
                accmask_yes = accmask_yes.type(torch.float32)
                frames_list = [torch.cat([frame, accmask_yes.type_as(frame)], dim=1)]
                if self.afmodel.framenum == 0:
                    frames_list = []
            elif self.afmodel.hasmask:
                frames_list = [torch.cat([bg, ma], dim=1) for bg, ma in zip(bglist, malist)]
            else:
                frames_list = [bg * (ma > eps) for bg, ma in zip(bglist, malist)]

            # frames_list = [torch.cat([curim.clone(), torch.zeros_like(accmask_yes)], dim=1)]

            if self.training and self.aflow_selfsu:  # should ensure that context aware is True and is AFNetBig
                hei, wid = ma.shape[-2:]
                rand_hei = np.random.randint(hei // 10, hei // 5)
                rand_wid = np.random.randint(wid // 10, wid // 5)
                rand_top = np.random.randint(hei // 10, hei - hei // 10)
                rand_left = np.random.randint(wid // 10, wid - wid // 10)

                ma_gen = torch.zeros_like(context)
                ma_gen[..., rand_top: rand_top + rand_hei, rand_left: rand_left + rand_wid] = True
                ma_gen = torch.logical_and(ma_gen, context)
                frame_supervision = curim
                imgin[ma_gen.expand(-1, 3, -1, -1)] = 0

            net_in = net_up[:, :2] if self.afmodel.netcnl == 2 else net_up
            net_in = net_in.detach()

            inp = torch.cat([net_in, imgin] + frames_list, dim=1)

            bg = self.afmodel(inp)
            # bg = frame
            if intermediate is not None and "bg_supervise" in intermediate.keys():
                intermediate["bg_supervise"].append([bg, erode(malist[0]), bglist[0]])
                intermediate["bg_supervise"].append([bg, erode(malist[1]), bglist[1]])
                if self.aflow_selfsu:
                    intermediate["bg_supervise"].append([bg, ma_gen, frame_supervision])
            return bg

        # =========================
        # old pipeline
        # =========================
        masks, bgs = [], []
        # lastmask = None
        for flow, img, flow_net, flow_upmask in zip(flows_list, imgs_list, flow_nets_list, flow_upmasks_list):
            if isinstance(self.afmodel, AFNet_HR_netflowin):
                flow = upsample_flow(flow, flow_upmask) if flow_upmask is not None else flow
                net_up = learned_upsample(net, upmask) if upmask is not None else net
                bgflow, mask = self.afmodel(net_up, flow)
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
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)
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
            if "net_smth_loss" in self.loss_weight.keys():
                if isinstance(self.mpimodel, MPILDI_AB_alpha):
                    net = disp_out[mpiframeidx].unsqueeze(1)
                    net_smth_loss = smooth_grad(net, refims[:, mpiframeidx + 1])
                    net_smth_loss = net_smth_loss.mean()
                elif isinstance(self.mpimodel, (MPI_LDI_res, MPI_LDI, MPI_LDIdiv, MPI_LDIbig, MPILDI_AB_nonet)):
                    fg_up, bg_up = intermediates["net_up"][mpiframeidx][:, :2].split(1, dim=1)
                    fg_down, bg_down = net_out[mpiframeidx][:, :2].split(1, dim=1)
                    guidence_up = refims[:, mpiframeidx + 1]
                    guidence_down = torchf.interpolate(guidence_up, (heiori // 8, widori // 8))

                    fg_up_loss = smooth_grad(fg_up, guidence_up)
                    fg_down_loss = smooth_grad(fg_down, guidence_down, g_min=0.5) * 0.25
                    bg_down_loss = smooth_loss(bg_down, g_min=0.5)
                    net_smth_loss = fg_up_loss.mean() + \
                                    fg_down_loss.mean() + \
                                    bg_down_loss.mean() * 0.2
                else:
                    raise RuntimeError(f"net_smth_loss::{type(self.mpimodel)} not recognized")

                final_loss += (net_smth_loss * self.loss_weight["net_smth_loss"] * smth_schedule)
                if "netsmth" not in loss_dict.keys():
                    loss_dict["netsmth"] = net_smth_loss.detach()
                else:
                    loss_dict["netsmth"] += net_smth_loss.detach()

            if "singlescale_smth_loss" in self.loss_weight.keys():
                # for ablation usage only
                fg_up, bg_up = intermediates["net_up"][mpiframeidx][:, :2].split(1, dim=1)
                guidence_up = refims[:, mpiframeidx + 1]
                fg_up_loss = smooth_grad(fg_up, guidence_up)
                net_smth_loss = fg_up_loss.mean()
                final_loss += (net_smth_loss * self.loss_weight["singlescale_smth_loss"] * smth_schedule)
                if "netsmth" not in loss_dict.keys():
                    loss_dict["netsmth"] = net_smth_loss.detach()
                else:
                    loss_dict["netsmth"] += net_smth_loss.detach()

            fg_down, bg_down, fg_t_down, bg_t_down = net_out[mpiframeidx][:, :4].split(1, dim=1)
            if isinstance(self.mpimodel, MPILDI_AB_alpha):
                disp_down = None
            else:
                disp_down = self.net2disparity(net_out[mpiframeidx])
            # alpha_down, _ = self.net2alpha(net_out[mpiframeidx].detach(), None)
            # disp_down = estimate_disparity_torch(alpha_down.unsqueeze(2), depth).unsqueeze(1)
            if "disp_smth_loss" in self.loss_weight.keys():
                assert isinstance(self.mpimodel, (MPI_LDI_res, MPI_LDI, MPI_LDIdiv, MPI_LDIbig, MPILDI_AB_nonet))
                disp_up = disp_out[mpiframeidx]
                guidence_up = refims[:, mpiframeidx + 1]
                guidence_down = torchf.interpolate(guidence_up, (heiori // 8, widori // 8))

                up_loss = smooth_grad(disp_up, guidence_up)
                down_loss = smooth_grad(disp_down, guidence_down)
                smth_loss = up_loss.mean() + down_loss.mean()
                final_loss += (smth_loss * self.loss_weight["disp_smth_loss"] * smth_schedule)
                if "dispsmth" not in loss_dict.keys():
                    loss_dict["dispsmth"] = smth_loss.detach()
                else:
                    loss_dict["dispsmth"] += smth_loss.detach()

            if "net_prior0" in self.loss_weight.keys():
                prior = (torch.relu(bg_down - fg_down.detach())).mean()
                final_loss += (prior * self.loss_weight["net_prior0"])
                if "net_prior0" not in loss_dict.keys():
                    loss_dict["net_prior0"] = prior.detach()
                else:
                    loss_dict["net_prior0"] += prior.detach()

            if "net_prior1" in self.loss_weight.keys():
                with torch.no_grad():
                    disp_dilate = dilate(dilate(disp_down.detach()))
                    bgmask = (disp_dilate - disp_down.detach())

                prior = ((bg_down - disp_down.detach()).abs() * bgmask).mean()
                final_loss += (prior * self.loss_weight["net_prior1"])
                if "net_prior1" not in loss_dict.keys():
                    loss_dict["net_prior1"] = prior.detach()
                else:
                    loss_dict["net_prior1"] += prior.detach()

            if "net_prior2" in self.loss_weight.keys():
                disp_erode = erode(erode(disp_down.detach()))
                bgmask = (disp_down.detach() - disp_erode)

                prior = ((bg_down - disp_erode.detach()).abs() * bgmask).mean()
                final_loss += (prior * self.loss_weight["net_prior2"] * smth_schedule)
                if "net_prior2" not in loss_dict.keys():
                    loss_dict["net_prior2"] = prior.detach()
                else:
                    loss_dict["net_prior2"] += prior.detach()

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
                    weight = torch.clamp(torch.exp(-weight * 50), 0, 1)
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
                        weight = (weightx.abs() + weighty.abs()).sum(dim=1, keepdim=True)[..., 1:-1, 1:-1]
                        weight = torch.clamp(weight, 0, 1)
                else:
                    weight = 1

                diff = (upmask_out[i] - flowmask)[..., 1:-1, 1:-1].abs().mean(dim=1) * weight
                mask_loss += diff.mean()

                if not issparse and self.upmask_lr:
                    with torch.no_grad():
                        flow, flowmask, _ = self.flow_estim(refims[:, i + 1], tarims[:, i], ret_upmask=True)
                        if self.upmask_magaware:
                            weightx, weighty = gradient(flow)
                            weight = (weightx.abs() + weighty.abs()).sum(dim=1, keepdim=True)[..., 1:-1, 1:-1]
                            weight = torch.clamp(weight, 0, 1)
                        else:
                            weight = 1
                        diff = (upmask_out[i] - flowmask)[..., 1:-1, 1:-1].abs().mean(dim=1) * weight
                        mask_loss += diff.mean()

            mask_loss /= len(upmask_out)
            weight = self.loss_weight["mask_warmup"]
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
        # temporal = torch.tensor(0.).type_as(refims)
        # temporal_scale = torch.tensor(0.).type_as(refims)
        # for disp, disp_warp in zip(disp_list, disp_warp_list):
        #     temporal += ((torch.abs(disp.unsqueeze(1) - disp_warp)) / scale.reshape(batchsz, 1, 1, 1)).mean()
        #     # print(f"disp_warp: min {disp_warp.min()}, max {disp_warp.max()}")
        #     # print(f"disp_warp: min {disp.min()}, max {disp.max()}")
        #     disp_warp = torch.max(disp_warp, torch.tensor(0.000001).type_as(disp_warp))
        #     disp = torch.max(disp, torch.tensor(0.000001).type_as(disp))
        #     temporal_scale += torch.pow(torch.log(disp / disp_warp), 2).mean()
        # temporal = temporal / len(disp_list)
        # temporal_scale /= len(disp_list)
        # val_dict["val_t_mae"] = temporal
        # val_dict["val_t_msle"] = temporal_scale

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
        self.winsz = cfg.pop("winsz", 9)  # level0 bootstrap len
        self.mididx = self.winsz // 2
        self.forwardaf_idx = cfg.get("forwardaf_idx", [-self.mididx, -self.mididx + 2,  self.mididx - 2, self.mididx])
        self.defer_num = self.mididx + self.flowskip

        print(f"PipelineFiltering::winsz={self.winsz}, flowskip={self.flowskip}\n")

        x = np.zeros(self.winsz)
        x[self.mididx] = 1
        kernel = gaussian_filter1d(x, sigma=2)
        self.filter_weight = torch.tensor(kernel)

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
                    net_warp_weight.append(torch.ones_like(self.net_window[i][:, 0:1]))
                else:
                    net_warp, occ_norm = self.warp_and_occ(self.estimate_flow(i, mididx)[0][0],
                                                           self.estimate_flow(mididx, i)[0][0],
                                                           self.net_window[i])
                    # net_warp_weight.append((occ_norm < 0.25).type(torch.float32))
                    net_warp_weight.append(torch.exp(-occ_norm / 0.20))
                    net_warp_list.append(net_warp)
            net_warps = torch.cat(net_warp_list, dim=0)
            weights = torch.cat(net_warp_weight, dim=0)
            self.filter_weight = self.filter_weight.type_as(net_warps).reshape(-1, 1, 1, 1)
            weights = self.filter_weight * weights
            net_filtered = (weights * net_warps).sum(dim=0, keepdim=True) / weights.sum(dim=0, keepdim=True)
            upmask = self.upmask_window[mididx]
            alpha, net_up, bw = self.net2alpha(net_filtered, upmask, ret_bw=True)

            flow_list = []
            flowb_list = []
            img_list = []
            for idx in self.forwardaf_idx:
                idx = mididx + idx
                idx = -idx if idx < 0 else idx
                flow_list.append(self.estimate_flow(mididx, idx)[0][1])
                flowb_list.append(self.estimate_flow(idx, mididx)[0][1])
                img_list.append(self.img_window[idx])

            flownet_list = [None] * len(flow_list)
            # [self.estimate_flow(self.mididx, idx0)[FLOWNET_IDX],
            # self.estimate_flow(self.mididx, idx1)[FLOWNET_IDX]]
            flowmask_list = [None] * len(flow_list)
            # [self.estimate_flow(self.mididx, idx0)[FLOWMASK_IDX],
            #  self.estimate_flow(self.mididx, idx1)[FLOWMASK_IDX]]
            imbg = self.forwardaf(None, net_filtered, upmask, self.img_window[mididx],
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
                mpi = alpha2mpi(alpha, imfg, imbg, blend_weight=bw)

                if "ret_net" in ret_cfg:
                    ret_net = torch.cat([net_up, imfg, imbg], dim=1)
                    return mpi, ret_net
                else:
                    return mpi
        return None

    def infer_multiple(self, imgs: list, ret_cfg="", device=torch.device('cpu')):
        self.eval()
        # ret_cfg += "ret_net"

        mpis = []
        disps = []
        for img in imgs:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.cuda()
            mpi = self.infer_forward(img, ret_cfg + 'ret_net')
            if mpi is not None:
                mpi, net = mpi
                disps.append(self.net2disparity(net[:, :4]).to(device))
                mpis.append(mpi.to(device))

        pad_img = [self.img_window[-i] for i in range(-self.defer_num, 0)][::-1]
        for img in pad_img:
            mpi = self.infer_forward(img, ret_cfg + 'ret_net')
            if mpi is not None:
                mpi, net = mpi
                disps.append(self.net2disparity(net[:, :4]).to(device))
                mpis.append(mpi.to(device))

        self.clear()
        if 'ret_disp' in ret_cfg:
            return mpis, disps
        else:
            return mpis


class PipelineFilteringSV(ModelandDispLoss):
    """
    mirror pading the first and last frames
    """

    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__(models, cfg)
        self.img_window = deque()  # level0 material
        self.net_window = deque()  # level1 material
        self.imbg_window = deque()
        self.flow_cache = dict()

        self.winsz = cfg.pop("winsz", 7)  # level0 bootstrap len
        self.mididx = self.winsz // 2
        self.defer_num = self.mididx

        x = np.zeros(self.winsz)
        x[self.mididx] = 1
        kernel = gaussian_filter1d(x, sigma=2)
        self.filter_weight = torch.tensor(kernel)

    def clear(self):
        self.img_window.clear()
        self.net_window.clear()
        self.imbg_window.clear()
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

    def _1update_one_net(self, net, imbg, ret_cfg=""):  # update level1 material / pipeline1 action / update net
        self.net_window.append(net)
        self.imbg_window.append(imbg)
        if len(self.net_window) > self.winsz:  # free memory in last level
            self.img_window.popleft()
            self.net_window.popleft()
            self.imbg_window.popleft()
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
                    net_warp_weight.append(torch.ones_like(self.net_window[i][:, 0:1]))
                else:
                    net_warp, occ_norm = self.warp_and_occ(self.estimate_flow(i, mididx)[0][1],
                                                           self.estimate_flow(mididx, i)[0][1],
                                                           self.net_window[i])
                    net_warp_weight.append(torch.exp(-occ_norm / 1.6))
                    net_warp_list.append(net_warp)
            net_warps = torch.cat(net_warp_list, dim=0)
            weights = torch.cat(net_warp_weight, dim=0)
            self.filter_weight = self.filter_weight.type_as(net_warps).reshape(-1, 1, 1, 1)
            weights = self.filter_weight * weights
            net_filtered = (weights * net_warps).sum(dim=0, keepdim=True) / weights.sum(dim=0, keepdim=True)
            imbg = self.imbg_window[mididx]
            mpi = netout2mpi(torch.cat([net_filtered, imbg], dim=1), self.img_window[mididx])
            return mpi

    def _0update_one_frame(self, img, ret_cfg=""):  # update level0 material / pipeline0 action / update net
        """return whether is ready for output one frame"""
        batchsz, cnl, hei, wid = img.shape
        self.prepareoffset(wid, hei)
        self.img_window.append(img)
        net = self.model(img)
        net, imbg = net[:, :-3], net[:, -3:]
        return net, imbg

    def infer_forward(self, img: torch.Tensor, ret_cfg=None):
        """
        estimate the mpi in index idx (self.img_window)
        """
        self.eval()
        ret_cfg = "" if ret_cfg is None else ret_cfg
        if img.dim() == 3:
            img = img.unsqueeze(0)
        ret0 = self._0update_one_frame(img, ret_cfg)
        ret1 = self._1update_one_net(*ret0, ret_cfg)
        if ret1 is not None:
            mpi = ret1
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


class PipelineLBTC(nn.Module):
    def __init__(self, models: nn.ModuleDict, cfg: dict):
        super().__init__()
        self.loss_weight = cfg["loss_weights"].copy()
        self.alpha = self.loss_weight["alpha"] if "alpha" in self.loss_weight else 50.0
        self.disp_consist = self.loss_weight["disp_consist"] if "disp_consist" in self.loss_weight else True
        self.criterion = nn.L1Loss(reduction='mean')

        self.tcmodel = models["LBTC"].cuda()
        if "AppearanceFlow" in models.keys():
            self.svtype = "my"
            self.svmodel = PipelineV2(models, {}).cuda()
        else:
            self.svtype = "sv"
            self.svmodel = models["MPI"].cuda()
        if hasattr(self.svmodel, "flow_estim"):
            self.flow_estim = self.svmodel.flow_estim
        else:
            self.flow_estim = RAFTNet(False).cuda()
            self.flow_estim.eval()
            state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.flow_estim.load_state_dict(state_dict)
        for param in self.flow_estim.parameters():
            param.requires_grad = False
        for param in self.svmodel.parameters():
            param.requires_grad = False

        self.offset, self.offset_bhw2 = None, None

        # for inference usage
        self.lstm_state = None
        self.frame_o_last = None
        self.frame_i_last = None

    def clear(self):
        self.lstm_state = None
        self.frame_o_last = None
        self.frame_i_last = None

    def infer_forward(self, img: torch.Tensor, ret_cfg=None):
        if self.svtype == "sv":
            return self.infer_forward_mpi(img, ret_cfg)
        elif self.svtype == "my":
            return self.infer_forward_my(img, ret_cfg)
        else:
            raise RuntimeError("svtype not recognized")

    @torch.no_grad()
    def infer_forward_mpi(self, img: torch.Tensor, ret_cfg=None):
        """
        estimate the mpi in index idx (self.img_window)
        """
        self.eval()

        netout = self.svmodel(img)
        framep = netout[:, :-3]
        imbg = netout[:, -3:]

        if self.frame_i_last is None:
            self.frame_o_last = framep
            self.frame_i_last = img
            return netout2mpi(netout, img)
        else:
            inp = torch.cat([framep, self.frame_o_last, img, self.frame_i_last], dim=1)
            framep, self.lstm_state = self.tcmodel(inp, self.lstm_state)

            self.frame_o_last = framep
            self.frame_i_last = img
            return netout2mpi(torch.cat([framep, imbg], dim=1), img)

    def infer_multiple(self, imgs: list, ret_cfg="", device=torch.device('cpu')):
        self.eval()
        mpis = []
        for img in imgs:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.cuda()
            mpi = self.infer_forward(img, ret_cfg)
            mpis.append(mpi.to(device))
        self.clear()
        return mpis

    def prepareoffset(self, wid, hei):
        if self.offset is None or self.offset.shape[-2:] != (hei, wid):
            offsety, offsetx = torch.meshgrid([
                torch.linspace(0, hei - 1, hei),
                torch.linspace(0, wid - 1, wid)
            ])
            self.offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).float().cuda()
            self.offset_bhw2 = self.offset.permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def estimate_flow(self, im1, im2):
        flow, flow_mask, flow_net = self.flow_estim(im1, im2, ret_upmask=True)
        return upsample_flow(flow, flow_mask), flow_net

    @torch.no_grad()
    def forward_svmodel(self, im, flownet_list):
        if self.svtype == "sv":
            frame_p = self.svmodel(im)[:, :-3]
        elif self.svtype == "my":
            d = {"net_up": None}
            self.svmodel.forwardmpi(d, im, flownet_list)
            frame_p = d.pop("net_up")
        else:
            raise RuntimeError("svtype not recognized")
        return frame_p

    def draw_disparity(self, netout):
        batchsz, layernum, height, width = netout.shape
        layernum += 1
        if self.svtype == "sv":
            depth = make_depths(layernum).type_as(netout).unsqueeze(0).repeat(batchsz, 1)
            alpha = torch.cat([torch.ones([batchsz, 1, height, width]).type_as(netout), netout], dim=1)
            disp = estimate_disparity_torch(alpha.unsqueeze(2), depth)
        elif self.svtype == "my":
            fg, bg = netout[:, :2].split(1, dim=1)
            disp = torch.cat([fg, bg], dim=-1)
        else:
            raise RuntimeError("svtype not recognized")
        return disp

    def forward(self, *args: torch.Tensor, **kwargs):
        self.train()
        refims = args[0].cuda()
        batchsz, framenum, _, heiori, widori = refims.shape
        frame_i = [refims[:, i].cuda() for i in range(refims.shape[1])]

        # process frame0
        _, net0 = self.estimate_flow(frame_i[0], frame_i[1])
        _, net1 = self.estimate_flow(frame_i[0], frame_i[2])
        frame_p1 = self.forward_svmodel(frame_i[0], [net0, net1])

        frame_o = [frame_p1]
        disp_o = [None] * framenum
        disp_p = [None] * framenum
        if self.disp_consist:
            disp0 = self.draw_disparity(frame_p1).unsqueeze(1)
            disp_o[0] = disp_p[0] = disp0

        lstm_state = None
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}

        # forward
        for t in range(1, framenum):

            frame_i1 = frame_i[t - 1]
            frame_i2 = frame_i[t]

            # process frame1
            flow_i21, net0 = self.estimate_flow(frame_i2, frame_i1)
            _, net1 = self.estimate_flow(frame_i2, frame_i[t + 1] if t != framenum - 1 else frame_i[t - 2])
            frame_p2 = self.forward_svmodel(frame_i2, [net0, net1])

            if t == 1:
                frame_o1 = frame_p1
            else:
                frame_o1 = frame_o2.detach()  # previous output frame

            # model input
            inputs = torch.cat((frame_p2, frame_o1, frame_i2, frame_i1), dim=1)
            frame_o2, lstm_state = self.tcmodel(inputs, lstm_state)

            # detach from graph and avoid memory accumulation
            lstm_state = lstm_state.detach() if isinstance(lstm_state, torch.Tensor) \
                else tuple(t.detach() for t in lstm_state)

            frame_o.append(frame_o2)
            if self.disp_consist:
                disp_o[t] = self.draw_disparity(frame_o2).unsqueeze(1)
                disp_p[t] = self.draw_disparity(frame_p2).unsqueeze(1)

            # short-term temporal loss
            if "short_term" in self.loss_weight.keys():
                # warp I1 and O1
                warp_i1 = warp_flow(frame_i1, flow_i21, self.offset_bhw2)
                noc_mask2 = torch.exp(-self.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2)).unsqueeze(1)

                frame1 = frame_o1 if not self.disp_consist else disp_o[t - 1]
                frame2 = frame_o2 if not self.disp_consist else disp_o[t]
                warp1 = warp_flow(frame1, flow_i21, self.offset_bhw2)
                loss = self.criterion(frame2 * noc_mask2, warp1 * noc_mask2)

                final_loss += (loss * self.loss_weight["short_term"])
                if "short_term" not in loss_dict.keys():
                    loss_dict["short_term"] = loss.detach()
                else:
                    loss_dict["short_term"] += loss.detach()

            # perceptual loss
            if "sv_loss" in self.loss_weight.keys():
                sv_loss = self.criterion(frame_o2, frame_p2)
                final_loss += (sv_loss * self.loss_weight["sv_loss"])
                if "svloss" not in loss_dict.keys():
                    loss_dict["svloss"] = sv_loss.detach()
                else:
                    loss_dict["svloss"] += sv_loss.detach()

            if "svg_loss" in self.loss_weight.keys():
                svg_loss = torch.tensor(0.).type_as(frame_o2)
                frame_o2_d = frame_o2 if not self.disp_consist else disp_o[t]
                frame_p2_d = frame_p2 if not self.disp_consist else disp_p[t]
                for i in range(3):
                    frame_o2_d = torchf.interpolate(frame_o2_d, scale_factor=0.5)
                    frame_p2_d = torchf.interpolate(frame_p2_d, scale_factor=0.5)
                    o2_gx, o2_gy = gradient(frame_o2_d)
                    p2_gx, p2_gy = gradient(frame_p2_d)
                    svg_loss += self.criterion(o2_gx, p2_gx)
                    svg_loss += self.criterion(o2_gy, p2_gy)
                svg_loss /= 3
                final_loss += (svg_loss * self.loss_weight["svg_loss"])
                if "svgloss" not in loss_dict.keys():
                    loss_dict["svgloss"] = svg_loss.detach()
                else:
                    loss_dict["svgloss"] += svg_loss.detach()

        # end of forward
        if "long_term" in self.loss_weight.keys():
            t1 = 0
            for t2 in range(t1 + 2, framenum):
                frame_i1 = frame_i[t1]
                frame_i2 = frame_i[t2]

                frame1 = frame_o[t1].detach() if not self.disp_consist else disp_o[t1].detach()
                frame2 = frame_o[t2] if not self.disp_consist else disp_o[t2]

                # compute flow (from I2 to I1)
                flow_i21, _ = self.estimate_flow(frame_i2, frame_i1)

                # warp I1 and O1
                warp_i1 = warp_flow(frame_i1, flow_i21, self.offset_bhw2)
                noc_mask2 = torch.exp(-self.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2)).unsqueeze(1)

                warp1 = warp_flow(frame1, flow_i21, self.offset_bhw2)
                lt_loss = self.criterion(frame2 * noc_mask2, warp1 * noc_mask2)

                final_loss += (lt_loss * self.loss_weight["long_term"])
                if "long_term" not in loss_dict.keys():
                    loss_dict["long_term"] = lt_loss.detach()
                else:
                    loss_dict["long_term"] += lt_loss.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}

    @torch.no_grad()
    def valid_forward(self, *args: torch.Tensor, **kwargs):
        self.eval()
        eval_framenum = 5
        refims = args[0][:, :eval_framenum].cuda()

        batchsz, framenum, _, heiori, widori = refims.shape
        frame_i = [refims[:, i].cuda() for i in range(refims.shape[1])]

        # process frame0
        _, net0 = self.estimate_flow(frame_i[0], frame_i[1])
        _, net1 = self.estimate_flow(frame_i[0], frame_i[2])
        frame_p1 = self.forward_svmodel(frame_i[0], [net0, net1])

        frame_o = [frame_p1]
        disp_o = [None] * framenum
        disp_p = [None] * framenum
        disp0 = self.draw_disparity(frame_p1).unsqueeze(1)
        disp_o[0] = disp_p[0] = disp0

        lstm_state = None
        val_dict = {}

        # forward
        for t in range(1, framenum):
            frame_i1 = frame_i[t - 1]
            frame_i2 = frame_i[t]

            # process frame1
            flow_i21, net0 = self.estimate_flow(frame_i2, frame_i1)
            _, net1 = self.estimate_flow(frame_i2, frame_i[t + 1] if t != framenum - 1 else frame_i[t - 2])
            frame_p2 = self.forward_svmodel(frame_i2, [net0, net1])

            if t == 1:
                frame_o1 = frame_p1
            else:
                frame_o1 = frame_o2.detach()  # previous output frame

            # model input
            inputs = torch.cat((frame_p2, frame_o1, frame_i2, frame_i1), dim=1)
            frame_o2, lstm_state = self.tcmodel(inputs, lstm_state)

            # detach from graph and avoid memory accumulation
            lstm_state = lstm_state.detach() if isinstance(lstm_state, torch.Tensor) \
                else tuple(t.detach() for t in lstm_state)

            frame_o.append(frame_o2)
            disp_o[t] = self.draw_disparity(frame_o2).unsqueeze(1)
            disp_p[t] = self.draw_disparity(frame_p2).unsqueeze(1)

            # short-term temporal loss
            # warp I1 and O1
            warp_i1 = warp_flow(frame_i1, flow_i21, self.offset_bhw2)
            noc_mask2 = torch.exp(-self.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2)).unsqueeze(1)

            warp_o1 = warp_flow(disp_o[t - 1], flow_i21, self.offset_bhw2)
            loss = self.criterion(disp_o[t] * noc_mask2, warp_o1 * noc_mask2)
            if "val_shorterm" not in val_dict.keys():
                val_dict["val_shorterm"] = loss.detach()
            else:
                val_dict["val_shorterm"] += loss.detach()

            # perceptual loss
            sv_loss = self.criterion(disp_o[t], disp_p[t])
            if "val_sv" not in val_dict.keys():
                val_dict["val_sv"] = sv_loss.detach()
            else:
                val_dict["val_sv"] += sv_loss.detach()

        # long term loss
        t1 = 0
        for t2 in range(t1 + 2, framenum):
            frame_i1 = frame_i[t1]
            frame_i2 = frame_i[t2]

            frame1 = disp_o[t1].detach()  # make a new Variable to avoid backwarding gradient
            frame2 = disp_o[t2]

            # compute flow (from I2 to I1)
            flow_i21, _ = self.estimate_flow(frame_i2, frame_i1)

            # warp I1 and O1
            warp_i1 = warp_flow(frame_i1, flow_i21, self.offset_bhw2)
            noc_mask2 = torch.exp(-self.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2)).unsqueeze(1)

            warp_o1 = warp_flow(frame1, flow_i21, self.offset_bhw2)
            lt_loss = self.criterion(frame2 * noc_mask2, warp_o1 * noc_mask2)

            if "val_longterm" not in val_dict.keys():
                val_dict["val_longterm"] = lt_loss.detach()
            else:
                val_dict["val_longterm"] += lt_loss.detach()

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            dispp = self.draw_disparity(frame_p2)
            dispo = self.draw_disparity(frame_o2)
            frame = (frame_i2[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_dispp"] = draw_dense_disp(dispp, 1)
            val_dict["vis_dispo"] = draw_dense_disp(dispo, 1)
            val_dict["vis_img"] = frame
        return val_dict


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
