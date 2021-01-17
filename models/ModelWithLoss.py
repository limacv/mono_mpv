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
        self.model.cuda()

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

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
        refim, tarim, refextrin, tarextrin, intrin, pt2ds, ptzs_gt = args

        batchsz, _, heiori, widori = refim.shape
        val_dict = {}
        with torch.no_grad():
            self.model.eval()
            netout = self.model(refim)
            self.model.train()

            if isinstance(netout, Tuple):
                netout = netout[0]
            # compute mpi from netout
            mpi, blend_weight = netout2mpi(netout, refim, ret_blendw=True)

            # estimate depth map and sample sparse point depth
            depth = make_depths(32).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            disparity = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)
            ptdis_e = torchf.grid_sample(disparity.unsqueeze(1), pt2ds.unsqueeze(1), align_corners=True)\
                .squeeze(1).squeeze(1)

            # compute scale
            scale = torch.exp(torch.log(ptdis_e * ptzs_gt).mean(dim=-1, keepdim=True))
            depth *= scale

            # render target view
            tarview, tarmask = render_newview(mpi, refextrin, tarextrin, intrin, depth, True)

            l1_map = self.photo_loss(tarview, tarim)
            l1_loss = (l1_map * tarmask).sum() / tarmask.sum()
            val_dict["val_l1diff"] = float(l1_loss)
            val_dict["val_scale"] = float(scale.mean())

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            view0 = (tarview[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_newv"] = view0
            val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt, depth[0, -1])
            val_dict["vis_disp"] = draw_dense_disp(disparity, 1)
            val_dict["save_mpi"] = mpi[0]
        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refim, tarim, refextrin, tarextrin, intrin, pt2ds, ptzs_gt = args

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
        ptdis_e = torchf.grid_sample(disparity.unsqueeze(1), pt2ds.unsqueeze(1), align_corners=True)\
            .squeeze(1).squeeze(1)

        # with torch.no_grad():  # compute scale
        scale = torch.exp(torch.log(ptdis_e * ptzs_gt).mean(dim=-1, keepdim=True))
        depth *= scale
        # render target view
        tarview, tarmask = render_newview(mpi, refextrin, tarextrin, intrin, depth, True)
        # sparsedepthgt = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt)
        # sparsedepth1 = draw_sparse_depth(refim, pt2ds, ptdis_e / scale)
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        loss_dict["s_bg"] = bg_pct
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = self.photo_loss(tarview, tarim)
            l1_loss = (l1_loss * tarmask).sum() / tarmask.sum()
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = l1_loss.detach()

        if "smooth_loss" in self.loss_weight.keys():
            smth_loss = smooth_grad(disparity, refim)
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"])
            loss_dict["smth"] = smth_loss.detach()

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(ptdis_e * ptzs_gt / scale)
            diff = torch.pow(diff, 2).mean()
            final_loss += (diff * self.loss_weight["depth_loss"])
            loss_dict["depth"] = diff.detach()

        if "sparse_loss" in self.loss_weight.keys():
            alphas = mpi[:, :, -1, :, :]
            alphas_norm = alphas / torch.norm(alphas, dim=1, keepdim=True)
            sparse_loss = torch.sum(torch.abs(alphas_norm), dim=1).mean()
            final_loss += sparse_loss * self.loss_weight["sparse_loss"]
            loss_dict["sparse_loss"] = sparse_loss.detach()

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

        self.model = model
        self.model.train()
        self.model.cuda()
        self.layernum = self.model.num_layers

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "vgg")

        # used for backward warp
        self.flow_estim = RAFTNet(False)
        self.flow_estim.eval()
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)
        self.offset = None
        self.offset_bhw2 = None

        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

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
        refims, tarims, disp_gts, certainty_maps, isleft = args
        shift_gt = isleft[:, 1]
        isleft = isleft[:, 0]
        batchsz, framenum, _, heiori, widori = refims.shape
        self.prepareoffset(widori, heiori)
        layernum = self.model.num_layers
        bfnum_1 = batchsz * (framenum - 1)
        bfnum_2 = batchsz * (framenum - 2)
        with torch.no_grad():
            flows_f = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                      refims[:, 1:].reshape(bfnum_1, 3, heiori, widori))
            flows_b = self.flow_estim(refims[:, 1:-1].reshape(bfnum_2, 3, heiori, widori),
                                      refims[:, :-2].reshape(bfnum_2, 3, heiori, widori))
            certainty_norm = certainty_maps.sum(dim=[-1, -2], keepdim=True)
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
        disp = torch.stack(disp_list, dim=1)
        disp_gt = (disp_gts[:, 1:-1] - shift_gt.reshape(batchsz, 1, 1, 1)) \
                  * scale.reshape(batchsz, 1, 1, 1) * -isleft.reshape(batchsz, 1, 1, 1)
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
        refim, tarim, disp_gt, certainty_map, isleft = args
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
        scale = torch.exp((torch.log(disp_diff) * certainty_map).sum(dim=[-1, -2], keepdim=True)
                          / certainty_norm.reshape(batchsz, 1, 1))

        disparities = torch.reciprocal(depth * scale.reshape(-1, 1) * -isleft) + shift_gt.reshape(-1, 1)
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
            # alphas = mpi[:, :, -1, :, :]
            # alphas_norm = alphas / torch.norm(alphas, dim=1, keepdim=True)
            # sparse_loss = torch.sum(torch.abs(alphas_norm), dim=1).mean()
            # final_loss += sparse_loss * self.loss_weight["sparse_loss"]
            # loss_dict["sparse_loss"] = sparse_loss.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


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
        models.cuda()
        self.mpimodel = models["MPI"]
        self.sfmodel = models["SceneFlow"] if "SceneFlow" in models.keys() else None
        self.afmodel = models["AppearanceFlow"]

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        # self.splat_func = forward_scatter_withweight

        # adjustable config
        self.pipe_optim_frame0 = self.loss_weight.pop("pipe_optim_frame0", False)
        self.mpi_flowgrad_in = self.loss_weight.pop("mpi_flowgrad_in", True)
        self.aflow_residual = self.loss_weight.pop("aflow_residual", True)
        self.aflow_fusefgpct = self.loss_weight.pop("aflow_fusefgpct", False)
        self.aflow_includeself = self.loss_weight.pop("aflow_includeself", False)
        self.depth_loss_mode = self.loss_weight.pop("depth_loss_mode", "norm")
        self.depth_loss_ord = self.loss_weight.pop("depth_loss_ord", 1)
        self.depth_loss_rmresidual = self.loss_weight.pop("depth_loss_rmresidual", True)
        self.temporal_loss_mode = self.loss_weight.pop("temporal_loss_mode", "msle")

        print(f"PipelineV2 activated config:\n"
              f"pipe_optim_frame0: {self.pipe_optim_frame0}\n"
              f"mpi_flowgrad_in: {self.mpi_flowgrad_in}\n"
              f"aflow_residual: {self.aflow_residual}\n"
              f"aflow_includeself: {self.aflow_includeself}\n"
              f"depth_loss_mode: {self.depth_loss_mode}\n"
              f"depth_loss_ord: {self.depth_loss_ord}\n"
              f"depth_loss_rmresidual: {self.depth_loss_rmresidual}\n"
              f"temporal_loss_mode: {self.temporal_loss_mode}\n"
              f"aflow_fusefgpct: {self.aflow_fusefgpct}\n")

        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.eval()
        self.flow_estim.cuda()
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

        self.pct_scheduler = ParamScheduler(
            milestones=[10e3, 20e3],
            values=[0.5, 1]
        ) if self.aflow_fusefgpct else None
        self.mask_loss_scheduler = ParamScheduler(
            milestones=[5e3, 10e3],
            values=[1, 0]
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

        if self.aflow_includeself or "aflow_includeself" in ret_cfg:
            img_list.append(self.img_window[1])
            with torch.no_grad():
                flowself, flownetself, flowmaskself = \
                    self.flow_estim(self.img_window[1], self.img_window[1])
            flow_list.append(flowself)
            flownet_list.append(flownetself)
            flowmask_list.append(flowmaskself)

        imbg = self.forwardaf(disp, net, upmask,
                              flows_list=flow_list,
                              imgs_list=img_list,
                              flow_nets_list=flownet_list,
                              flow_upmasks_list=flowmask_list)
        mpi = alpha2mpi(alpha, self.img_window[1], imbg, blend_weight=blend_weight)

        # update last frame info
        self.lastmpi = mpi
        self.net_warp = self.forwardsf(self.flowf_win[1], self.flowb_win[1], net)
        return mpi

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        self.eval()
        refims, tarims, disp_gts, certainty_maps, isleft = args
        shift_gt = isleft[:, 1]
        isleft = isleft[:, 0]
        batchsz, framenum, _, heiori, widori = refims.shape
        self.prepareoffset(widori, heiori)
        layernum = self.mpimodel.num_layers
        bfnum_1 = batchsz * (framenum - 1)
        bfnum_2 = batchsz * (framenum - 2)
        with torch.no_grad():
            flows_f, flows_f_upmask, flows_f_net = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                                                   refims[:, 1:].reshape(bfnum_1, 3, heiori, widori),
                                                                   ret_upmask=True)
            flows_b, flows_b_upmask, flows_b_net = self.flow_estim(refims[:, 1:-1].reshape(bfnum_2, 3, heiori, widori),
                                                                   refims[:, :-2].reshape(bfnum_2, 3, heiori, widori),
                                                                   ret_upmask=True)
            flows_f_up = upsample_flow(flows_f, flows_f_upmask)
            flows_b_up = upsample_flow(flows_b, flows_b_upmask)
            certainty_norm = certainty_maps.sum(dim=[-1, -2])
            flows_f = flows_f.reshape(batchsz, framenum - 1, 2, heiori // 8, widori // 8)
            flows_f_up = flows_f_up.reshape(batchsz, framenum - 1, -1, heiori, widori)
            flows_f_net = flows_f_net.reshape(batchsz, framenum - 1, -1, heiori // 8, widori // 8)
            flows_f_upmask = flows_f_upmask.reshape(batchsz, framenum - 1, -1, heiori // 8, widori // 8)
            flows_b = flows_b.reshape(batchsz, framenum - 2, 2, heiori // 8, widori // 8)
            flows_b_up = flows_b_up.reshape(batchsz, framenum - 2, -1, heiori, widori)
            flows_b_net = flows_b_net.reshape(batchsz, framenum - 2, -1, heiori // 8, widori // 8)
            flows_b_upmask = flows_b_upmask.reshape(batchsz, framenum - 2, -1, heiori // 8, widori // 8)

            depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

            alpha0, net, upmask = self.forwardmpi(
                None,
                refims[:, 0],
                [flows_f_net[:, 0]],
                None
            )
            net_warp = self.forwardsf(flows_f[:, 0], flows_b[:, 0], net)
            disp0 = estimate_disparity_torch(alpha0.unsqueeze(2), depth)
            disp0_warp = self.flowfwarp_warp(flows_f_up[:, 0], flows_b_up[:, 0], disp0.unsqueeze(1))

            net_warp_list = [net_warp]
            disp_warp_list = [disp0_warp]
            disp_list = []
            mpi_list = []
            net_list = []

            for frameidx in range(1, framenum - 1):
                alpha, net, upmask = self.forwardmpi(
                    None,
                    refims[:, frameidx],
                    [flows_b_net[:, frameidx - 1], flows_f_net[:, frameidx]],
                    net_warp
                )
                disp, blend_weight = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)

                flow_list = [flows_b[:, frameidx - 1], flows_f[:, frameidx]]
                flow_net_list = [flows_b_net[:, frameidx - 1], flows_f_net[:, frameidx]]
                flow_mask_list = [flows_b_upmask[:, frameidx - 1], flows_f_upmask[:, frameidx]]
                image_list = [refims[:, frameidx - 1], refims[:, frameidx + 1]]
                if self.aflow_includeself:
                    image_list.append(refims[:, frameidx])
                    with torch.no_grad():
                        flow_self, flow_net_self, flow_upmask_self = \
                            self.flow_estim(refims[:, frameidx], refims[:, frameidx])
                        flow_list.append(flow_self)
                        flow_net_list.append(flow_net_self)
                        flow_mask_list.append(flow_upmask_self)
                imbg = self.forwardaf(disp, net, upmask,
                                      flows_list=flow_list,
                                      imgs_list=image_list,
                                      flow_nets_list=flow_net_list,
                                      flow_upmasks_list=flow_mask_list)
                mpi = alpha2mpi(alpha, refims[:, frameidx], imbg, blend_weight=blend_weight)

                disp_list.append(disp)
                mpi_list.append(mpi)
                net_list.append(net)

                if frameidx >= framenum - 2:
                    break
                net_warp = self.forwardsf(
                    flowf=flows_f[:, frameidx],
                    flowb=flows_b[:, frameidx],
                    net=net
                )
                disp_warp = self.flowfwarp_warp(flows_f_up[:, frameidx], flows_b_up[:, frameidx], disp.unsqueeze(1))
                net_warp_list.append(net_warp)
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
        disp = torch.stack(disp_list, dim=1)
        disp_gt = (disp_gts[:, 1:-1] - shift_gt.reshape(batchsz, 1, 1, 1)) \
                  * scale.reshape(batchsz, 1, 1, 1) * -isleft.reshape(batchsz, 1, 1, 1)
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

        if net_warp is None:
            net_cnl = self.mpimodel.outcnl
            net_warp = torch.zeros(batchsz, net_cnl, hei // 8, wid // 8).type_as(img)

        # randomly drop one flow net if training
        if len(flow_net_list) == 1:
            with torch.no_grad():
                selfflow, _, flow_net = self.flow_estim(img, img, ret_upmask=True)
                flow_net_list.append(flow_net)

        flow_net = torch.cat(flow_net_list, dim=1)
        net, upmask = self.mpimodel(img, flow_net, net_warp)

        alpha = self.net2alpha(net, upmask)
        return alpha, net, upmask

    def net2alpha(self, net, upmask) -> torch.Tensor:
        layernum = self.mpimodel.num_layers
        netout = learned_upsample(net, upmask)
        batchsz, outcnl, hei, wid = netout.shape
        alpha = torch.linspace(0, 1, layernum).reshape(1, layernum, 1, 1).type_as(netout)
        if outcnl == 3:
            depth, thick, scale = torch.split(netout, 1, dim=1)
            alpha = self.square2alpha(alpha, depth, thick, scale)
        elif outcnl == 6:
            depth1, thick1, scale1, depth2, thick2, scale2 = torch.split(netout, 1, dim=1)
            alpha1 = self.square2alpha(alpha, depth1, thick1, scale1)
            alpha2 = self.square2alpha(alpha, depth2, thick2, scale2)
            alpha = alpha1 + alpha2

        alpha = torch.cat([torch.ones([batchsz, 1, hei, wid]).type_as(alpha), alpha], dim=1)
        return alpha

    def square2alpha(self, x, depth, thick, scale=1):
        denorm = self.mpimodel.num_layers - 1
        n = denorm * scale * (x - depth)
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

    def forwardaf(self, disparity, net, upmask,
                  flows_list: list, imgs_list: list, flow_nets_list: list, flow_upmasks_list: list):
        """
        Guarantee the flows[0] and imgs_list[0] to be the last frame
        """
        assert len(flows_list) == len(imgs_list) == len(flow_nets_list)
        assert isinstance(self.afmodel, (ASPFNetWithMaskInOut, ASPFNetWithMaskOut)), f"model not compatible"

        masks, bgs = [], []
        lastmask = None
        for flow, img, flow_net, flow_upmask in zip(flows_list, imgs_list, flow_nets_list, flow_upmasks_list):
            flow = learned_upsample(flow, flow_upmask)
            if isinstance(self.afmodel, ASPFNetWithMaskOut):
                bgflow, mask = self.afmodel(flow, disparity)
            else:
                bgflow, mask = self.afmodel(flow, disparity, lastmask)
                lastmask = mask if lastmask is None else mask + lastmask

            if self.aflow_residual:
                bgflow = flow + bgflow
            bg = warp_flow(img, bgflow, offset=self.offset_bhw2)
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
        layernum = self.mpimodel.num_layers
        # remove first frame's gt
        tarims = tarims[:, 1:]
        disp_gts = disp_gts[:, 1:]
        shift_gt = isleft[:, 1]
        isleft = isleft[:, 0]
        certainty_maps = certainty_maps[:, 1:]
        fg_pct = None
        step = kwargs["step"]

        batchsz, framenum, _, heiori, widori = refims.shape
        self.prepareoffset(widori, heiori)
        bfnum_1 = batchsz * (framenum - 1)
        bfnum_2 = batchsz * (framenum - 2)
        # All are [B x Frame [x cnl] x H x W]
        with torch.no_grad():
            # the flow in /8 resolution
            flows_f, flows_f_upmask, flows_f_net = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                                                   refims[:, 1:].reshape(bfnum_1, 3, heiori, widori),
                                                                   ret_upmask=True)
            flows_b, flows_b_upmask, flows_b_net = self.flow_estim(refims[:, 1:-1].reshape(bfnum_2, 3, heiori, widori),
                                                                   refims[:, :-2].reshape(bfnum_2, 3, heiori, widori),
                                                                   ret_upmask=True)
            certainty_norm = certainty_maps.sum(dim=[-1, -2])
            flows_f = flows_f.reshape(batchsz, framenum - 1, 2, heiori // 8, widori // 8)
            flows_f_upmask = flows_f_upmask.reshape(batchsz, framenum - 1, -1, heiori // 8, widori // 8)
            flows_f_net = flows_f_net.reshape(batchsz, framenum - 1, -1, heiori // 8, widori // 8)
            flows_b = flows_b.reshape(batchsz, framenum - 2, 2, heiori // 8, widori // 8)
            flows_b_upmask = flows_b_upmask.reshape(batchsz, framenum - 2, -1, heiori // 8, widori // 8)
            flows_b_net = flows_b_net.reshape(batchsz, framenum - 2, -1, heiori // 8, widori // 8)
        disp_out = []
        mpi_out = []
        net_out = []
        upmask_out = []
        intermediates = {
            "flow_gradx": [],
            "flow_grady": [],
            "blend_weight": [],
            "net_warp": []
        }
        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        # for the first frame, process it specially
        with torch.set_grad_enabled(self.pipe_optim_frame0):
            flow0_1 = flows_f[:, 0]
            alphas, net, upmask = self.forwardmpi(
                None,
                refims[:, 0],
                [flows_f_net[:, 0]],
                None
            )

            # disp = estimate_disparity_torch(alphas.unsqueeze(2), depth)
            net_warp = self.forwardsf(flow0_1, flows_b[:, 0], net)
            intermediates["net_warp"].append(net_warp)

        # there will be framenum - 2 mpi,
        for frameidx in range(1, framenum - 1):
            # predict alphas
            alphas, net, upmask = self.forwardmpi(
                intermediates,
                refims[:, frameidx],
                [flows_b_net[:, frameidx - 1], flows_f_net[:, frameidx]],
                net_warp
            )
            disp, blend_weight = estimate_disparity_torch(alphas.unsqueeze(2), depth, retbw=True)
            disp_out.append(disp)
            net_out.append(net)
            upmask_out.append(upmask)
            intermediates["blend_weight"].append(blend_weight)

            # predict appearance flow
            flow_list = [flows_b[:, frameidx - 1], flows_f[:, frameidx]]
            flow_net_list = [flows_b_net[:, frameidx - 1], flows_f_net[:, frameidx]]
            flow_mask_list = [flows_b_upmask[:, frameidx - 1], flows_f_upmask[:, frameidx]]
            image_list = [refims[:, frameidx - 1], refims[:, frameidx + 1]]
            if self.aflow_includeself:
                image_list.append(refims[:, frameidx])
                with torch.no_grad():
                    flow_self, flow_net_self, flow_upmask_self = \
                        self.flow_estim(refims[:, frameidx], refims[:, frameidx])
                    flow_list.append(flow_self)
                    flow_net_list.append(flow_net_self)
                    flow_mask_list.append(flow_upmask_self)
            imbg = self.forwardaf(disp, net, upmask,
                                  flows_list=flow_list,
                                  imgs_list=image_list,
                                  flow_nets_list=flow_net_list,
                                  flow_upmasks_list=flow_mask_list)

            if self.pct_scheduler is not None:
                fg_pct = self.pct_scheduler.get_value(step)
                imfg = imbg * (1. - fg_pct) + refims[:, frameidx] * fg_pct
            else:
                imfg = refims[:, frameidx]

            mpi = alpha2mpi(alphas, imfg, imbg, blend_weight=blend_weight)
            mpi_out.append(mpi)

            if frameidx >= framenum - 2:
                break

            # predict sceneflow and warp
            net_warp = self.forwardsf(
                flowf=flows_f[:, frameidx],
                flowb=flows_b[:, frameidx],
                net=net,
                intcfg=intermediates
            )
            intermediates["net_warp"].append(net_warp)

        # scale and shift invariant
        # with torch.no_grad():  # compute scale
        disp_diffs = [
            disp_out[i] / (torch.abs(disp_gts[:, i] - shift_gt.reshape(batchsz, 1, 1)) + 0.000001)
            for i in range(framenum - 2)
        ]
        # currently use first frame to compute scale and shift
        scale = torch.exp((torch.log(disp_diffs[0]) * certainty_maps[:, 0]).sum(dim=[-1, -2]) / certainty_norm[:, 0])
        scale = scale + 0.005
        # disparity in ground truth space
        disparities = torch.reciprocal(depth * scale.reshape(-1, 1) * -isleft.reshape(-1, 1)) + shift_gt.reshape(-1, 1)
        # render target view
        tarviews = [
            shift_newview(mpi_, -disparities, ret_mask=False)
            for mpi_ in mpi_out
        ]
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        if fg_pct is not None:
            loss_dict["fg_pct"] = torch.tensor(fg_pct).type_as(final_loss)
        # MPI loss
        for mpiframeidx in range(len(mpi_out)):
            if "pixel_loss" in self.loss_weight.keys():
                l1_loss = self.photo_loss(tarviews[mpiframeidx],
                                          tarims[:, mpiframeidx])
                l1_loss = l1_loss.mean()
                final_loss += (l1_loss * self.loss_weight["pixel_loss"])
                if "pixel" not in loss_dict.keys():
                    loss_dict["pixel"] = l1_loss.detach()
                else:
                    loss_dict["pixel"] += l1_loss.detach()

            if "smooth_loss" in self.loss_weight.keys():
                smth_loss = smooth_grad(
                    disp_out[mpiframeidx],
                    refims[:, mpiframeidx + 1])
                smth_loss = smth_loss.mean()
                final_loss += (smth_loss * self.loss_weight["smooth_loss"])
                if "smth" not in loss_dict.keys():
                    loss_dict["smth"] = smth_loss.detach()
                else:
                    loss_dict["smth"] += smth_loss.detach()

            if "depth_loss" in self.loss_weight.keys():
                diff = torch.log(disp_diffs[mpiframeidx] / scale.reshape(-1, 1, 1))
                diff = (torch.pow(diff, 2) * certainty_maps[:, mpiframeidx]).mean()
                final_loss += (diff * self.loss_weight["depth_loss"])
                if "depth" not in loss_dict.keys():
                    loss_dict["depth"] = diff.detach()
                else:
                    loss_dict["depth"] += diff.detach()

        if "tempdepth_loss" in self.loss_weight.keys():
            temporal = torch.tensor(0.).type_as(refims)
            for net, net_warp in zip(net_out, intermediates["net_warp"]):
                if self.temporal_loss_mode == "mse":
                    # scheme 0: mae
                    diff = ((net - net_warp).abs()).mean()
                elif self.temporal_loss_mode == "msle":
                    # scheme 1: msle
                    net_warp = torch.max(net_warp, torch.tensor(0.000001).type_as(net_warp))
                    diff = (torch.log(net_warp / net).abs()).mean()
                else:
                    raise NotImplementedError(f"{self.name}::temporal_loss_mode "
                                              f"{self.temporal_loss_mode} not recognized")
                temporal += diff

            temporal /= len(disp_out)
            final_loss += (temporal * self.loss_weight["tempdepth_loss"])
            loss_dict["tempdepth"] = temporal.detach()

        if "mask_loss" in self.loss_weight.keys():
            mask_loss = torch.tensor(0.).type_as(refims)
            for upmask, flow_upmask in zip(upmask_out, torch.split(flows_b_upmask, 1, dim=1)):
                diff = (upmask - flow_upmask.squeeze(1)).abs().mean()
                mask_loss += diff
            mask_loss /= len(upmask)
            weight = self.mask_loss_scheduler.get_value(step)
            final_loss += (weight * self.loss_weight["mask_loss"] * mask_loss)
            loss_dict["upmask"] = mask_loss.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class PipelineV2SV(PipelineV2):
    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__(models, cfg)

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        refims, tarims, refextrins, tarextrin, intrin, pt2ds, ptzs_gt = args
        batchsz, framenum, _, heiori, widori = refims.shape
        self.prepareoffset(widori, heiori)
        layernum = self.mpimodel.num_layers
        with torch.no_grad():
            flows_f = self.flow_estim(refims[:, :2].reshape(batchsz * 2, 3, heiori, widori),
                                      refims[:, 1:3].reshape(batchsz * 2, 3, heiori, widori))
            flows_b = self.flow_estim(refims[:, 1], refims[:, 0])
            flows_f = flows_f.reshape(batchsz, 2, 2, heiori, widori)
            depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)
            alpha0 = self.forwardmpi(
                None,
                refims[:, 0],
                [flows_f[:, 0]],
                None
            )
            disp0 = estimate_disparity_torch(alpha0.unsqueeze(2), depth)
            disp_warp = self.forwardsf(flows_f[:, 0], flows_b, disp0, alpha0, refims[:, 0])
            alpha1 = self.forwardmpi(
                None,
                refims[:, 1],
                [flows_f[:, 1], flows_b],
                disp_warp
            )
            disp1, bw1 = estimate_disparity_torch(alpha1.unsqueeze(2), depth, retbw=True)
            flow_list = [flows_b, flows_f[:, 1]]
            image_list = [refims[:, 0], refims[:, 2]]
            if self.aflow_includeself:
                image_list.append(refims[:, 1])
                flow_list.append(torch.zeros_like(flow_list[-1]))
            imbg1 = self.forwardaf(alpha1, disp1,
                                   flows=flow_list,
                                   mpis_last=refims[:, 0],
                                   imgs_list=image_list
                                   )
            mpi1 = alpha2mpi(alpha1, refims[:, 1], imbg1, blend_weight=bw1)
            # render target view
            tarview1 = shift_newview(mpi1, torch.reciprocal(depth * 0.5))

        val_dict = {}
        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            # diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_newv"] = (tarview1[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            # val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt, depth[0, -1])
            val_dict["vis_disp"] = draw_dense_disp(disp1, 1)
            if imbg1.dim() == 5:
                imbg1 = imbg1[:, 9]
            val_dict["vis_bgimg"] = (imbg1[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, refextrins, tarextrin, intrin, pt2ds, ptzs_gts = args
        layernum = self.mpimodel.num_layers
        # remove first frame's gt
        pt2ds = pt2ds[:, 1:]
        ptzs_gts = ptzs_gts[:, 1:]

        batchsz, framenum, _, heiori, widori = refims.shape
        self.prepareoffset(widori, heiori)
        bfnum_1 = batchsz * (framenum - 1)
        bfnum_2 = batchsz * (framenum - 2)
        # All are [B x Frame [x cnl] x H x W]
        with torch.no_grad():
            flows_f = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                      refims[:, 1:].reshape(bfnum_1, 3, heiori, widori))
            flows_b = self.flow_estim(refims[:, 1:-1].reshape(bfnum_2, 3, heiori, widori),
                                      refims[:, :-2].reshape(bfnum_2, 3, heiori, widori))
            flows_f = flows_f.reshape(batchsz, framenum - 1, 2, heiori, widori)
            flows_b = flows_b.reshape(batchsz, framenum - 2, 2, heiori, widori)
        disp_out = []
        mpi_out = []
        intermediates = {
            "flow_gradx": [],
            "flow_grady": [],
            "blend_weight": [],
            "occ_mask": [],
            "disp_warp": []
        }
        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        # for the first frame, process it specially
        with torch.set_grad_enabled(self.pipe_optim_frame0):
            flow0_1 = flows_f[:, 0]
            alphas = self.forwardmpi(
                None,
                refims[:, 0],
                [flow0_1],
                None
            )
            disp = estimate_disparity_torch(alphas.unsqueeze(2), depth)
            disp_warp = self.forwardsf(flow0_1, flows_b[:, 0], disp, alphas, refims[:, 0])
            intermediates["disp_warp"].append(disp_warp)

        # there will be framenum - 2 mpi,
        for frameidx in range(1, framenum - 1):
            # predict alphas
            flow_list = [flows_b[:, frameidx - 1], flows_f[:, frameidx]]
            alphas = self.forwardmpi(
                intermediates,
                refims[:, frameidx],
                flow_list,
                disp_warp
            )
            disp, blend_weight = estimate_disparity_torch(alphas.unsqueeze(2), depth, retbw=True)
            disp_out.append(disp)
            intermediates["blend_weight"].append(blend_weight)

            # predict appearance flow
            image_list = [refims[:, frameidx - 1], refims[:, frameidx + 1]]
            if self.aflow_includeself:
                image_list.append(refims[:, frameidx])
                flow_list.append(torch.zeros_like(flow_list[-1]))
            imbg = self.forwardaf(alphas, disp,
                                  flows=flow_list,
                                  mpis_last=mpi_out[-1][:, :, :3] if len(mpi_out) > 0 else refims[:, 0],
                                  imgs_list=image_list
                                  )
            mpi = alpha2mpi(alphas, refims[:, frameidx], imbg, blend_weight=blend_weight)

            mpi_out.append(mpi)

            if frameidx >= framenum - 2:
                break

            # predict sceneflow and warp
            disp_warp = self.forwardsf(
                flowf=flows_f[:, frameidx],
                flowb=flows_b[:, frameidx],
                disparity=disp,
                alpha=alphas, img=refims[:, frameidx],
                intcfg=intermediates
            )
            intermediates["disp_warp"].append(disp_warp)

        # with torch.no_grad():  # compute scale
        ptdis_es = [
            torchf.grid_sample(disp_out[i].unsqueeze(1),
                               pt2ds[:, i:i + 1], align_corners=True).reshape(batchsz, -1)
            for i in range(len(disp_out))
        ]
        # currently use first frame to compute scale
        scale = torch.exp(torch.log(ptdis_es[0] * ptzs_gts[:, 0]).mean(dim=-1, keepdim=True))

        depth = depth * scale.reshape(-1, 1)
        # render target view
        tarviews = [
            render_newview(
                mpi=mpi_out[i],
                srcextrin=refextrins[:, i + 1],
                tarextrin=tarextrin,
                intrin=intrin,
                depths=depth,
                ret_mask=False
            )
            for i in range(len(mpi_out))
        ]
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        # MPI loss
        for mpiframeidx in range(len(mpi_out)):
            if "pixel_loss" in self.loss_weight.keys():
                l1_loss = self.photo_loss(tarviews[mpiframeidx],
                                          tarims)
                l1_loss = l1_loss.mean()
                final_loss += (l1_loss * self.loss_weight["pixel_loss"])
                if "pixel" not in loss_dict.keys():
                    loss_dict["pixel"] = l1_loss.detach()
                else:
                    loss_dict["pixel"] += l1_loss.detach()

            if "smooth_loss" in self.loss_weight.keys():
                smth_loss = smooth_grad(
                    disp_out[mpiframeidx],
                    refims[:, mpiframeidx + 1])
                smth_loss = smth_loss.mean()
                final_loss += (smth_loss * self.loss_weight["smooth_loss"])
                if "smth" not in loss_dict.keys():
                    loss_dict["smth"] = smth_loss.detach()
                else:
                    loss_dict["smth"] += smth_loss.detach()

            if "smooth_flowgrad_loss" in self.loss_weight.keys():
                flow_gradx = intermediates["flow_gradx"][mpiframeidx]
                flow_grady = intermediates["flow_grady"][mpiframeidx]
                disp_dx, disp_dy = gradient(disp_out[mpiframeidx].unsqueeze(-3))
                smoothx = torch.max(disp_dx - 0.02, torch.tensor(0.).type_as(refims))
                smoothy = torch.max(disp_dy - 0.02, torch.tensor(0.).type_as(refims))
                with torch.no_grad():
                    weightsx = - torch.min(flow_gradx / 2, torch.tensor(1.).type_as(refims)) + 1
                    weightsy = - torch.min(flow_grady / 2, torch.tensor(1.).type_as(refims)) + 1
                smth_loss = smoothx * weightsx + smoothy * weightsy
                smth_loss = smth_loss.mean()
                final_loss += (smth_loss * self.loss_weight["smooth_flowgrad_loss"])
                if "flowgrad" not in loss_dict.keys():
                    loss_dict["flowgrad"] = smth_loss.detach()
                else:
                    loss_dict["flowgrad"] += smth_loss.detach()

            if "depth_loss" in self.loss_weight.keys():
                diff = torch.log(ptdis_es[mpiframeidx] * ptzs_gts[:, mpiframeidx] / scale.reshape(-1, 1))
                diff = torch.pow(diff, 2).mean()
                final_loss += (diff * self.loss_weight["depth_loss"])
                if "depth" not in loss_dict.keys():
                    loss_dict["depth"] = diff.detach()
                else:
                    loss_dict["depth"] += diff.detach()

        if "tempdepth_loss" in self.loss_weight.keys():
            temporal = torch.tensor(0.).type_as(refims)
            for disp, disp_warp in zip(disp_out, intermediates["disp_warp"]):
                mask = (disp_warp > 0).type_as(disp_warp)
                temporal += (torch.abs(disp.unsqueeze(1) - disp_warp) * mask).mean()
            temporal /= len(disp_out)
            final_loss += (temporal * self.loss_weight["tempdepth_loss"])
            loss_dict["tempdepth"] = temporal.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


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


class PipelineV4(PipelineV2):
    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__(models, cfg)
        self.img_window = deque()
        self.net_window = deque()
        self.upmask_window = deque()
        self.flow_cache = dict()

        self.winsz = cfg.pop("winsz", 7)
        self.mididx = self.winsz // 2
        self.filter_weight = torch.tensor([0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.1])

    def estimate_flow(self, idx0, idx1):
        with torch.no_grad():
            if idx0 < 0:
                idx0 = len(self.img_window) + idx0
                assert idx0 >= 0
            if idx1 < 0:
                idx1 = len(self.img_window) + idx1
                assert idx1 >= 0
            if idx0 == idx1:
                return torch.zeros_like(self.offset)
            key = (idx0, idx1)
            if key not in self.flow_cache.keys():
                flowinfo = self.flow_estim(self.img_window[idx0], self.img_window[idx1], ret_upmask=True)
                self.flow_cache[key] = flowinfo
                return flowinfo
            else:
                return self.flow_cache[key]

    def update_one_frame(self, img):
        """return whether is ready for output one frame"""
        batchsz, cnl, hei, wid = img.shape
        self.prepareoffset(wid, hei)
        self.img_window.append(img)
        if len(self.img_window) > self.winsz + 1:  # an extra image for flow estimation
            self.img_window.popleft()
            self.net_window.popleft()
            self.upmask_window.popleft()
            # update flow_cache
            self.flow_cache = {(idx0 - 1, idx1 - 1): v
                               for (idx0, idx1), v in self.flow_cache.items()
                               if idx0 > 0 and idx1 > 0}
        elif len(self.img_window) < self.winsz + 1:
            return False
        return True

    def infer_forward(self, img, ret_cfg: str):
        """
        estimate the mpi in index idx (self.img_window)
        """
        if len(self.img_window) == 0:  # bootstrap
            self.update_one_frame(img)
            return None

        isready = self.update_one_frame(img)
        # compute net and upmask of *last second frame*
        if self.net_warp is None:
            flow_list = [self.estimate_flow(-2, -1)[FLOW_IDX]]
            flownet_list = [self.estimate_flow(-2, -1)[FLOWNET_IDX]]
            flowmask_list = [self.estimate_flow(-2, -1)[FLOWMASK_IDX]]
        else:
            flow_list = [self.estimate_flow(-2, -1)[FLOW_IDX], self.estimate_flow(-2, -3)[FLOW_IDX]]
            flownet_list = [self.estimate_flow(-2, -1)[FLOWNET_IDX], self.estimate_flow(-2, -3)[FLOWNET_IDX]]
            flowmask_list = [self.estimate_flow(-2, -1)[FLOWMASK_IDX], self.estimate_flow(-2, -3)[FLOWMASK_IDX]]
        alpha, net, upmask = self.forwardmpi(None, img, flownet_list, self.net_warp)
        self.net_window.append(net)
        self.upmask_window.append(upmask)
        self.net_warp = self.forwardsf(self.estimate_flow(-2, -1)[FLOW_IDX],
                                       self.estimate_flow(-1, -2)[FLOW_IDX], net)

        if not isready:
            return None
        else:
            net_warp_list = []
            net_warp_weight = []
            for i in range(self.winsz):
                if i == self.mididx:
                    net_warp_list.append(self.net_window[i])
                    net_warp_weight.append(torch.ones_like(net_warp_weight[-1]))
                else:
                    net_warp, occ_norm = self.warp_and_occ(self.estimate_flow(i, self.mididx)[FLOW_IDX],
                                                           self.estimate_flow(self.mididx, i)[FLOW_IDX],
                                                           self.net_window[i])
                    net_warp_weight.append((occ_norm < 0.25).type(torch.float32))
                    net_warp_list.append(net_warp)
            net_warps = torch.cat(net_warp_list, dim=0)
            weights = torch.cat(net_warp_weight, dim=0)
            self.filter_weight = self.filter_weight.type_as(net_warps).reshape(-1, 1, 1, 1)
            weights = self.filter_weight * weights
            net_filtered = (weights * net_warps).sum(dim=0, keepdim=True) / weights.sum(dim=0, keepdim=True)
            upmask = self.upmask_window[self.mididx]
            alpha = self.net2alpha(net_filtered, upmask)

            depth = make_depths(self.mpimodel.num_layers).type_as(self.img_window[0]).unsqueeze(0).repeat(1, 1)
            disp, bw = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)
            flow_list = [self.estimate_flow(self.mididx, 0)[FLOW_IDX],
                         self.estimate_flow(self.mididx, -1)[FLOW_IDX]]
            flownet_list = [self.estimate_flow(self.mididx, 0)[FLOWNET_IDX],
                            self.estimate_flow(self.mididx, -1)[FLOWNET_IDX]]
            flowmask_list = [self.estimate_flow(self.mididx, 0)[FLOWMASK_IDX],
                             self.estimate_flow(self.mididx, -1)[FLOWMASK_IDX]]
            img_list = [self.img_window[0], self.img_window[-1]]
            imbg = self.forwardaf(disp, net_filtered, upmask, flow_list, img_list, flownet_list, flowmask_list)
            mpi = alpha2mpi(alpha, self.img_window[self.mididx], imbg, blend_weight=bw)
            return mpi


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
