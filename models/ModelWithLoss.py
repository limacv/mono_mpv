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
            ptdis_e = torchf.grid_sample(disparity.unsqueeze(1), pt2ds.unsqueeze(1)).squeeze(1).squeeze(1)

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
        ptdis_e = torchf.grid_sample(disparity.unsqueeze(1), pt2ds.unsqueeze(1)).squeeze(1).squeeze(1)

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
        assert "loss_weights" in cfg.keys(), "ModelAndLossBase: didn't find 'loss_weights' in cfg"
        self.loss_weight = cfg["loss_weights"].copy()

        torch.set_default_tensor_type(torch.FloatTensor)

        self.model = model
        self.model.train()
        self.model.cuda()
        self.layernum = self.model.num_layers

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

    def infer_forward(self, im: torch.Tensor):
        with torch.no_grad():
            if im.dim() == 3:
                im = im.unsqueeze(0)
            im = im.cuda()
            self.model.eval()
            netout = self.model(im)
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
        refim, tarim, disp_gt, certainty_map, isleft = args

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
            depth = make_depths(self.layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            disp_hat = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)

            # compute scale
            certainty_norm = certainty_map.sum(dim=[-1, -2])
            disp_scale = disp_hat / (torch.abs(disp_gt) + 0.01)
            scale = torch.exp((torch.log(disp_scale) * certainty_map).sum(dim=[-1, -2]) / certainty_norm)

            depth = depth * scale.reshape(-1, 1) * isleft
            # render target view
            tarview, tarmask = shift_newview(mpi, torch.reciprocal(depth), ret_mask=True)

            l1_map = self.photo_loss(tarview, tarim)
            l1_loss = (l1_map * tarmask).sum() / tarmask.sum()
            val_dict["val_l1diff"] = float(l1_loss)
            val_dict["val_scale"] = float(scale.mean())

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            view0 = (tarview[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            # diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_newv"] = view0
            # val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt, depth[0, -1])
            val_dict["vis_dispgt"] = draw_dense_disp(disp_gt, 1. / disp_gt.max())
            val_dict["vis_disp"] = draw_dense_disp(disp_hat, 1)
            val_dict["save_mpi"] = mpi[0]
        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refim, tarim, disp_gt, certainty_map, isleft = args
        with torch.no_grad():
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

        # with torch.no_grad():  # compute scale
        disp_scale = disp_hat / (torch.abs(disp_gt) + 0.01)
        scale = torch.exp((torch.log(disp_scale) * certainty_map).sum(dim=[-1, -2]) / certainty_norm)

        depth = depth * scale.reshape(-1, 1) * isleft
        # render target view
        tarview = shift_newview(mpi, torch.reciprocal(depth), ret_mask=False)
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        loss_dict["s_bg"] = torch.tensor(bg_pct).type_as(scale.detach())
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = self.photo_loss(tarview, tarim)
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = l1_loss.detach()

        if "smooth_loss" in self.loss_weight.keys():
            smth_loss = smooth_grad(disp_hat, refim)
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"])
            loss_dict["smth"] = smth_loss.detach()

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(disp_scale / scale.reshape(-1, 1, 1))
            diff = (torch.pow(diff, 2) * certainty_map).mean()
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


class ModelandFullv1Loss(nn.Module):
    """
    The entire pipeline using forward warping
    """

    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__()
        self.loss_weight = cfg["loss_weights"].copy()
        torch.set_default_tensor_type(torch.FloatTensor)

        assert (isinstance(models, nn.ModuleDict))
        models.train()
        models.cuda()
        self.mpimodel = models["MPI"]
        self.mpfmodel = models["MPF"]

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        self.learnmpf = self.loss_weight.pop("learnmpf", True)
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

        if "splat_mode" not in self.loss_weight.keys():
            self.splat_func = forward_scatter_mpi_HRLR
        elif self.loss_weight["splat_mode"] == "nearest":
            self.splat_func = forward_scatter_mpi_nearest
        elif self.loss_weight["splat_mode"] == "bilinear":
            self.splat_func = forward_scatter_mpi
        else:
            self.splat_func = forward_scatter_mpi_HRLR

        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.eval()
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)
        self.dilate_mpfin = self.loss_weight.pop("dilate_mpfin", True)
        print(f"Info::ModelandFullv1Loss.dilate_mpfin set to {self.dilate_mpfin}")
        self.renderw2mpf = not self.loss_weight.pop("alpha2mpf", True)
        print(f"Info::ModelandFullv1Loss.renderw2mpf set to {self.renderw2mpf}")
        self.flow_loss_ord = self.loss_weight.pop("flow_smth_ord", 2)
        self.flow_loss_consider_weight = self.loss_weight.pop("flow_smth_bw", False)

        # used for inference
        self.last_img = None
        self.last_mpi_warp = None

    def infer_forward(self, img: torch.Tensor, restart=False, ret_cfg='', mpffromspf=False):
        """
        Will actually deferred one frame, i.e. input f1 -> f2 -> ...   output 0 -> o1 -> o2 -> ...
        """
        if restart:
            self.last_img = None
            self.last_mpi_warp = None
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.cuda()
        batchsz, cnl, hei, wid = img.shape
        layernum = self.mpimodel.num_layers
        with torch.no_grad():
            if self.last_img is None:
                self.last_img = img
                self.last_mpi_warp = None
                return torch.ones(batchsz, layernum, 4, hei, wid).type_as(img), \
                       torch.ones(batchsz, layernum, 2, hei, wid).type_as(img)
            else:
                flows = self.flow_estim(self.last_img, img)
                flow_gradx, flow_grady = gradient(flows)
                flow_gradx = flow_gradx.abs().sum(dim=1).reshape(batchsz, 1, hei, wid)
                flow_grady = flow_grady.abs().sum(dim=1).reshape(batchsz, 1, hei, wid)

                mpi, blend_weight, netout = self.forwardmpi(
                    self.last_img,
                    flow_gradx,
                    flow_grady,
                    mpi_warp=self.last_mpi_warp,
                    ret_netout=True
                )
                # post render needed component
                if len(ret_cfg) > 0:
                    lastmpiwap = self.last_mpi_warp if self.last_mpi_warp is not None else torch.zeros_like(mpi)
                    fg_w = blend_weight.unsqueeze(2)
                    last_w = (-blend_weight.unsqueeze(2) + 1.) * lastmpiwap[:, :, -1:]
                    bg_w = (-blend_weight.unsqueeze(2) + 1.) * (1 - lastmpiwap[:, :, -1:])
                if "bg_only" in ret_cfg:
                    ret = fg_w * self.last_img.unsqueeze(1) + bg_w * netout[:, -3:].unsqueeze(1)  # + last_w * 0.
                    ret = torch.cat([ret, mpi[:, :, -1:]], dim=2)
                elif "warp_only" in ret_cfg or "last_only" in ret_cfg:
                    ret = fg_w * self.last_img.unsqueeze(1) + last_w * lastmpiwap[:, :, :3]
                    ret = torch.cat([ret, mpi[:, :, -1:]], dim=2)
                elif "visualize_weight" in ret_cfg:
                    fg = torch.tensor([0., 0., 1.]).type_as(mpi).reshape(1, 1, 3, 1, 1)
                    bg = torch.tensor([0., 1., 0.]).type_as(mpi).reshape(1, 1, 3, 1, 1)
                    last = torch.tensor([1., 0., 0]).type_as(mpi).reshape(1, 1, 3, 1, 1)
                    ret = fg_w * fg + bg_w * bg + last_w * last
                    ret = torch.cat([ret, mpi[:, :, -1:]], dim=2)
                else:
                    ret = mpi

                if "rm_transparent" in ret_cfg:
                    mask = mpi[:, :, -1:] < 0.05
                    ret = torch.where(mask, torch.ones_like(ret[:, :, :3]), ret[:, :, :3])
                    ret = torch.cat([ret, mpi[:, :, -1:]], dim=2)

                if mpffromspf:
                    mpf = flows.reshape(batchsz, 1, 2, hei, wid).expand(-1, layernum, -1, -1, -1)
                else:
                    mpf = self.forwardmpf(mpi, flows, blend_weight)
                self.last_mpi_warp = self.splat_func(
                    flow01=mpf, mpi=mpi
                )
                self.last_img = img
                return ret, mpf

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        batchsz, framenum, _, heiori, widori = refims.shape
        layernum = self.mpimodel.num_layers
        with torch.no_grad():
            flows = self.flow_estim(refims[:, :2].reshape(batchsz * 2, 3, heiori, widori),
                                    refims[:, 1:3].reshape(batchsz * 2, 3, heiori, widori))
            flow_gradx, flow_grady = gradient(flows)
            flow_gradx = flow_gradx.abs().sum(dim=1).reshape(batchsz, 2, 1, heiori, widori)
            flow_grady = flow_grady.abs().sum(dim=1).reshape(batchsz, 2, 1, heiori, widori)
            flows = flows.reshape(batchsz, 2, 2, heiori, widori)

            mpi, blend_weight = self.forwardmpi(
                refims[:, 0],
                flow_gradx[:, 0],
                flow_grady[:, 0],
                mpi_warp=None
            )
            mpf = self.forwardmpf(mpi, flows[:, 0], blend_weight)
            mpi_warp = self.splat_func(
                flow01=mpf, mpi=mpi
            )
            mpi1, blend_weight1 = self.forwardmpi(
                refims[:, 1],
                flow_gradx[:, 1],
                flow_grady[:, 1],
                mpi_warp
            )
            # estimate depth map and sample sparse point depth
            depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            disp_hat1 = estimate_disparity_torch(mpi1, depth, blendweight=blend_weight1)
            flow_render = overcompose(mpi, blend_weight, ret_mask=False, blend_content=mpf)
            # render target view
            tarview, tarmask = shift_newview(mpi1, torch.reciprocal(depth * 0.05), ret_mask=True)

        val_dict = {}
        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            view0 = (tarview[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            # diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_newv"] = view0
            # val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt, depth[0, -1])
            val_dict["vis_dispgt"] = draw_dense_disp(disp_gts[:, 1], 1. / disp_gts[0, 1].max())
            val_dict["vis_disp"] = draw_dense_disp(disp_hat1, 1)
            val_dict["vis_mpf"] = flow_to_png_middlebury(make_grid(mpf[0]).detach().cpu().numpy())
            val_dict["vis_flow"] = flow_to_png_middlebury(flow_render[0].detach().cpu().numpy())
            val_dict["vis_flowgt"] = flow_to_png_middlebury(flows[0, 0].detach().cpu().numpy())

        return val_dict

    def forwardmpi(self, img, flowgx, flowgy, mpi_warp=None, ret_netout=False):
        # scheduling the s_bg
        netout = self.mpimodel(torch.cat([img, flowgx, flowgy], dim=1), mpi_warp)
        # cnlnum = netout.shape[1]
        # layernum = netout.la
        if mpi_warp is None:
            mpi, blend_weight = netout2mpi(netout,  # [:, self.mpimodel.num_layers - 1 + 3],
                                           img,
                                           ret_blendw=True)
        else:
            mpi, blend_weight = netoutupdatempi_maskfree(netout=netout,
                                                         img=img,
                                                         mpi_last=mpi_warp,
                                                         ret_blendw=True)
        if ret_netout:
            return mpi, blend_weight, netout
        else:
            return mpi, blend_weight

    def forwardmpf(self, mpi, flows, blend_weight, ret_intermedia=False):
        batchsz, planenum, _, heiori, widori = mpi.shape
        alpha = mpi[:, :, -1]
        if self.dilate_mpfin:
            alpha = dilate(alpha)
            blend_weight = torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                                      torch.ones([batchsz, 1, heiori, widori]).type_as(alpha)], dim=1)
        renderw = alpha * blend_weight
        if self.renderw2mpf:
            mpf = self.mpfmodel(torch.cat([renderw, flows], dim=1))
        else:
            mpf = self.mpfmodel(torch.cat([alpha, flows], dim=1))
        # mpf = self.mpfmodel(torch.cat([mpi[:, :, -1], flows[:, frameidx]], dim=1))
        mpf = mpf.reshape(batchsz, -1, 2, heiori, widori)
        if ret_intermedia:
            return mpf, renderw
        else:
            return mpf

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        layernum = self.mpimodel.num_layers
        batchsz, framenum, _, heiori, widori = refims.shape
        bfnum_1 = batchsz * (framenum - 1)
        # All are [B x Frame [x cnl] x H x W]
        with torch.no_grad():
            flows = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                    refims[:, 1:].reshape(bfnum_1, 3, heiori, widori))
            certainty_norm = certainty_maps.sum(dim=[-1, -2])
            flow_gradx, flow_grady = gradient(flows)
            flow_gradx = flow_gradx.abs().sum(dim=1).reshape(batchsz, framenum - 1, 1, heiori, widori)
            flow_grady = flow_grady.abs().sum(dim=1).reshape(batchsz, framenum - 1, 1, heiori, widori)
            flows = flows.reshape(batchsz, framenum - 1, 2, heiori, widori)
        mpi_warp = None
        mpi_out = []
        mpf_out = []
        intermediates = {
            "blend_weight": [],
            "mpf_renderw": [],
            "mpi_warp": []
        }
        # there will be framenum - 1 mpi,
        for frameidx in range(framenum - 1):
            mpi, blend_weight = self.forwardmpi(
                refims[:, frameidx],
                flow_gradx[:, frameidx],
                flow_grady[:, frameidx],
                mpi_warp
            )
            mpi_out.append(mpi)
            intermediates["blend_weight"].append(blend_weight)

            if frameidx == framenum - 2:  # don't need to estimate mpf in the final frame
                break

            mpf, mpfrenderw = self.forwardmpf(mpi, flows[:, frameidx], blend_weight, ret_intermedia=True)
            intermediates["mpf_renderw"].append(mpfrenderw)
            mpf_out.append(mpf)

            mpi_warp = self.splat_func(
                flow01=mpf, mpi=mpi
            )
            intermediates["mpi_warp"].append(mpi_warp)
        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        disps_hats = [
            estimate_disparity_torch(mpi_, depth, blendweight=bw_)
            for mpi_, bw_ in zip(mpi_out, intermediates["blend_weight"])
        ]
        # with torch.no_grad():  # compute scale
        disp_diffs = [
            disps_hats[i] / (torch.abs(disp_gts[:, i]) + 0.0001)
            for i in range(framenum - 1)
        ]
        # currently use first frame to compute scale
        scale = torch.exp((torch.log(disp_diffs[0]) * certainty_maps[:, 0]).sum(dim=[-1, -2]) / certainty_norm[:, 0])

        depth = depth * scale.reshape(-1, 1) * isleft
        # render target view
        tarviews = [
            shift_newview(mpi_, torch.reciprocal(depth), ret_mask=False)
            for mpi_ in mpi_out
        ]
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        # MPI loss
        for mpiframeidx in range(len(mpi_out)):
            if "pixel_loss" in self.loss_weight.keys():
                l1_loss = self.photo_loss(
                    tarviews[mpiframeidx],
                    tarims[:, mpiframeidx])
                final_loss += (l1_loss * self.loss_weight["pixel_loss"])
                if "pixel" not in loss_dict.keys():
                    loss_dict["pixel"] = l1_loss.detach()
                else:
                    loss_dict["pixel"] += l1_loss.detach()

            if "smooth_loss" in self.loss_weight.keys():
                smth_loss = smooth_grad(
                    disps_hats[mpiframeidx],
                    refims[:, mpiframeidx])
                smth_loss = smth_loss.mean()
                final_loss += (smth_loss * self.loss_weight["smooth_loss"])
                if "smth" not in loss_dict.keys():
                    loss_dict["smth"] = smth_loss.detach()
                else:
                    loss_dict["smth"] += smth_loss.detach()

            if "smooth_flowgrad_loss" in self.loss_weight.keys():
                disp_dx, disp_dy = gradient(disps_hats[mpiframeidx].unsqueeze(-3))
                smoothx = torch.max(disp_dx - 0.02, torch.tensor(0.).type_as(refims))
                smoothy = torch.max(disp_dy - 0.02, torch.tensor(0.).type_as(refims))
                with torch.no_grad():
                    weightsx = - torch.min(flow_gradx[:, mpiframeidx] / 2, torch.tensor(1.).type_as(refims)) + 1
                    weightsy = - torch.min(flow_grady[:, mpiframeidx] / 2, torch.tensor(1.).type_as(refims)) + 1
                smth_loss = smoothx * weightsx + smoothy * weightsy
                smth_loss = smth_loss.mean()
                final_loss += (smth_loss * self.loss_weight["smooth_flowgrad_loss"])
                if "flowgrad" not in loss_dict.keys():
                    loss_dict["flowgrad"] = smth_loss.detach()
                else:
                    loss_dict["flowgrad"] += smth_loss.detach()

            if "depth_loss" in self.loss_weight.keys():
                diff = torch.log(disp_diffs[mpiframeidx] / scale.reshape(-1, 1, 1))
                diff = (torch.pow(diff, 2) * certainty_maps).mean()
                final_loss += (diff * self.loss_weight["depth_loss"])
                if "depth" not in loss_dict.keys():
                    loss_dict["depth"] = diff.detach()
                else:
                    loss_dict["depth"] += diff.detach()

        # MPF loss
        for mpfframeidx in range(len(mpf_out)):
            if not self.learnmpf:
                break

            mpi = mpi_out[mpfframeidx]
            renderweight = intermediates["mpf_renderw"][mpfframeidx]
            mpf = mpf_out[mpfframeidx]
            if "flow_epe" in self.loss_weight.keys():
                flow_epe = torch.norm(mpf - flows[:, mpfframeidx].unsqueeze(1), dim=2)
                weight = torch.max(renderweight - 0.05, torch.tensor([0.]).type_as(mpi))
                flow_epe = flow_epe * weight
                epe = flow_epe.mean()
                final_loss += epe * self.loss_weight["flow_epe"]
                if "flow_epe" not in loss_dict.keys():
                    loss_dict["flow_epe"] = epe.detach()
                else:
                    loss_dict["flow_epe"] += epe.detach()

            if "flow_smth" in self.loss_weight.keys():
                mpf_gradx, mpf_grady = gradient(mpf.reshape(-1, 2, heiori, widori), order=self.flow_loss_ord)
                mpf_grad = (mpf_gradx.abs() + mpf_grady.abs()).sum(dim=-3).reshape(batchsz, -1, heiori, widori)
                if self.flow_loss_consider_weight:
                    print("Warning: flow_loss_consider_weight depricated")
                flow_smth = mpf_grad.mean()
                final_loss += flow_smth * self.loss_weight["flow_smth"]
                if "flow_smth" not in loss_dict.keys():
                    loss_dict["flow_smth"] = flow_smth.detach()
                else:
                    loss_dict["flow_smth"] += flow_smth.detach()

        if "tempdepth_loss" in self.loss_weight.keys():
            temporal = torch.tensor(0.).type_as(refims)
            for mpiframeidx in range(framenum - 2):
                with torch.no_grad():
                    disp_warp = forward_scatter(flows[:, mpiframeidx], disps_hats[mpiframeidx].unsqueeze(1)).squeeze(1)
                    mask = (disp_warp > 0).type_as(disp_warp)
                temporal += (torch.abs(disps_hats[mpiframeidx + 1] - disp_warp) * mask).mean()
            temporal /= len(mpi_out) - 1
            final_loss += (temporal * self.loss_weight["tempdepth_loss"])
            loss_dict["tempdepth"] = temporal.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class ModelandFullSVv1Loss(ModelandFullv1Loss):
    """
    The entire pipeline with sparse 3D point supervision
    """

    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__(models, cfg)

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        refims, tarims, refextrins, tarextrin, intrin, pt2ds, ptzs_gt = args
        batchsz, framenum, _, heiori, widori = refims.shape
        layernum = self.mpimodel.num_layers
        with torch.no_grad():
            flows = self.flow_estim(refims[:, :2].reshape(batchsz * 2, 3, heiori, widori),
                                    refims[:, 1:3].reshape(batchsz * 2, 3, heiori, widori))
            flow_gradx, flow_grady = gradient(flows)
            flow_gradx = flow_gradx.abs().sum(dim=1).reshape(batchsz, 2, 1, heiori, widori)
            flow_grady = flow_grady.abs().sum(dim=1).reshape(batchsz, 2, 1, heiori, widori)
            flows = flows.reshape(batchsz, 2, 2, heiori, widori)

            mpi, blend_weight = self.forwardmpi(
                refims[:, 0],
                flow_gradx[:, 0],
                flow_grady[:, 0],
                mpi_warp=None
            )
            mpf = self.forwardmpf(mpi, flows[:, 0], blend_weight)
            mpi_warp = self.splat_func(
                flow01=mpf, mpi=mpi
            )
            mpi1, blend_weight1 = self.forwardmpi(
                refims[:, 1],
                flow_gradx[:, 1],
                flow_grady[:, 1],
                mpi_warp
            )
            # estimate depth map and sample sparse point depth
            depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            disp_hat1 = estimate_disparity_torch(mpi1, depth, blendweight=blend_weight1)
            flow_render = overcompose(mpi, blend_weight, ret_mask=False, blend_content=mpf)
            # render target view
            tarview, tarmask = shift_newview(mpi1, torch.reciprocal(depth * 0.05), ret_mask=True)

        val_dict = {}
        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            view0 = (tarview[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            # diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_newv"] = view0
            # val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            val_dict["vis_disp"] = draw_dense_disp(disp_hat1, 1)
            val_dict["vis_mpf"] = flow_to_png_middlebury(make_grid(mpf[0]).detach().cpu().numpy())
            val_dict["vis_flow"] = flow_to_png_middlebury(flow_render[0].detach().cpu().numpy())
            val_dict["vis_flowgt"] = flow_to_png_middlebury(flows[0, 0].detach().cpu().numpy())

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, refextrins, tarextrin, intrin, pt2ds, ptzs_gts = args
        layernum = self.mpimodel.num_layers
        batchsz, framenum, _, heiori, widori = refims.shape
        bfnum_1 = batchsz * (framenum - 1)
        # All are [B x Frame [x cnl] x H x W]
        with torch.no_grad():
            flows = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                    refims[:, 1:].reshape(bfnum_1, 3, heiori, widori))
            flow_gradx, flow_grady = gradient(flows)
            flow_gradx = flow_gradx.abs().sum(dim=1).reshape(batchsz, framenum - 1, 1, heiori, widori)
            flow_grady = flow_grady.abs().sum(dim=1).reshape(batchsz, framenum - 1, 1, heiori, widori)
            flows = flows.reshape(batchsz, framenum - 1, 2, heiori, widori)
        mpi_warp = None
        mpi_out = []
        mpf_out = []
        intermediates = {
            "blend_weight": [],
            "mpf_renderw": [],
            "mpi_warp": []
        }
        # there will be framenum - 1 mpi,
        for frameidx in range(framenum - 1):
            mpi, blend_weight = self.forwardmpi(
                refims[:, frameidx],
                flow_gradx[:, frameidx],
                flow_grady[:, frameidx],
                mpi_warp
            )
            mpi_out.append(mpi)
            intermediates["blend_weight"].append(blend_weight)

            if frameidx == framenum - 2:  # don't need to estimate mpf in the final frame
                break

            mpf, mpfrenderw = self.forwardmpf(mpi, flows[:, frameidx], blend_weight, ret_intermedia=True)
            intermediates["mpf_renderw"].append(mpfrenderw)
            mpf_out.append(mpf)

            mpi_warp = self.splat_func(
                flow01=mpf, mpi=mpi
            )
            intermediates["mpi_warp"].append(mpi_warp)
        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        disps_hats = [
            estimate_disparity_torch(mpi_, depth, blendweight=bw_)
            for mpi_, bw_ in zip(mpi_out, intermediates["blend_weight"])
        ]
        ptdis_es = [
            torchf.grid_sample(disps_hats[i].unsqueeze(1),
                               pt2ds[:, i].reshape(batchsz, 1, -1, 2)).reshape(batchsz, -1)
            for i in range(len(disps_hats))
        ]
        # currently use first frame to compute scale
        scale = torch.exp(torch.log(ptdis_es[0] * ptzs_gts[:, 0]).mean(dim=-1, keepdim=True))

        depth = depth * scale.reshape(-1, 1)
        # render target view
        tarviews = [
            render_newview(mpi=mpi_out[i],
                           srcextrin=refextrins[:, i],
                           tarextrin=tarextrin,
                           intrin=intrin,
                           depths=depth,
                           ret_mask=False)
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
                    disps_hats[mpiframeidx],
                    refims[:, mpiframeidx])
                smth_loss = smth_loss.mean()
                final_loss += (smth_loss * self.loss_weight["smooth_loss"])
                if "smth" not in loss_dict.keys():
                    loss_dict["smth"] = smth_loss.detach()
                else:
                    loss_dict["smth"] += smth_loss.detach()

            if "smooth_flowgrad_loss" in self.loss_weight.keys():
                disp_dx, disp_dy = gradient(disps_hats[mpiframeidx].unsqueeze(-3))
                smoothx = torch.max(disp_dx - 0.02, torch.tensor(0.).type_as(refims))
                smoothy = torch.max(disp_dy - 0.02, torch.tensor(0.).type_as(refims))
                with torch.no_grad():
                    weightsx = - torch.min(flow_gradx[:, mpiframeidx] / 2, torch.tensor(1.).type_as(refims)) + 1
                    weightsy = - torch.min(flow_grady[:, mpiframeidx] / 2, torch.tensor(1.).type_as(refims)) + 1
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

        # MPF loss
        for mpfframeidx in range(len(mpf_out)):
            if not self.learnmpf:
                break

            mpi = mpi_out[mpfframeidx]
            renderweight = intermediates["mpf_renderw"][mpfframeidx]
            mpf = mpf_out[mpfframeidx]
            if "flow_epe" in self.loss_weight.keys():
                flow_epe = torch.norm(mpf - flows[:, mpfframeidx].unsqueeze(1), dim=2)
                weight = torch.max(renderweight - 0.05, torch.tensor([0.]).type_as(mpi))
                flow_epe = flow_epe * weight
                epe = flow_epe.mean()
                final_loss += epe * self.loss_weight["flow_epe"]
                if "flow_epe" not in loss_dict.keys():
                    loss_dict["flow_epe"] = epe.detach()
                else:
                    loss_dict["flow_epe"] += epe.detach()

            if "flow_smth" in self.loss_weight.keys():
                mpf_gradx, mpf_grady = gradient(mpf.reshape(-1, 2, heiori, widori), order=self.flow_loss_ord)
                mpf_grad = (mpf_gradx.abs() + mpf_grady.abs()).sum(dim=-3).reshape(batchsz, -1, heiori, widori)
                if self.flow_loss_consider_weight:
                    print("Warning: flow_loss_consider_weight depricated")
                flow_smth = mpf_grad.mean()
                final_loss += flow_smth * self.loss_weight["flow_smth"]
                if "flow_smth" not in loss_dict.keys():
                    loss_dict["flow_smth"] = flow_smth.detach()
                else:
                    loss_dict["flow_smth"] += flow_smth.detach()

        if "tempdepth_loss" in self.loss_weight.keys():
            temporal = torch.tensor(0.).type_as(refims)
            for mpiframeidx in range(framenum - 2):
                with torch.no_grad():
                    disp_warp = forward_scatter(flows[:, mpiframeidx], disps_hats[mpiframeidx].unsqueeze(1)).squeeze(1)
                    mask = (disp_warp > 0).type_as(disp_warp)
                temporal += (torch.abs(disps_hats[mpiframeidx + 1] - disp_warp) * mask).mean()
            temporal /= len(mpi_out) - 1
            final_loss += (temporal * self.loss_weight["tempdepth_loss"])
            loss_dict["tempdepth"] = temporal.detach()

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
        self.sfmodel = models["SceneFlow"]
        self.afmodel = models["AppearanceFlow"]

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.photo_loss = select_photo_loss(self.pixel_loss_mode).cuda()
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])
        self.splat_func = forward_scatter_withweight

        # adjustable config
        self.pipe_optim_frame0 = self.loss_weight.pop("pipe_optim_frame0", False)
        self.sceneflow_mode = self.loss_weight.pop("sflow_mode", "backward")
        self.aflow_residual = self.loss_weight.pop("aflow_residual", True)
        self.aflow_includeself = self.loss_weight.pop("aflow_includeself", False)
        self.depth_loss_mode = self.loss_weight.pop("depth_loss_mode", "norm")
        self.depth_loss_ord = self.loss_weight.pop("depth_loss_ord", 1)
        self.depth_loss_rmresidual = self.loss_weight.pop("depth_loss_rmresidual", True)

        print(f"PipelineV2 activated config:\n"
              f"pipe_optim_frame0: {self.pipe_optim_frame0}\n"
              f"sceneflow_mode: {self.sceneflow_mode}\n"
              f"aflow_residual: {self.aflow_residual}\n"
              f"aflow_includeself: {self.aflow_includeself}\n"
              f"depth_loss_mode: {self.depth_loss_mode}\n"
              f"depth_loss_ord: {self.depth_loss_ord}\n"
              f"depth_loss_rmresidual: {self.depth_loss_rmresidual}\n")

        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.eval()
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)

        # self.flow_loss_ord = self.loss_weight.pop("flow_smth_ord", 2)
        # self.flow_loss_consider_weight = self.loss_weight.pop("flow_smth_bw", False)

        # used for backward warp
        self.offset = None
        self.offset_bhw2 = None

        # used for inference
        self.neighbor_sz = 1
        self.max_win_sz = self.neighbor_sz * 2 + 1
        self.img_window = deque()
        self.flowf_win, self.flowb_win = deque(), deque()
        self.disp_warp = None
        self.lastmpi = None

    def infer_forward(self, img: torch.Tensor, ret_cfg: str):
        """
        new frames will be pushed into a queue, and will only output if len(queue)==self.maxwinsz
        """
        if "restart" in ret_cfg:
            self.img_window.clear()
            self.flowb_win.clear()
            self.flowf_win.clear()
            self.disp_warp = None
            self.lastmpi = None

        if "nolast" in ret_cfg:
            self.disp_warp = None
            self.lastmpi = None

        if img.dim() == 3:
            img = img.unsqueeze(0)
        bsz, _, hei, wid = img.shape
        self.prepareoffset(wid, hei)
        self.img_window.append(img)
        if len(self.img_window) > 1:
            self.flowf_win.append(self.flow_estim(self.img_window[-2], self.img_window[-1]))
            self.flowb_win.append(self.flow_estim(self.img_window[-1], self.img_window[-2]))

        if len(self.img_window) > self.max_win_sz:
            self.img_window.popleft()
            self.flowf_win.popleft()
            self.flowb_win.popleft()
        elif len(self.img_window) < self.max_win_sz:
            return None

        flow_list = [self.flowb_win[0], self.flowf_win[1]]
        img_list = [self.img_window[0], self.img_window[2]]

        alpha = self.forwardmpi(
            None,
            self.img_window[1],
            flow_list,
            self.disp_warp
        )
        depth = make_depths(alpha.shape[1]).type_as(alpha).unsqueeze(0).repeat(1, 1)
        disp, blend_weight = estimate_disparity_torch(alpha.unsqueeze(2), depth, retbw=True)
        if self.aflow_includeself or "aflow_includeself" in ret_cfg:
            flow_list.append(torch.zeros_like(flow_list[-1]))
            img_list.append(self.img_window[1])
        if "aflow_selfonly" in ret_cfg:
            flow_list = [torch.zeros_like(flow_list[-1])]
            img_list = [self.img_window[1]]
        imbg = self.forwardaf(alpha, disp,
                              flows=flow_list,
                              mpis_last=self.img_window[0] if self.lastmpi is None else self.lastmpi[:, :, :3],
                              imgs_list=img_list)
        mpi = alpha2mpi(alpha, self.img_window[1], imbg, blend_weight=blend_weight)

        # update last frame info
        self.lastmpi = mpi
        self.disp_warp = self.forwardsf(self.flowf_win[1], self.flowb_win[1], disp)
        return mpi

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        batchsz, framenum, _, heiori, widori = refims.shape
        self.prepareoffset(widori, heiori)
        layernum = self.mpimodel.num_layers
        bfnum_1 = batchsz * (framenum - 1)
        bfnum_2 = batchsz * (framenum - 2)
        with torch.no_grad():
            flows_f = self.flow_estim(refims[:, :-1].reshape(bfnum_1, 3, heiori, widori),
                                      refims[:, 1:].reshape(bfnum_1, 3, heiori, widori))
            flows_b = self.flow_estim(refims[:, 1:-1].reshape(bfnum_2, 3, heiori, widori),
                                      refims[:, :-2].reshape(bfnum_2, 3, heiori, widori))
            certainty_norm = certainty_maps.sum(dim=[-1, -2])
            flows_f = flows_f.reshape(batchsz, framenum - 1, 2, heiori, widori)
            flows_b = flows_b.reshape(batchsz, framenum - 2, 2, heiori, widori)
            depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

            disp_list = []

            alpha0 = self.forwardmpi(
                None,
                refims[:, 0],
                [flows_f[:, 0]],
                None
            )
            disp0 = estimate_disparity_torch(alpha0.unsqueeze(2), depth)
            disp_warp = self.forwardsf(flows_f[:, 0], flows_b[:, 0], disp0)
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
            tarview1 = shift_newview(mpi1, torch.reciprocal(depth * 0.15))

        val_dict = {}
        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            # diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_newv"] = (tarview1[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            # val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt, depth[0, -1])
            val_dict["vis_dispgt"] = draw_dense_disp(disp_gts[:, 1], 1. / disp_gts[0, 1].max())
            val_dict["vis_disp"] = draw_dense_disp(disp1, 1)
            if imbg1.dim() == 5:
                imbg1 = imbg1[:, 9]
            val_dict["vis_bgimg"] = (imbg1[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()

        return val_dict

    def prepareoffset(self, wid, hei):
        if self.offset is None:
            offsety, offsetx = torch.meshgrid([
                torch.linspace(0, hei - 1, hei),
                torch.linspace(0, wid - 1, wid)
            ])
            self.offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).float().cuda()
            self.offset_bhw2 = self.offset.permute(0, 2, 3, 1).contiguous()

    def forwardmpi(self, tmpcfg, img, flow_list: list, disp_warp=None):
        """
        Now this only return alpha channles (like depth estimation)
        """
        batchsz, cnl, hei, wid = img.shape
        flows = torch.cat(flow_list, dim=0)
        with torch.no_grad():
            flow_gradx, flow_grady = gradient(flows)
            flow_gradx = flow_gradx.abs().sum(dim=1).reshape(-1, batchsz, 1, hei, wid)
            flow_grady = flow_grady.abs().sum(dim=1).reshape(-1, batchsz, 1, hei, wid)
            flow_gradx = torch.max(flow_gradx, dim=0)[0]
            flow_grady = torch.max(flow_grady, dim=0)[0]
        # scheduling the s_bg
        if tmpcfg is not None:
            tmpcfg["flow_gradx"].append(flow_gradx)
            tmpcfg["flow_grady"].append(flow_grady)
        if disp_warp is None:
            disp_warp = torch.zeros_like(img[:, :1])
        alpha = self.mpimodel(torch.cat([img, flow_gradx, flow_grady, disp_warp], dim=1))

        alpha = torch.cat([torch.ones([batchsz, 1, hei, wid]).type_as(alpha), alpha], dim=1)
        return alpha

    def forwardsf(self, flowf, flowb, disparity, intcfg=None):
        if self.sceneflow_mode == "forward":
            occ_f = warp_flow(flowb, flowf, offset=self.offset_bhw2) + flowf
            occ_f = torch.sum(torch.abs(occ_f), dim=1, keepdim=True)
            occ_mask = torch.exp(-occ_f * 0.5)

            sflow = self.sfmodel(flowf, disparity)
            disp_warp = self.splat_func(
                flow01=flowf,
                content=disparity.unsqueeze(1) + sflow,
                softmask=occ_mask,
                offset=self.offset
            )
            if intcfg is not None:
                intcfg["sflow"].append(sflow)
                intcfg["occ_mask"].append(occ_mask)

        elif self.sceneflow_mode == "backward":
            sflow = self.sfmodel(flowf, disparity)
            flowf_warp = warp_flow(flowf, flowb, offset=self.offset_bhw2, pad_mode="border")
            # scheme1: warp using flowb and remove occmask
            # occ_b = flowf_warp + flowb
            # occ_b = torch.sum(torch.abs(occ_b), dim=1, keepdim=True)
            # occ_mask = (occ_b > 2)
            # disp_warp = warp_flow(
            #     content=disparity.unsqueeze(1) + sflow,
            #     flow=flowb,
            #     offset=self.offset_bhw2
            # )
            # disp_warp[occ_mask] = 0
            # scheme2: warp using -flowf_warp
            disp_warp = warp_flow(
                content=disparity.unsqueeze(1) + sflow,
                flow=-flowf_warp,
                offset=self.offset_bhw2,
                pad_mode="border"
            )
            if intcfg is not None:
                intcfg["sflow"].append(sflow)
                # intcfg["occ_mask"].append(occ_mask)

        elif self.sceneflow_mode == "no":
            if self.sfmodel is not None:
                print(f"PipelineV2:: not using scene flow model, releasing memory..")
                del self.sfmodel
                self.sfmodel = None
            occ_b = warp_flow(flowf, flowb, offset=self.offset_bhw2) + flowb
            occ_b = torch.sum(torch.abs(occ_b), dim=1, keepdim=True)
            occ_mask = (occ_b > 2)
            disp_warp = warp_flow(
                content=disparity.unsqueeze(1),
                flow=flowb,
                offset=self.offset_bhw2
            )
            disp_warp[occ_mask] = 0
            if intcfg is not None:
                intcfg["occ_mask"].append(occ_mask)
        else:
            raise RuntimeError(f"Pipelinev2:: unrecognized token self.sceneflow_mode = {self.sceneflow_mode}")
        return disp_warp

    def forwardaf(self, alphas, disparity, flows: list, mpis_last, imgs_list: list):
        """
        Guarantee the flows[0] and imgs_list[0] to be the last frame
        """
        assert len(flows) == len(imgs_list)
        if isinstance(self.afmodel, (ASPFNetDIn, ASPFNetAIn)):
            structure_in = disparity if isinstance(self.afmodel, ASPFNetDIn) else alphas
            bgs = []
            masks = []
            for flow, img in zip(flows, imgs_list):
                bgflow = self.afmodel(flow, structure_in)
                if self.aflow_residual:
                    mask = torch.abs(bgflow).sum(dim=1)
                    bgflow = flow + bgflow
                else:
                    mask = torch.abs(bgflow - flow).sum(dim=1)
                bg = warp_flow(img, bgflow, offset=self.offset_bhw2)
                bgs.append(bg)
                masks.append(mask)
            bgs = torch.stack(bgs)
            masks = torch.stack(masks).unsqueeze(2)
            weight = torchf.softmax(masks, dim=0)
            bg = (bgs * weight).sum(dim=0)

        elif isinstance(self.afmodel, (ASPFNetWithMaskInOut, ASPFNetWithMaskOut)):
            masks, bgs = [], []
            lastmask = None
            for flow, img in zip(flows, imgs_list):
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

        elif isinstance(self.afmodel, AMPFNetAIn2D):
            bgflows = self.afmodel(flows[0], alphas)
            if self.aflow_residual:
                bgflows += flows[0].unsqueeze(1)
            if mpis_last.dim() == 4:
                mpis_last = mpis_last.unsqueeze(1).expand(-1, self.mpimodel.num_layers, -1, -1, -1)
            bg = warp_flow(mpis_last, bgflows, self.offset_bhw2)

        else:
            raise RuntimeError(f"model not compatible")

        return bg

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        layernum = self.mpimodel.num_layers
        # remove first frame's gt
        tarims = tarims[:, 1:]
        disp_gts = disp_gts[:, 1:]
        shift_gt = isleft[:, 1]
        isleft = isleft[:, 0]
        certainty_maps = certainty_maps[:, 1:]

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
            certainty_norm = certainty_maps.sum(dim=[-1, -2])
            flows_f = flows_f.reshape(batchsz, framenum - 1, 2, heiori, widori)
            flows_b = flows_b.reshape(batchsz, framenum - 2, 2, heiori, widori)
        disp_out = []
        mpi_out = []
        intermediates = {
            "flow_gradx": [],
            "flow_grady": [],
            "blend_weight": [],
            "sflow": [],
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
            disp_warp = self.forwardsf(flow0_1, flows_b[:, 0], disp)
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
                intcfg=intermediates
            )
            intermediates["disp_warp"].append(disp_warp)

        # scale and shift invariant
        # with torch.no_grad():  # compute scale
        disp_diffs = [
            disp_out[i] / (torch.abs(disp_gts[:, i] - shift_gt.reshape(batchsz, 1, 1)) + 0.0001)
            for i in range(framenum - 2)
        ]
        # currently use first frame to compute scale and shift
        scale = torch.exp((torch.log(disp_diffs[0]) * certainty_maps[:, 0]).sum(dim=[-1, -2]) / certainty_norm[:, 0])

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
                diff = torch.log(disp_diffs[mpiframeidx] / scale.reshape(-1, 1, 1))
                diff = (torch.pow(diff, 2) * certainty_maps[:, mpiframeidx]).mean()
                final_loss += (diff * self.loss_weight["depth_loss"])
                if "depth" not in loss_dict.keys():
                    loss_dict["depth"] = diff.detach()
                else:
                    loss_dict["depth"] += diff.detach()

        if "sflow_loss" in self.loss_weight.keys() and len(intermediates["sflow"]) > 0:
            diffs = torch.tensor(0.).type_as(refims)
            for sflow_idx in range(len(intermediates["sflow"])):
                sflow = intermediates["sflow"][sflow_idx]
                occ_mask = intermediates["occ_mask"][sflow_idx]
                if self.sceneflow_mode == "forward":
                    dispgt_warp = self.splat_func(
                        flow01=flows_f[:, sflow_idx + 1],
                        content=(disp_gts[:, sflow_idx] - shift_gt) * scale.unsqueeze(-1) + sflow,
                        softmask=occ_mask,
                        offset=self.offset
                    )
                elif self.sceneflow_mode == "backward":
                    dispgt_warp = warp_flow(
                        content=(disp_gts[:, sflow_idx] - shift_gt) * scale.reshape(batchsz, 1, 1, 1) + sflow,
                        flow=flows_b[:, sflow_idx + 1],
                        offset=self.offset_bhw2
                    )
                    dispgt_warp[occ_mask] = 0
                else:
                    raise NotImplementedError
                diff = torch.abs(dispgt_warp -
                                 (disp_gts[:, sflow_idx + 1] - shift_gt) * scale.reshape(batchsz, 1, 1, 1)).mean()
                diffs += diff
            diffs /= len(intermediates["sflow"])
            final_loss += (diffs * self.loss_weight["sflow_loss"])
            loss_dict["sflow"] = diffs.detach()

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
            disp_warp = self.forwardsf(flows_f[:, 0], flows_b, disp0)
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
            "sflow": [],
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
            disp_warp = self.forwardsf(flow0_1, flows_b[:, 0], disp)
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
                intcfg=intermediates
            )
            intermediates["disp_warp"].append(disp_warp)

        # with torch.no_grad():  # compute scale
        ptdis_es = [
            torchf.grid_sample(disp_out[i].unsqueeze(1),
                               pt2ds[:, i:i+1]).reshape(batchsz, -1)
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

        if "sflow_loss" in self.loss_weight.keys() and len(intermediates["sflow"]) > 0:
            print("PipelineSVV2::Warning, sflow not implemented, will not estimate sflow_loss")

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
