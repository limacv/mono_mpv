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

            l1_map = photo_l1loss(tarview, tarim)
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
            l1_loss = photometric_loss(tarview, tarim, mode=self.pixel_loss_mode)
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


class ModelandTimeLoss(nn.Module):
    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        assert "loss_weights" in cfg.keys(), "ModelAndLossBase: didn't find 'loss_weights' in cfg"
        self.loss_weight = cfg["loss_weights"].copy()

        torch.set_default_tensor_type(torch.FloatTensor)

        self.model = model
        self.model.train()
        self.model.cuda()

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.scheduler = ParamScheduler([3e4, 3e5], [0.5, 1])
        # self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "ssim")
        self.last_feature = None

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        """
        basically same as train_forward
        only evaluate L1 loss
        also output displacement heat map & newview image
        """
        args = [_t.cuda() for _t in args]
        refim, tarim, refextrin, tarextrin, intrin, pt2ds, ptzs_gt = args
        batchsz, framenum, _, heiori, widori = refim.shape
        evaluation_frame_list = [0, framenum - 1]

        val_dict = {}
        last_feat = None
        mpilist = []
        with torch.no_grad():
            self.model.eval()
            for frameidx in range(framenum):
                if isinstance(self.model, MPINet):
                    netout = self.model(refim[:, frameidx])
                elif isinstance(self.model, MPVNet):
                    netout, last_feat = self.model(refim[:, frameidx], last_feat)
                else:
                    print(f"ModelandTimeLoss:: model type {type(self.model)} not recognized")

                if frameidx in evaluation_frame_list:
                    # compute mpi from netout
                    mpi, blend_weight = netout2mpi(netout, refim[:, frameidx], ret_blendw=True)
                    mpilist.append((mpi, blend_weight))

            # estimate depth map and sample sparse point depth
            depth = make_depths(32).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            for i, (mpi, blend_weight) in enumerate(mpilist):
                disparity = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)

                if "visualize" in kwargs.keys() and kwargs["visualize"]:
                    # view0 = (tarview[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
                    # diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
                    # val_dict["vis_newv"] = view0
                    # val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                    # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt, depth[0, -1])
                    val_dict[f"vis_disp{i}"] = draw_dense_disp(disparity, 1)
                    val_dict["save_mpi"] = mpi[0]
                    self.model.train()
        return val_dict

    def forward(self, *args, **kwargs):
        # scheduling
        step = kwargs["step"] if "step" in kwargs else 0
        bg_pct = self.scheduler.get_value(step)

        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarim, refextrins, tarextrin, intrin, pt2ds, ptzs_gts = args
        # if refims.shape[0] != 0:
        #     raise NotImplementedError("now only batch size 1 is supported")
        batchsz, framenum, _, heiori, widori = refims.shape

        mpis, blend_weights = [], []
        feat_last = None

        for frameidx in range(0, framenum):
            # infer forward
            if isinstance(self.model, MPINet):
                netout = self.model(refims[:, frameidx])
            elif isinstance(self.model, MPVNet):
                netout, feat_last = self.model(refims[:, frameidx], feat_last)
            else:
                raise NotImplementedError(f"ModelandTimeLoss:: model type {type(self.model)} not recognized")

            mpi, blend_weight = netout2mpi(netout, refims[:, frameidx], bg_pct=bg_pct, ret_blendw=True)
            mpis.append(mpi)
            blend_weights.append(blend_weight)

        layernum = mpis[0].shape[1]
        bsz_fnum = batchsz * framenum
        mpis = torch.stack(mpis, dim=1).reshape(bsz_fnum, layernum, 4, heiori, widori)
        blend_weights = torch.stack(blend_weights, dim=1).reshape(bsz_fnum, layernum, heiori, widori)

        # estimate depth map and sample sparse point depth
        # use first frame's scale as the global scale
        depth = make_depths(32).type_as(mpis).unsqueeze(0).repeat(bsz_fnum, 1)
        disparitys = estimate_disparity_torch(mpis, depth, blendweight=blend_weights)
        ptdis_e = torchf.grid_sample(disparitys.unsqueeze(1),
                                     pt2ds.reshape(bsz_fnum, 1, -1, 2)).reshape(batchsz, framenum, -1)

        # TODO: decide whether scale = mean on time dim is good, or simply use first frame's scale
        scale = torch.exp(torch.log(ptdis_e * ptzs_gts).mean(dim=[-1, -2], keepdim=True))  # [batchsz x 1 x 1]
        depth = (scale * depth.reshape(batchsz, framenum, layernum)).reshape(bsz_fnum, layernum)

        # expand single frame tensor to temporal tensor
        tarextrin = tarextrin.reshape(batchsz, 1, 3, 4).expand(-1, framenum, -1, -1)
        tarim = tarim.reshape(batchsz, 1, 3, heiori, widori).expand(-1, framenum, -1, -1, -1)
        intrin = intrin.reshape(batchsz, 1, 3, 3).expand(-1, framenum, -1, -1)

        # render target view
        (tarviews, tarmasks), tardisps = render_newview(mpi=mpis,  # [batchsz*frameNum x layerNum x 4 x hei x wid]
                                                        srcextrin=refextrins.reshape(bsz_fnum, 3, 4),
                                                        tarextrin=tarextrin.reshape(bsz_fnum, 3, 4),
                                                        # or tarextrin.repeat_interleave(framenum, 3, 4)
                                                        intrin=intrin.reshape(bsz_fnum, 3, 3),
                                                        depths=depth,  # [batchsz*frameNum x layerNum]
                                                        ret_mask=True, ret_dispmap=True)
        # tarviews: tensor of shape [batchsz*frameNum x 3 x H x W]
        # tarmasks: tensor of shape [batchsz*frameNum x H x W] (per-frame mask)
        # tardisps: tensor of shape [batchsz*frameNum x H x W]
        # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt)
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        intermediate_var = {}  # used for store intermediate variable to reduce redundant computation
        loss_dict["scale"] = scale.detach().mean()
        loss_dict["s_bg"] = torch.tensor(bg_pct).type_as(scale.detach())
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = photometric_loss(imref=tarim.reshape(bsz_fnum, 3, heiori, widori),
                                       imreconstruct=tarviews,
                                       mode=self.pixel_loss_mode)
            l1_loss = (l1_loss * tarmasks).sum() / tarmasks.sum()
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = l1_loss.detach()

        if "pixel_std_loss" in self.loss_weight.keys():
            tarview_mean = tarviews.reshape(batchsz, framenum, 3, heiori, widori).mean(dim=1, keepdim=True)
            valid_mask = tarmasks.reshape(batchsz, framenum, heiori, widori).prod(dim=1, keepdim=True)
            intermediate_var["per_batch_mask"] = valid_mask
            std_loss = torch.abs(tarviews.reshape(batchsz, framenum, 3, heiori, widori) - tarview_mean).sum(dim=2)
            std_loss = (std_loss * valid_mask).mean()
            final_loss += (std_loss * self.loss_weight["pixel_std_loss"])
            loss_dict["pixel_std"] = std_loss.detach()

        if "smooth_loss" in self.loss_weight.keys():
            smth_loss = smooth_grad(disparitys, refims.reshape(bsz_fnum, 3, heiori, widori))
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"])
            loss_dict["smth"] = smth_loss.detach()

        if "smooth_tar_loss" in self.loss_weight.keys():
            print("ModelWithLoss::warning! smooth_tar_loss not tested")
            smth_loss = smooth_grad(tardisps, tarim.reshape(bsz_fnum, 3, heiori, widori))
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_tar_loss"])
            loss_dict["smthtar"] = smth_loss.detach()

        if "temporal_loss" in self.loss_weight.keys():
            tardisp_mean = tardisps.reshape(batchsz, framenum, heiori, widori).mean(dim=1, keepdim=True)
            valid_mask = tarmasks.reshape(batchsz, framenum, heiori, widori).prod(dim=1, keepdim=True) \
                if "per_batch_mask" not in intermediate_var.keys() else intermediate_var["per_batch_mask"]
            temp_loss = torch.abs(tardisps.reshape(batchsz, framenum, heiori, widori) - tardisp_mean)
            temp_loss = (temp_loss * valid_mask).mean()
            final_loss += (temp_loss * self.loss_weight["temporal_loss"])
            loss_dict["temp"] = temp_loss.detach()

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(ptdis_e * ptzs_gts / scale)
            diff = torch.pow(diff, 2).mean()  # mean among: batch & frames & points
            final_loss += (diff * self.loss_weight["depth_loss"])
            loss_dict["depth"] = diff.detach()

        if "sparse_loss" in self.loss_weight.keys():
            alphas = mpis[:, :, -1, :, :]  # shape [batchsz*framenum x layerNum x H x W]
            alphas_norm = alphas / torch.norm(alphas, dim=1, keepdim=True)
            sparse_loss = torch.sum(torch.abs(alphas_norm), dim=1).mean()
            final_loss += sparse_loss * self.loss_weight["sparse_loss"]
            loss_dict["sparse_loss"] = sparse_loss.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}

    def infer_forward(self, im: torch.Tensor, mode='pad_constant', restart=False):
        """
        Restart will reset the last_feature
        """
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

            if restart:
                self.last_feature = None
            self.model.eval()

            if isinstance(self.model, MPINet):
                netout = self.model(img_padded)
            elif isinstance(self.model, MPVNet):
                netout, self.last_feature = self.model(img_padded, self.last_feature)
            else:
                raise NotImplementedError(f"ModelandTimeLoss:: model type {type(self.model)} not recognized")

            # depad
            if mode == 'resize':
                netout = Resize([hei_new, wid_new])(netout)
            else:
                netout = netout[..., padding[2]: hei_new - padding[3], padding[0]: wid_new - padding[1]]
            mpi = netout2mpi(netout, im)
            self.model.train()

        return mpi


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

            l1_map = photo_l1loss(tarview, tarim)
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
            l1_loss = photometric_loss(tarview, tarim, mode=self.pixel_loss_mode)
            l1_loss = l1_loss.mean()
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


class ModelandFlowLoss(nn.Module):
    """
    takes model that with two frames input, output mpf directly
    Model that estimate multi-plane flow
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
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["kitti"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)
        self.flow_estim.eval()

        self.last_img = None
        self.last_netout = None

    def infer_forward(self, img: torch.Tensor, restart=False):
        if restart:
            self.last_img = None
            self.last_netout = None

        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.cuda()
        batchsz, _, heiori, widori = img.shape
        layernum = self.model.num_layers
        with torch.no_grad():
            if self.last_img is None:
                self.last_img = img
                return torch.ones(batchsz, layernum, 4, heiori, widori).type_as(img), \
                       torch.ones(batchsz, layernum, 2, heiori, widori).type_as(img)
            else:
                self.model.eval()
                netout, mpf = self.model(torch.cat([img, self.last_img], dim=1))
                self.model.train()

                mpf = mpf.reshape(batchsz, layernum, 2, heiori, widori)

                return netout2mpi(netout, img), mpf

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        """
        basically same as train_forward
        only evaluate L1 loss
        also output displacement heat map & newview image
        """
        args = [_t.cuda() for _t in args]
        refim, tarim, disp_gt, certainty_map, isleft = args
        with torch.no_grad():
            flowgt = self.flow_estim(refim[:, 1], refim[:, 0])
        refim, refim_last = refim[:, 1], refim[:, 0]
        tarim = tarim[:, 1]
        disp_gt = disp_gt[:, 1]
        certainty_map = certainty_map[:, 1]
        batchsz, _, heiori, widori = refim.shape
        val_dict = {}
        with torch.no_grad():
            self.model.eval()
            netout, mpf = self.model(torch.cat([refim, refim_last], dim=1))
            self.model.train()

            layernum = self.model.num_layers
            mpf = mpf.reshape(batchsz, layernum, 2, heiori, widori)
            mpi, blend_weight = netout2mpi(netout, refim, ret_blendw=True)

            flow_render = overcompose(mpi, blend_weight, ret_mask=False, blend_content=mpf)
            epe = torch.norm(flow_render - flowgt, dim=1, p=1).mean()

            # estimate depth map and sample sparse point depth
            depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            disp_hat = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)

            # compute scale
            certainty_norm = certainty_map.sum(dim=[-1, -2])
            disp_scale = disp_hat / (torch.abs(disp_gt) + 0.01)
            scale = torch.exp((torch.log(disp_scale) * certainty_map).sum(dim=[-1, -2]) / certainty_norm)

            depth = depth * scale.reshape(-1, 1) * isleft
            depthloss = torch.log(disp_scale / scale.reshape(-1, 1, 1))
            depthloss = (torch.pow(depthloss, 2) * certainty_map).mean()
            # render target view
            tarview, tarmask = shift_newview(mpi, torch.reciprocal(depth), ret_mask=True)

            l1_map = photo_l1loss(tarview, tarim)
            l1_loss = (l1_map * tarmask).sum() / tarmask.sum()
            val_dict["val_l1diff"] = float(l1_loss)
            val_dict["val_scale"] = float(scale.mean())
            val_dict["val_depth"] = float(depthloss)
            val_dict["val_epe"] = float(epe)

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            view0 = (tarview[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            # diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_newv"] = view0
            # val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt, depth[0, -1])
            val_dict["vis_dispgt"] = draw_dense_disp(disp_gt, 1. / disp_gt.max())
            val_dict["vis_disp"] = draw_dense_disp(disp_hat, 1)
            val_dict["vis_flow"] = flow_to_png_middlebury(flow_render[0].detach().cpu().numpy())
            val_dict["vis_mpf"] = flow_to_png_middlebury(mpf[0, 16].detach().cpu().numpy())
            val_dict["vis_flowgt"] = flow_to_png_middlebury(flowgt[0].detach().cpu().numpy())
            val_dict["save_mpi"] = mpi[0]

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args

        with torch.no_grad():
            flows = self.flow_estim(refims[:, 1], refims[:, 0])
        batchsz, framenum, _, heiori, widori = refims.shape
        refims_last = refims[:, 0]
        refims = refims[:, 1]
        tarims = tarims[:, 1]
        disp_gts = disp_gts[:, 1]
        certainty_maps = certainty_maps[:, 1]
        with torch.no_grad():
            certainty_norm = certainty_maps.sum(dim=[-1, -2])

        layernum = self.model.num_layers
        netout, mpf = self.model(torch.cat([refims, refims_last], dim=1))
        mpf = mpf.reshape(batchsz, layernum, 2, heiori, widori)
        # compute mpi from netout
        # scheduling the s_bg
        step = kwargs["step"] if "step" in kwargs else 0
        bg_pct = self.scheduler.get_value(step)
        mpi, blend_weight = netout2mpi(netout, refims, bg_pct=bg_pct, ret_blendw=True)

        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
        disp_hat = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)

        # with torch.no_grad():  # compute scale
        disp_scale = disp_hat / (torch.abs(disp_gts) + 0.01)
        scale = torch.exp((torch.log(disp_scale) * certainty_maps).sum(dim=[-1, -2]) / certainty_norm)

        depth = depth * scale.reshape(-1, 1) * isleft
        # render target view
        tarview = shift_newview(mpi, torch.reciprocal(depth), ret_mask=False)
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        loss_dict["s_bg"] = torch.tensor(bg_pct).type_as(scale.detach())
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = photometric_loss(tarview, tarims, mode=self.pixel_loss_mode)
            l1_loss = l1_loss.mean()
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = l1_loss.detach()

        if "smooth_loss" in self.loss_weight.keys():
            smth_loss = smooth_grad(disp_hat, refims)
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"])
            loss_dict["smth"] = smth_loss.detach()

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(disp_scale / scale.reshape(-1, 1, 1))
            diff = (torch.pow(diff, 2) * certainty_maps).mean()
            final_loss += (diff * self.loss_weight["depth_loss"])
            loss_dict["depth"] = diff.detach()

        if "flow_epe" in self.loss_weight.keys():
            flow_render = overcompose(mpi, blend_weight, ret_mask=False, blend_content=mpf)
            epe = torch.norm(flow_render - flows, dim=1).mean()
            final_loss += epe * self.loss_weight["flow_epe"]
            loss_dict["flow_epe"] = epe.detach()

        if "flow_smth" in self.loss_weight.keys():
            mpf_gradx, mpf_grady = gradient(mpf.reshape(layernum * batchsz, 2, heiori, widori), order=2)
            mpf_grad = (mpf_gradx.abs() + mpf_grady.abs()).sum(dim=-3).reshape(batchsz, layernum, heiori, widori)
            weights = (- blend_weight + 1) * mpi[:, :, -1]
            flow_smth = (mpf_grad * weights).mean()
            final_loss += flow_smth * self.loss_weight["flow_smth"]
            loss_dict["flow_smth"] = flow_smth.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class ModelandMPFLoss(nn.Module):
    """
    Model that only estimate multi-plane flow
    optional cfg:
        flow_smth_ord: order of smooth term
        flow_smth_bw: whether to mask out those with small alpha value
    """
    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        assert "loss_weights" in cfg.keys(), "ModelAndLossBase: didn't find 'loss_weights' in cfg"
        self.loss_weight = cfg["loss_weights"].copy()

        torch.set_default_tensor_type(torch.FloatTensor)
        self.mpimodel = model["MPI"]
        self.mpimodel.train()
        self.mpimodel.cuda()

        self.mpfmodel = model["MPF"]
        self.mpfmodel.train()
        self.mpfmodel.cuda()

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])
        self.flow_loss_ord = self.loss_weight.pop("flow_smth_ord", 2)
        self.flow_loss_consider_weight = self.loss_weight.pop("flow_smth_bw", False)
        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["kitti"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)
        self.flow_estim.eval()

        self.last_img = None

    def infer_forward(self, img: torch.Tensor, restart=False):
        if restart:
            self.last_img = None

        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.cuda()
        batchsz, _, heiori, widori = img.shape
        layernum = self.mpimodel.num_layers
        with torch.no_grad():
            if self.last_img is None:
                if isinstance(self.mpimodel, MPI_FlowGrad):
                    mpi = torch.ones(batchsz, layernum, 4, heiori, widori).type_as(img)
                elif isinstance(self.mpimodel, MPINetv2):
                    mpi = netout2mpi(self.mpimodel(img), img, ret_blendw=True)
                self.last_img = img
                return mpi, torch.ones(batchsz, layernum, 2, heiori, widori).type_as(img)
            else:
                flowgt = self.flow_estim(img, self.last_img)
                if isinstance(self.mpimodel, MPI_FlowGrad):
                    flow_gradx, flow_grady = gradient(flowgt)
                    flow_gradx = flow_gradx.abs().sum(dim=1, keepdim=True)
                    flow_grady = flow_grady.abs().sum(dim=1, keepdim=True)
                    netout = self.mpimodel(torch.cat([img, flow_gradx, flow_grady], dim=1))
                elif isinstance(self.mpimodel, MPINetv2):
                    netout = self.mpimodel(img)
                mpi = netout2mpi(netout, img)
                mpf = self.mpfmodel(torch.cat([mpi[:, :, -1], flowgt], dim=1))
                mpf = mpf.reshape(batchsz, -1, 2, heiori, widori)
                return mpi, mpf

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        """
        basically same as train_forward
        only evaluate L1 loss
        also output displacement heat map & newview image
        """
        args = [_t.cuda() for _t in args]
        refim, tarim, disp_gt, certainty_map, isleft = args
        refim, refim_last = refim[:, 1], refim[:, 0]
        with torch.no_grad():
            flowgt = self.flow_estim(refim, refim_last)
        batchsz, _, heiori, widori = refim.shape
        val_dict = {}
        with torch.no_grad():
            if isinstance(self.mpimodel, MPI_FlowGrad):
                flow_gradx, flow_grady = gradient(flowgt)
                flow_gradx = flow_gradx.abs().sum(dim=1, keepdim=True)
                flow_grady = flow_grady.abs().sum(dim=1, keepdim=True)
                netout = self.mpimodel(torch.cat([refim, flow_gradx, flow_grady], dim=1))
            elif isinstance(self.mpimodel, MPINetv2):
                netout = self.mpimodel(refim)
            mpi, blend_weight = netout2mpi(netout, refim, ret_blendw=True)
            mpf = self.mpfmodel(torch.cat([mpi[:, :, -1], flowgt], dim=1))
            mpf = mpf.reshape(batchsz, -1, 2, heiori, widori)

            flow_render = overcompose(mpi, blend_weight, ret_mask=False, blend_content=mpf)
            flow_epe = torch.norm(mpf - flowgt.unsqueeze(1), dim=2)
            flow_epe = flow_epe * mpi[:, :, -1] * blend_weight
            epe = flow_epe.mean()
            depth = make_depths(mpi.shape[1]).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            disp_hat = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)
            val_dict["val_epe"] = float(epe)

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            val_dict["vis_flow"] = flow_to_png_middlebury(flow_render[0].detach().cpu().numpy())
            val_dict["vis_mpf"] = flow_to_png_middlebury(make_grid(mpf[0]).detach().cpu().numpy())
            val_dict["vis_flowgt"] = flow_to_png_middlebury(flowgt[0].detach().cpu().numpy())
            val_dict["vis_disp"] = draw_dense_disp(disp_hat, 1)

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        refims_last = refims[:, 0]
        refims = refims[:, 1]

        batchsz, _, heiori, widori = refims.shape
        with torch.no_grad():
            flows = self.flow_estim(refims, refims_last)
            if isinstance(self.mpimodel, MPI_FlowGrad):
                flow_gradx, flow_grady = gradient(flows)
                flow_gradx = flow_gradx.abs().sum(dim=1, keepdim=True)
                flow_grady = flow_grady.abs().sum(dim=1, keepdim=True)
                mpi = self.mpimodel(torch.cat([refims, flow_gradx, flow_grady], dim=1))
            elif isinstance(self.mpimodel, MPINetv2):
                mpi = self.mpimodel(refims)
            mpi, blend_weight = netout2mpi(mpi, refims, ret_blendw=True)

        mpf = self.mpfmodel(torch.cat([mpi[:, :, -1], flows], dim=1))
        mpf = mpf.reshape(batchsz, -1, 2, heiori, widori)
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        if "flow_epe" in self.loss_weight.keys():
            flow_epe = torch.norm(mpf - flows.unsqueeze(1), dim=2)
            flow_epe = flow_epe * mpi[:, :, -1] * blend_weight
            epe = flow_epe.mean()
            final_loss += epe * self.loss_weight["flow_epe"]
            loss_dict["flow_epe"] = epe.detach()

        if "flow_smth" in self.loss_weight.keys():
            mpf_gradx, mpf_grady = gradient(mpf.reshape(-1, 2, heiori, widori), order=self.flow_loss_ord)
            mpf_grad = (mpf_gradx.abs() + mpf_grady.abs()).sum(dim=-3).reshape(batchsz, -1, heiori, widori)
            if self.flow_loss_consider_weight:
                weights = (- blend_weight + 1) * mpi[:, :, -1]
                flow_smth = (mpf_grad * weights).mean()
            else:
                flow_smth = mpf_grad.mean()
            final_loss += flow_smth * self.loss_weight["flow_smth"]
            loss_dict["flow_smth"] = flow_smth.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class ModelandReccuNetLoss(nn.Module):
    """
    Recurrent Model that achieve temporal consistency (on netout)
    """
    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        assert "loss_weights" in cfg.keys(), "ModelAndLossBase: didn't find 'loss_weights' in cfg"
        self.loss_weight = cfg["loss_weights"].copy()

        torch.set_default_tensor_type(torch.FloatTensor)
        self.mpimodel = model
        self.mpimodel.train()
        self.mpimodel.cuda()

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.eval()
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)

        self.last_img = None
        self.last_netout = None

    def infer_forward(self, img: torch.Tensor, restart=False):
        if restart:
            self.last_img = None
            self.last_netout = None

        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.cuda()

        with torch.no_grad():
            if self.last_img is None:
                flow = None
                last_netout_warp = None
            else:
                flow = self.flow_estim(img, self.last_img)
                last_netout_warp = warp_flow(self.last_netout, flow)

            if isinstance(self.mpimodel, MPIRecuFlowNet):
                netout = self.mpimodel(img, flow, last_netout_warp)
            elif isinstance(self.mpimodel, MPIReccuNet):
                netout = self.mpimodel(img, last_netout_warp)
            else:
                raise NotImplementedError

            self.last_netout = netout
            self.last_img = img
            return netout2mpi(netout, img)

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        args = [_t.cuda() for _t in args]
        refim, tarim, disp_gt, certainty_map, isleft = args
        # with torch.no_grad():
        #     flow = self.flow_estim(refim[:, 1], refim[:, 0])
        refim = refim[:, 1]
        tarim = tarim[:, 1]
        disp_gt = disp_gt[:, 1]
        certainty_map = certainty_map[:, 1]
        batchsz, _, heiori, widori = refim.shape
        val_dict = {}
        with torch.no_grad():
            self.mpimodel.eval()
            netout = self.mpimodel(refim)
            self.mpimodel.train()

            # compute mpi from netout
            layernum = self.mpimodel.num_layers
            mpi, blend_weight = netout2mpi(netout, refim, ret_blendw=True)

            # estimate depth map and sample sparse point depth
            depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            disp_hat = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)

            # compute scale
            certainty_norm = certainty_map.sum(dim=[-1, -2])
            disp_scale = disp_hat / (torch.abs(disp_gt) + 0.0001)
            scale = torch.exp((torch.log(disp_scale) * certainty_map).sum(dim=[-1, -2]) / certainty_norm)

            depth = depth * scale.reshape(-1, 1) * isleft
            depthloss = torch.log(disp_scale / scale.reshape(-1, 1, 1))
            depthloss = (torch.pow(depthloss, 2) * certainty_map).mean()
            # render target view
            tarview, tarmask = shift_newview(mpi, torch.reciprocal(depth), ret_mask=True)

            l1_map = photo_l1loss(tarview, tarim)
            l1_loss = (l1_map * tarmask).sum() / tarmask.sum()
            val_dict["val_l1diff"] = float(l1_loss)
            val_dict["val_scale"] = float(scale.mean())
            val_dict["val_depth"] = float(depthloss)

        if "visualize" in kwargs.keys() and kwargs["visualize"]:
            view0 = (tarview[0] * 255).permute(1, 2, 0).detach().cpu().type(torch.uint8).numpy()
            # diff = (l1_map[0] * 255).detach().cpu().type(torch.uint8).numpy()
            val_dict["vis_newv"] = view0
            # val_dict["vis_diff"] = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # sparsedepth = draw_sparse_depth(refim, pt2ds, 1 / ptzs_gt, depth[0, -1])
            val_dict["vis_dispgt"] = draw_dense_disp(disp_gt, 1. / disp_gt.max())
            val_dict["vis_disp"] = draw_dense_disp(disp_hat, 1)

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        layernum = self.mpimodel.num_layers
        batchsz, framenum, _, heiori, widori = refims.shape
        bfnum = batchsz * (framenum - 1)
        # All are [B x Frame [x cnl] x H x W]
        with torch.no_grad():
            flows = self.flow_estim(refims[:, 1:].reshape(bfnum, 3, heiori, widori),
                                    refims[:, :-1].reshape(bfnum, 3, heiori, widori))
            certainty_norm = certainty_maps.sum(dim=[-1, -2])
            flows = flows.reshape(batchsz, framenum-1, 2, heiori, widori)
        netout = self.mpimodel(refims[:, 0])
        # scheduling the s_bg
        # step = kwargs["step"] if "step" in kwargs else 0
        # bg_pct = self.scheduler.get_value(step)
        mpi, blend_weight = netout2mpi(netout,
                                       refims[:, 0],
                                       ret_blendw=True)
        mpifinals = [mpi]
        netoutfinals = [netout]
        netoutlast = []
        for frameidx in range(1, framenum):
            netout_warp = warp_flow(netout, flows[:, frameidx - 1])
            netout = self.mpimodel(refims[:, frameidx], netout_warp)
            mpifinals.append(
                netout2mpi(netout,
                           refims[:, frameidx])
            )
            netoutfinals.append(netout)
            netoutlast.append(netout_warp)

        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)

        disps_hats = [
            estimate_disparity_torch(mpi_, depth)
            for mpi_ in mpifinals
        ]

        # with torch.no_grad():  # compute scale
        disp_diffs = [
            disps_hats[i] / (torch.abs(disp_gts[:, i]) + 0.0001)
            for i in range(framenum)
        ]
        # currently use first frame to compute scale
        scale = torch.exp((torch.log(disp_diffs[0]) * certainty_maps[:, 0]).sum(dim=[-1, -2]) / certainty_norm[:, 0])

        depth = depth * scale.reshape(-1, 1) * isleft
        # render target view
        tarviews = [
            shift_newview(mpi_, torch.reciprocal(depth), ret_mask=False)
            for mpi_ in mpifinals
                    ]
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        for mpiframeidx in range(len(mpifinals)):
            if "pixel_loss" in self.loss_weight.keys():
                l1_loss = photometric_loss(
                    tarviews[mpiframeidx],
                    tarims[:, mpiframeidx],
                    mode=self.pixel_loss_mode)
                l1_loss = l1_loss.mean()
                final_loss += (l1_loss * self.loss_weight["pixel_loss"])
                loss_dict[f"pixel{mpiframeidx}"] = l1_loss.detach()

            if "smooth_loss" in self.loss_weight.keys():
                smth_loss = smooth_grad(
                    disps_hats[mpiframeidx],
                    refims[:, mpiframeidx])
                smth_loss = smth_loss.mean()
                final_loss += (smth_loss * self.loss_weight["smooth_loss"])
                loss_dict[f"smth{mpiframeidx}"] = smth_loss.detach()

            if "depth_loss" in self.loss_weight.keys():
                diff = torch.log(disp_diffs[mpiframeidx] / scale.reshape(-1, 1, 1))
                diff = (torch.pow(diff, 2) * certainty_maps).mean()
                final_loss += (diff * self.loss_weight["depth_loss"])
                loss_dict[f"depth{mpiframeidx}"] = diff.detach()

        if "templ1_loss" in self.loss_weight.keys():
            temporal = torch.tensor(0.).type_as(mpi)
            for mpiframeidx in range(len(netoutlast)):
                temporal += torch.abs(netoutfinals[mpiframeidx+1] - netoutlast[mpiframeidx]).mean()
            temporal /= len(netoutlast)
            final_loss += (temporal * self.loss_weight["templ1_loss"])
            loss_dict["templ1"] = temporal.detach()

        # if "tempdepth_loss" in self.loss_weight.keys():
        #     temporal = torch.tensor(0.).type_as(mpi)
        #     for mpiframeidx in range(len(netoutlast)):
        #         temporal += torch.abs(disps_hats[mpiframeidx + 1] - disps_hats_last[mpiframeidx]).mean()
        #     temporal /= len(netoutlast)
        #     final_loss += (temporal * self.loss_weight["tempdepth_loss"])
        #     loss_dict["tempdepth"] = temporal.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class ModelandFlowGradLoss(nn.Module):
    """
    takes model that with two frames input, output mpf directly
    Model that estimate multi-plane flow
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
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["kitti"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)
        self.flow_estim.eval()

        self.last_img = None
        self.last_netout = None
        self.last_flow = None

    def infer_forward(self, img: torch.Tensor, restart=False):
        if restart:
            self.last_img = None
            self.last_netout = None

        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.cuda()
        batchsz, _, heiori, widori = img.shape
        layernum = self.model.num_layers
        with torch.no_grad():
            if self.last_img is None:
                self.last_img = img
                return torch.ones(batchsz, layernum, 4, heiori, widori).type_as(img)
            else:
                flows = self.flow_estim(img, self.last_img)
                flow_gradx, flow_grady = gradient(flows)
                flow_gradx = flow_gradx.abs().sum(dim=1, keepdim=True)
                flow_grady = flow_grady.abs().sum(dim=1, keepdim=True)
                netout = self.model(torch.cat([img, flow_gradx, flow_grady], dim=1))
                mpi, blend_weight = netout2mpi(netout, img, ret_blendw=True)
                self.last_img = img
                return mpi

    def infer_forward_reverse(self, img: torch.Tensor, restart=False):
        """
        the image is feed in reverse manner, i.e. the flow is t-1 ==> t,
        so the returned mpi is actually the mpi of last frame
        """
        if restart:
            self.last_img = None
            self.last_netout = None

        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.cuda()
        batchsz, _, heiori, widori = img.shape
        layernum = self.model.num_layers
        with torch.no_grad():
            if self.last_img is None:
                self.last_img = img
                return torch.ones(batchsz, layernum, 4, heiori, widori).type_as(img)
            else:
                if self.last_flow is None:
                    flows = self.flow_estim(self.last_img, img)
                else:
                    self.last_flow = downflow8(self.last_flow)
                    flow_ini = forward_scatter(self.last_flow, self.last_flow)
                    flows = self.flow_estim(self.last_img, img, flow_ini)
                flow_gradx, flow_grady = gradient(flows)
                flow_gradx = flow_gradx.abs().sum(dim=1, keepdim=True)
                flow_grady = flow_grady.abs().sum(dim=1, keepdim=True)
                netout = self.model(torch.cat([self.last_img, flow_gradx, flow_grady], dim=1))
                mpi, blend_weight = netout2mpi(netout, img, ret_blendw=True)
                self.last_img = img
                self.last_flow = flows
                return mpi

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        """
        basically same as train_forward
        only evaluate L1 loss
        also output displacement heat map & newview image
        """
        val_dict = {}
        with torch.no_grad():
            args = [_t.cuda() for _t in args]
            refim, tarim, disp_gt, certainty_map, isleft = args
            flows = self.flow_estim(refim[:, 1], refim[:, 0])
            flow_gradx, flow_grady = gradient(flows)
            flow_gradx = flow_gradx.abs().sum(dim=1, keepdim=True)
            flow_grady = flow_grady.abs().sum(dim=1, keepdim=True)
            batchsz, framenum, _, heiori, widori = refim.shape
            refim = refim[:, 1]
            tarim = tarim[:, 1]
            disp_gt = disp_gt[:, 1]
            certainty_map = certainty_map[:, 1]
            self.model.eval()
            netout = self.model(torch.cat([refim, flow_gradx, flow_grady], dim=1))
            self.model.train()

            # compute mpi from netout
            mpi, blend_weight = netout2mpi(netout, refim, ret_blendw=True)

            # estimate depth map and sample sparse point depth
            layernum = self.model.num_layers
            depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
            disp_hat = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)

            # compute scale
            certainty_norm = certainty_map.sum(dim=[-1, -2])
            disp_scale = disp_hat / (torch.abs(disp_gt) + 0.01)
            scale = torch.exp((torch.log(disp_scale) * certainty_map).sum(dim=[-1, -2]) / certainty_norm)

            depth = depth * scale.reshape(-1, 1) * isleft
            # render target view
            tarview, tarmask = shift_newview(mpi, torch.reciprocal(depth), ret_mask=True)

            l1_map = photo_l1loss(tarview, tarim)
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
        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args
        with torch.no_grad():
            flows = self.flow_estim(refims[:, 1], refims[:, 0])
            flow_gradx, flow_grady = gradient(flows)
            flow_gradx = flow_gradx.abs().sum(dim=1, keepdim=True)
            flow_grady = flow_grady.abs().sum(dim=1, keepdim=True)
        batchsz, framenum, _, heiori, widori = refims.shape
        refims = refims[:, 1]
        tarims = tarims[:, 1]
        disp_gts = disp_gts[:, 1]
        certainty_maps = certainty_maps[:, 1]
        with torch.no_grad():
            certainty_norm = certainty_maps.sum(dim=[-1, -2])

        layernum = self.model.num_layers
        netout = self.model(torch.cat([refims, flow_gradx, flow_grady], dim=1))
        # scheduling the s_bg
        step = kwargs["step"] if "step" in kwargs else 0
        bg_pct = self.scheduler.get_value(step)
        mpi, blend_weight = netout2mpi(netout, refims, bg_pct=bg_pct, ret_blendw=True)

        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
        disp_hat = estimate_disparity_torch(mpi, depth, blendweight=blend_weight)

        # with torch.no_grad():  # compute scale
        disp_scale = disp_hat / (torch.abs(disp_gts) + 0.01)
        scale = torch.exp((torch.log(disp_scale) * certainty_maps).sum(dim=[-1, -2]) / certainty_norm)

        depth = depth * scale.reshape(-1, 1) * isleft
        # render target view
        tarview = shift_newview(mpi, torch.reciprocal(depth), ret_mask=False)
        # compute final loss
        final_loss, loss_dict = torch.tensor(0., dtype=torch.float32).cuda(), {}
        loss_dict["scale"] = scale.detach().mean()
        loss_dict["s_bg"] = torch.tensor(bg_pct).type_as(scale.detach())
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = photometric_loss(tarview, tarims, mode=self.pixel_loss_mode)
            l1_loss = l1_loss.mean()
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = l1_loss.detach()

        if "smooth_loss" in self.loss_weight.keys():
            smth_loss = smooth_grad(disp_hat, refims)
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"])
            loss_dict["smth"] = smth_loss.detach()

        if "smooth_flowgrad_loss" in self.loss_weight.keys():
            disp_dx, disp_dy = gradient(disp_hat.unsqueeze(-3))
            smoothx = torch.max(disp_dx - 0.02, torch.tensor(0.).type_as(refims))
            smoothy = torch.max(disp_dy - 0.02, torch.tensor(0.).type_as(refims))
            with torch.no_grad():
                weightsx = - torch.min(flow_gradx / 2, torch.tensor(1.).type_as(refims)) + 1
                weightsy = - torch.min(flow_grady / 2, torch.tensor(1.).type_as(refims)) + 1
            smth_loss = smoothx * weightsx + smoothy * weightsy
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_flowgrad_loss"])
            loss_dict["flowgrad"] = smth_loss.detach()

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(disp_scale / scale.reshape(-1, 1, 1))
            diff = (torch.pow(diff, 2) * certainty_maps).mean()
            final_loss += (diff * self.loss_weight["depth_loss"])
            loss_dict["depth"] = diff.detach()

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

        assert(isinstance(models, nn.ModuleDict))
        models.train()
        models.cuda()
        self.mpimodel = models["MPI"]
        self.mpfmodel = models["MPF"]

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.learnmpf = self.loss_weight.pop("learnmpf", True)
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.eval()
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)
        self.flow_loss_ord = self.loss_weight.pop("flow_smth_ord", 2)
        self.flow_loss_consider_weight = self.loss_weight.pop("flow_smth_bw", False)

        # used for inference
        self.last_img = None
        self.last_mpi_warp = None

    def infer_forward(self, img: torch.Tensor, restart=False, mpiwarp_none=False, mpffromspf=False):
        """
        Will actually deferred one frame, i.e. input f1 -> f2 -> ...   output 0 -> o1 -> o2 -> ...
        """
        if restart:
            self.last_img = None
            self.last_mpi_warp = None
        if mpiwarp_none:
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

                mpi, blend_weight = self.forwardmpi(
                    self.last_img,
                    flow_gradx,
                    flow_grady,
                    mpi_warp=self.last_mpi_warp
                )
                if mpffromspf:
                    mpf = flows.reshape(batchsz, 1, 2, hei, wid).expand(-1, layernum, -1, -1, -1)
                else:
                    mpf = self.mpfmodel(torch.cat([mpi[:, :, -1], flows], dim=1))
                    mpf = mpf.reshape(batchsz, -1, 2, hei, wid)
                self.last_mpi_warp = forward_scatter_mpi(
                    flow01=mpf, mpi=mpi
                )
                self.last_img = img
                return mpi, mpf

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
            mpf = self.mpfmodel(torch.cat([mpi[:, :, -1], flows[:, 0]], dim=1))
            mpf = mpf.reshape(batchsz, -1, 2, heiori, widori)
            mpi_warp = forward_scatter_mpi(
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

        return val_dict

    def forwardmpi(self, img, flowgx, flowgy, mpi_warp=None):
        # todo: implement this
        # scheduling the s_bg
        netout = self.mpimodel(torch.cat([img, flowgx, flowgy], dim=1), mpi_warp)
        if mpi_warp is None:
            mpi, blend_weight = netout2mpi(netout,  # [:, self.mpimodel.num_layers - 1 + 3],
                                           img,
                                           ret_blendw=True)
        else:
            mpi, blend_weight = netoutupdatempi_maskfree(netout=netout,
                                                         img=img,
                                                         mpi_last=mpi_warp,
                                                         ret_blendw=True)
        return mpi, blend_weight

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
            flow_gradx = flow_gradx.abs().sum(dim=1).reshape(batchsz, framenum-1, 1, heiori, widori)
            flow_grady = flow_grady.abs().sum(dim=1).reshape(batchsz, framenum-1, 1, heiori, widori)
            flows = flows.reshape(batchsz, framenum-1, 2, heiori, widori)
        mpi_warp = None
        mpi_out = []
        mpf_out = []
        intermediates = {
            "blend_weight": [],
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

            mpf = self.mpfmodel(torch.cat([mpi[:, :, -1], flows[:, frameidx]], dim=1))
            mpf = mpf.reshape(batchsz, -1, 2, heiori, widori)
            mpf_out.append(mpf)

            mpi_warp = forward_scatter_mpi(
                flow01=mpf, mpi=mpi
            )
            intermediates["mpi_warp"].append(mpi_warp)
        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(refims).unsqueeze(0).repeat(batchsz, 1)

        disps_hats = [
            estimate_disparity_torch(mpi_, depth)
            for mpi_ in mpi_out
        ]
        # with torch.no_grad():  # compute scale
        disp_diffs = [
            disps_hats[i] / (torch.abs(disp_gts[:, i]) + 0.0001)
            for i in range(framenum-1)
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
                l1_loss = photometric_loss(
                    tarviews[mpiframeidx],
                    tarims[:, mpiframeidx],
                    mode=self.pixel_loss_mode)
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
            blend_weight = intermediates["blend_weight"][mpfframeidx]
            mpf = mpf_out[mpfframeidx]
            if "flow_epe" in self.loss_weight.keys():
                flow_epe = torch.norm(mpf - flows[:, mpfframeidx].unsqueeze(1), dim=2)
                flow_epe = flow_epe * mpi[:, :, -1] * blend_weight
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
                    weights = (- blend_weight + 1) * mpi[:, :, -1]
                    flow_smth = (mpf_grad * weights).mean()
                else:
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
