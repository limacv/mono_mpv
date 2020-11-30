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
from .mpi_network import MPINet
from .mpv_network import MPVNet
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
        self.layernum = self.model.num_layers

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


class ModelandDispFlowLoss(nn.Module):
    """
    SV stand for single view
    """
    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        assert "loss_weights" in cfg.keys(), "ModelAndLossBase: didn't find 'loss_weights' in cfg"
        self.loss_weight = cfg["loss_weights"].copy()

        torch.set_default_tensor_type(torch.FloatTensor)
        assert(isinstance(model, MPI_SPF_Net) or isinstance(model, MPI_MPF_Net))
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

    # def infer_forward(self, im: torch.Tensor, mode='pad_constant'):
    #     with torch.no_grad():
    #         if im.dim() == 3:
    #             im = im.unsqueeze(0)
    #
    #         im = im.cuda()
    #         batchsz, cnl, hei, wid = im.shape
    #         if mode == 'resize':
    #             hei_new, wid_new = 512, 512 + 128 * 3
    #             img_padded = Resize([hei_new, wid_new])(im)
    #         elif 'pad' in mode:
    #             hei_new, wid_new = ((hei - 1) // 128 + 1) * 128, ((wid - 1) // 128 + 1) * 128
    #             padding = [
    #                 (wid_new - wid) // 2,
    #                 (wid_new - wid + 1) // 2,
    #                 (hei_new - hei) // 2,
    #                 (hei_new - hei + 1) // 2
    #             ]
    #             mode = mode[4:] if "pad_" in mode else "constant"
    #             img_padded = torchf.pad(im, padding, mode)
    #         else:
    #             raise NotImplementedError
    #
    #         self.model.eval()
    #         netout = self.model(img_padded)
    #
    #         # depad
    #         if mode == 'resize':
    #             netout = Resize([hei_new, wid_new])(netout)
    #         else:
    #             netout = netout[..., padding[2]: hei_new - padding[3], padding[0]: wid_new - padding[1]]
    #         mpi = netout2mpi(netout, im)
    #         self.model.train()
    #
    #     return mpi

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        """
        basically same as train_forward
        only evaluate L1 loss
        also output displacement heat map & newview image
        """
        args = [_t.cuda() for _t in args]
        refim, tarim, disp_gt, certainty_map, isleft = args
        with torch.no_grad():
            flow = self.flow_estim(refim[:, 1], refim[:, 0])
        refim = refim[:, 1]
        tarim = tarim[:, 1]
        disp_gt = disp_gt[:, 1]
        certainty_map = certainty_map[:, 1]
        batchsz, _, heiori, widori = refim.shape
        val_dict = {}
        with torch.no_grad():
            self.model.eval()
            netout = self.model(refim, flow)
            self.model.train()

            if isinstance(netout, Tuple):
                netout = netout[0]
            # compute mpi from netout
            layernum = self.model.num_layers
            mpichannel = layernum - 1 + 3
            netout, mpf = torch.split(netout, [mpichannel, netout.shape[1] - mpichannel], dim=1)
            mpi, blend_weight = netout2mpi(netout, refim, ret_blendw=True)

            if isinstance(self.model, MPI_SPF_Net):
                # flows: [B, 1, 2, H, W]: [B, 2, H, W]
                mpf = blend_weight.unsqueeze(2) * flow.unsqueeze(1) + \
                      (-blend_weight.unsqueeze(2) + 1.) * mpf.unsqueeze(1)
            elif isinstance(self.model, MPI_MPF_Net):
                mpf = mpf.reshape(batchsz, layernum, 2, heiori, widori)
            flow_render = overcompose(mpi, blend_weight, ret_mask=False, blend_content=mpf)
            epe = torch.norm(flow_render - flow, dim=1, p=1).mean()

            # estimate depth map and sample sparse point depth
            depth = make_depths(32).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)
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
            val_dict["vis_flowgt"] = flow_to_png_middlebury(flow[0].detach().cpu().numpy())
            val_dict["save_mpi"] = mpi[0]

        return val_dict

    def forward(self, *args: torch.Tensor, **kwargs):
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarims, disp_gts, certainty_maps, isleft = args

        with torch.no_grad():
            flows = self.flow_estim(refims[:, 1], refims[:, 0])
        batchsz, framenum, _, heiori, widori = refims.shape
        refims = refims[:, 1]
        tarims = tarims[:, 1]
        disp_gts = disp_gts[:, 1]
        certainty_maps = certainty_maps[:, 1]

        with torch.no_grad():
            certainty_norm = certainty_maps.sum(dim=[-1, -2])
        # TODO: currently just use second images
        # TODO: flow Plane+parallax decomposition
        layernum = self.model.num_layers
        mpichannel = layernum - 1 + 3
        netout = self.model(refims, flows)
        netout, mpf = torch.split(netout, [mpichannel, netout.shape[1] - mpichannel], dim=1)
        # compute mpi from netout
        # scheduling the s_bg
        step = kwargs["step"] if "step" in kwargs else 0
        bg_pct = self.scheduler.get_value(step)
        mpi, blend_weight = netout2mpi(netout, refims, bg_pct=bg_pct, ret_blendw=True)

        if isinstance(self.model, MPI_SPF_Net):
            # flows: [B, 1, 2, H, W]: [B, 2, H, W]
            mpf = blend_weight.unsqueeze(2) * flows.unsqueeze(1) + \
                  (-blend_weight.unsqueeze(2) + 1.) * mpf.unsqueeze(1)
        elif isinstance(self.model, MPI_MPF_Net):
            mpf = mpf.reshape(batchsz, layernum, 2, heiori, widori)
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

        if "sparse_loss" in self.loss_weight.keys():
            alphas = mpi[:, :, -1, :, :]
            alphas_norm = alphas / torch.norm(alphas, dim=1, keepdim=True)
            sparse_loss = torch.sum(torch.abs(alphas_norm), dim=1).mean()
            final_loss += sparse_loss * self.loss_weight["sparse_loss"]
            loss_dict["sparse_loss"] = sparse_loss.detach()

        if "flow_epe" in self.loss_weight.keys():
            flow_render = overcompose(mpi, blend_weight, ret_mask=False, blend_content=mpf)
            epe = torch.norm(flow_render - flows, dim=1, p=1).mean()
            final_loss += epe * self.loss_weight["flow_epe"]
            loss_dict["flow_epe"] = epe.detach()

        if "flow_smth" in self.loss_weight.keys():
            # TODO: constraint flow in homography space
            # currently just simple L1 smooth loss
            mpf_gradx, mpf_grady = gradient(mpf.reshape(layernum * batchsz, 2, heiori, widori))
            mpf_grad = (mpf_gradx.abs() + mpf_grady.abs()).sum(dim=-3).reshape(batchsz, layernum, heiori, widori)
            weights = (- blend_weight + 1) * mpi[:, :, -1]
            flow_smth = (mpf_grad * weights).mean()
            final_loss += flow_smth * self.loss_weight["flow_smth"]
            loss_dict["flow_smth"] = flow_smth.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}


class ModelandFullLoss(nn.Module):
    def __init__(self, models: nn.Module, cfg: dict):
        super().__init__()
        assert "loss_weights" in cfg.keys(), "ModelAndLossBase: didn't find 'loss_weights' in cfg"
        self.loss_weight = cfg["loss_weights"].copy()

        torch.set_default_tensor_type(torch.FloatTensor)
        assert(isinstance(models, nn.ModuleDict))
        models.train()
        models.cuda()
        self.mpimodel = models["MPI"]
        self.fusemodel = models["Fuser"]

        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.scheduler = ParamScheduler([1e4, 2e4], [0.5, 1])

        # optical flow estimator
        self.flow_estim = RAFTNet(False)
        self.flow_estim.eval()
        self.flow_estim.cuda()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flow_estim.load_state_dict(state_dict)

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        """
        basically same as train_forward
        only evaluate L1 loss
        also output displacement heat map & newview image
        """
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
        bfnumfull = batchsz * framenum
        bfnum = batchsz * (framenum - 1)
        # All are [B x Frame [x cnl] x H x W]
        with torch.no_grad():
            flows = self.flow_estim(refims[:, 1:].reshape(bfnum, 3, heiori, widori),
                                    refims[:, :-1].reshape(bfnum, 3, heiori, widori))
            certainty_norm = certainty_maps.sum(dim=[-1, -2])

        netout = self.mpimodel(refims.reshape(bfnumfull, 3, heiori, widori))
        # scheduling the s_bg
        # step = kwargs["step"] if "step" in kwargs else 0
        # bg_pct = self.scheduler.get_value(step)
        mpi, blend_weight = netout2mpi(netout,
                                       refims.reshape(bfnumfull, 3, heiori, widori),
                                       ret_blendw=True)

        mpf = flows.reshape(bfnum, 1, 2, heiori, widori).expand(-1, layernum, -1, -1, -1)

        mpi = mpi.reshape(batchsz, framenum, layernum, 4, heiori, widori)
        mpf = mpf.reshape(batchsz, framenum - 1, layernum, 2, heiori, widori)

        mpifinals = [mpi[:, 0]]
        mpilasts = []
        for frameidx in range(1, framenum):
            mpi_warpped = warp_flow(mpifinals[-1], mpf[:, frameidx - 1])
            mpifinal = self.fusemodel(mpi[:, frameidx], mpi_warpped)
            mpifinals.append(mpifinal)
            mpilasts.append(mpi_warpped)

        # estimate depth map and sample sparse point depth
        depth = make_depths(layernum).type_as(mpi).unsqueeze(0).repeat(batchsz, 1)

        disps_hats = [
            estimate_disparity_torch(mpi_, depth)
            for mpi_ in mpifinals
        ]
        disps_hats_last = [
            estimate_disparity_torch(mpi_, depth)
            for mpi_ in mpilasts
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
            for mpiframeidx in range(len(mpilasts)):
                temporal += torch.abs(mpifinals[mpiframeidx+1] - mpilasts[mpiframeidx]).mean()
            temporal /= len(mpilasts)
            final_loss += (temporal * self.loss_weight["templ1_loss"])
            loss_dict["templ1"] = temporal.detach()

        if "tempdepth_loss" in self.loss_weight.keys():
            temporal = torch.tensor(0.).type_as(mpi)
            for mpiframeidx in range(len(mpilasts)):
                temporal += torch.abs(disps_hats[mpiframeidx + 1] - disps_hats_last[mpiframeidx]).mean()
            temporal /= len(mpilasts)
            final_loss += (temporal * self.loss_weight["tempdepth_loss"])
            loss_dict["tempdepth"] = temporal.detach()

        final_loss = final_loss.unsqueeze(0)
        loss_dict = {k: v.unsqueeze(0) for k, v in loss_dict.items()}
        return {"loss": final_loss, "loss_dict": loss_dict}
