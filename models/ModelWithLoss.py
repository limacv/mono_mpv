import torch
import torch.nn as nn
import torch.nn.functional as torchf
from typing import List, Callable
import numpy as np
import sys
from .loss_utils import *
from torchvision.transforms import ToTensor, Resize
from .mpi_utils import *
import cv2

'''
This class is a model container provide interface between
                                            1. dataset <-> model
                                            2. model <-> loss
'''


class ModelandLossBase:
    def __init__(self, model: nn.Module, loss_cfg: Dict):
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
        self.loss_weight = loss_cfg.copy()
        torch.set_default_tensor_type(torch.FloatTensor)
        self.model = model
        self.model.train()
        self.model.cuda()

    def train_forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Abstract call for training forward process
        step is used to control weight scheduling
        return the final loss to be backward() and loss dict contains all the loss term
        """
        raise NotImplementedError

    def valid_forward(self, *args: torch.Tensor, **kwargs):
        """
        basically same as train_forward
        only evaluate L1 loss
        also output displacement heat map & newview image
        """
        args = [_t.cuda() for _t in args]
        refim, tarim, refextrin, tarextrin, intrin, pt2ds, ptzs_gt = args
        # TODO: valid for seq input
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


class ModelandSVLoss(ModelandLossBase):
    """
    SV stand for single view
    """

    def __init__(self, model: nn.Module, loss_cfg: dict):
        super(ModelandSVLoss, self).__init__(model, loss_cfg)
        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.scheduler = ParamScheduler([1e5, 2e5], [0.5, 1])

    def train_forward(self, *args: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
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
        loss_dict["scale"] = float(scale.detach().mean())
        loss_dict["s_bg"] = bg_pct
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = photometric_loss(tarview, tarim, mode=self.pixel_loss_mode)
            l1_loss = (l1_loss * tarmask).sum() / tarmask.sum()
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = float(l1_loss.detach())

        if "smooth_loss" in self.loss_weight.keys():
            smth_loss = smooth_grad(disparity, refim)
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"])
            loss_dict["smth"] = float(smth_loss.detach())

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(ptdis_e * ptzs_gt / scale)
            diff = torch.pow(diff, 2).mean()
            final_loss += (diff * self.loss_weight["depth_loss"])
            loss_dict["depth"] = float(diff.detach())

        if "sparse_loss" in self.loss_weight.keys():
            alphas = mpi[:, :, -1, :, :]
            alphas_norm = alphas / torch.norm(alphas, dim=1, keepdim=True)
            sparse_loss = torch.sum(torch.abs(alphas_norm), dim=1).mean()
            final_loss += sparse_loss * self.loss_weight["sparse_loss"]
            loss_dict["sparse_loss"] = float(sparse_loss.detach())

        return final_loss, loss_dict


class ModelandTimeLoss(ModelandLossBase):
    def __init__(self, model: nn.Module, loss_cfg: dict):
        super().__init__(model, loss_cfg)
        self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "l1")
        self.scheduler = ParamScheduler([1e5, 2e5], [0.5, 1])
        # self.pixel_loss_mode = self.loss_weight.pop("pixel_loss_cfg", "ssim")

    def train_forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        # tocuda and unzip
        args = [_t.cuda() for _t in args]
        refims, tarim, refextrins, tarextrin, intrin, pt2ds, ptzs_gts = args
        # if refims.shape[0] != 0:
        #     raise NotImplementedError("now only batch size 1 is supported")
        batchsz, framenum, _, heiori, widori = refims.shape
        layernum = self.model.mpi_layers

        mpis, blend_weights = [], []
        netout, feat_last = self.model(refims[:, 0])
        mpi, blend_weight = netout2mpi(netout, refims[:, 0], ret_blendw=True)
        mpis.append(mpi)
        blend_weights.append(blend_weight)

        for frameidx in range(1, framenum):
            netout, feat_last = self.model(refims[:, frameidx], feat_last)
            mpi, blend_weight = netout2mpi(netout, refims[:, frameidx], ret_blendw=True)
            mpis.append(mpi)
            blend_weights.append(blend_weight)

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
        if "pixel_loss" in self.loss_weight.keys():
            l1_loss = photometric_loss(imref=tarim.reshape(bsz_fnum, 3, heiori, widori),
                                       imreconstruct=tarviews,
                                       mode=self.pixel_loss_mode)
            l1_loss = (l1_loss * tarmasks).sum() / tarmasks.sum()
            final_loss += (l1_loss * self.loss_weight["pixel_loss"])
            loss_dict["pixel"] = float(l1_loss.detach())

        if "pixel_std_loss" in self.loss_weight.keys():
            tarview_mean = tarviews.reshape(batchsz, framenum, 3, heiori, widori).mean(dim=1, keepdim=True)
            valid_mask = tarmasks.reshape(batchsz, framenum, heiori, widori).prod(dim=1, keepdim=True)
            intermediate_var["per_batch_mask"] = valid_mask
            std_loss = torch.abs(tarviews.reshape(batchsz, framenum, 3, heiori, widori) - tarview_mean).sum(dim=2)
            std_loss = (std_loss * valid_mask).sum() / valid_mask.sum() / framenum
            final_loss += (std_loss * self.loss_weight["pixel_std_loss"])
            loss_dict["pixel_std"] = float(std_loss.detach())

        if "smooth_loss" in self.loss_weight.keys():
            smth_loss = smooth_grad(disparitys, refims.reshape(bsz_fnum, 3, heiori, widori))
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_loss"])
            loss_dict["smth"] = float(smth_loss.detach())

        if "smooth_tar_loss" in self.loss_weight.keys():
            print("ModelWithLoss::warning! smooth_tar_loss not tested")
            smth_loss = smooth_grad(tardisps, tarim.reshape(bsz_fnum, 3, heiori, widori))
            smth_loss = smth_loss.mean()
            final_loss += (smth_loss * self.loss_weight["smooth_tar_loss"])
            loss_dict["smthtar"] = float(smth_loss.detach())

        if "temporal_loss" in self.loss_weight.keys():
            tardisp_mean = tardisps.reshape(batchsz, framenum, heiori, widori).mean(dim=1, keepdim=True)
            valid_mask = tarmasks.reshape(batchsz, framenum, heiori, widori).prod(dim=1, keepdim=True) \
                if "per_batch_mask" not in intermediate_var.keys() else intermediate_var["per_batch_mask"]
            temp_loss = torch.abs(tardisps.reshape(batchsz, framenum, heiori, widori) - tardisp_mean)
            temp_loss = (temp_loss * valid_mask).sum() / valid_mask.sum() / framenum
            final_loss += (temp_loss * self.loss_weight["temporal_loss"])
            loss_dict["temp"] = float(temp_loss.detach())

        if "depth_loss" in self.loss_weight.keys():
            diff = torch.log(ptdis_e * ptzs_gts / scale)
            diff = torch.pow(diff, 2).mean()  # mean among: batch & frames & points
            final_loss += (diff * self.loss_weight["depth_loss"])
            loss_dict["depth"] = float(diff.detach())

        if "sparse_loss" in self.loss_weight.keys():
            alphas = mpis[:, :, -1, :, :]  # shape [batchsz*framenum x layerNum x H x W]
            alphas_norm = alphas / torch.norm(alphas, dim=1, keepdim=True)
            sparse_loss = torch.sum(torch.abs(alphas_norm), dim=1).mean()
            final_loss += sparse_loss * self.loss_weight["sparse_loss"]
            loss_dict["sparse_loss"] = float(sparse_loss.detach())

        return final_loss, loss_dict
