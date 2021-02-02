import os
from typing import Sequence
from collections import Counter
from utils import *
from models.mpi_utils import *
from models.flow_utils import *
from models.img_utils import *
from dataset import COMPUTER_NAME


SCALE_AND_SHIFT_INVARIANT = False


class MetricCounter:
    def __init__(self):
        self.metrics = {}
        self.metrics_num = {}

    def collect(self, name, val, count=1):
        if name in self.metrics.keys():
            self.metrics[name] += val
            self.metrics_num[name] += count
        else:
            self.metrics[name] = val
            self.metrics_num[name] += count

    def __add__(self, other: 'MetricCounter'):
        for k, v in other.metrics.items():
            self.collect(k, v, other.metrics_num[k])
        return self

    def make_table(self):
        s_ = "metric\t|\tvalue\t|\tcount\n" \
             + '\n'.join([f"{k}\t|\t{v / self.metrics_num[k]:.4f}\t|\t{self.metrics_num[k]}"
                          for k, v in self.metrics.items()])
        return s_


@torch.no_grad()
def evaluation(cfg):
    datasetname = cfg["dataset"]
    datasetcfg = cfg["datasetcfg"]
    dataset = select_evalset(datasetname, **datasetcfg)

    checkpointname = cfg.pop("checkpoint", "")
    modelname = cfg.pop("model", "auto")
    pipelinename = cfg.pop("pipeline", "auto")
    ret_cfg = cfg.pop("infer_cfg", {})
    pipeline = smart_select_pipeline(
        checkpoint=checkpointname,
        force_modelname=modelname,
        force_pipelinename=pipelinename,
    )
    checkpoint = torch.load(checkpointname)
    checkpointcfg = checkpoint["cfg"] if "cfg" in checkpoint.keys() else "no cfg"
    del checkpoint

    saveroot = cfg["saveroot"]
    save_persceneinfo = cfg["save_perscene"]
    save_tarviews = cfg["save_tarviews"]
    save_disparity = cfg["save_disparity"]
    if saveroot == "auto":
        saveroot = "/home/lmaag/xgpu-scratch/PI/psander/mali_data/Visual"
        if COMPUTER_NAME == "msi":
            saveroot = "Z:\\tmp\\Visual"
            if not os.path.exists(saveroot):
                saveroot = "D:\\MSI_NB\\source\\data\\Visual"
        saveroot = os.path.join(saveroot, f"{datasetname}_{modelname}_{pipelinename}")
    mkdir_ifnotexist(saveroot)

    header = f"""
+===================================
| Dataset: {dataset.name}
|       with config: {', '.join([f'{k}: {v}' for k, v in datasetcfg.items()])}
| Model: {modelname}
| Pipeline: {pipelinename}
| checkpoint: {checkpointname}
|       with config: {checkpointcfg}
| inference_cfg: {', '.join([f'{k}: {v}' for k, v in ret_cfg.items()])}
+===================================================\n
    """
    print(header, flush=True)

    # For estimating the temporal performance
    if hasattr(pipeline, "flow_estim"):
        flow_estim = pipeline.flow_estim
    else:
        flow_estim = RAFTNet(False)
        flow_estim.eval()
        flow_estim.cuda()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        flow_estim.load_state_dict(state_dict)
        for param in flow_estim.parameters():
            param.requires_grad = False

    EVAL_DICT = MetricCounter()
    for datas in dataset:
        scene_name = datas['scene_name']
        EVAL_DICT_PERSCENE = MetricCounter()
        newviews_save = []
        disparitys_save = []
        depths_raw = make_depths(32).cuda()
        dd = None
        for frameidx, refimg in enumerate(datas["in_imgs"]):
            mpi = pipeline.infer_forward(refimg.unsqueeze(0).cuda(), ret_cfg)

            # ===========================================
            # estimate the scale & shift and estimate disparity map
            # ===========================================
            if frameidx == 0:  # only estimate scale in first frame
                disp_raw = estimate_disparity_torch(mpi, depths_raw).squeeze(0)

                if "gt_depth" in datas.keys():  # semi-dense depth map
                    gt_depth = datas["gt_depth"][frameidx].cuda()
                    valid_mask = torch.logical_and(-1e3 < gt_depth, gt_depth < 1e3)
                    gt_depth[torch.logical_not(valid_mask)] = 1e3

                    scale = torch.exp((torch.log(disp_raw * gt_depth) * valid_mask.type_as(disp_raw)).sum(dim=[-1, -2]) \
                                      / valid_mask.sum(dim=[-1, -2]))
                    dd = depths_raw * scale

                elif "gt_disparity" in datas.keys():  # semi-dense disparity map
                    gt_disparity = datas["gt_disparity"][frameidx].cuda()
                    valid_mask = torch.logical_and(-1e3 < gt_disparity, gt_disparity < 1e3)
                    gt_disparity[torch.logical_not(valid_mask)] = 1e3
                    if SCALE_AND_SHIFT_INVARIANT:
                        shift_es = torch.median(disp_raw.reshape(-1)[valid_mask.reshape(-1)])
                        shift_gt = torch.median(gt_disparity.reshape(-1)[valid_mask.reshape(-1)])
                        scale_es = ((disp_raw - shift_es).abs() * valid_mask).sum(dim=[-1, -2]) \
                                   / valid_mask.sum(dim=[-1, -2])
                        scale_gt = ((gt_disparity - shift_gt).abs() * valid_mask).sum(dim=[-1, -2]) \
                                   / valid_mask.sum(dim=[-1, -2])
                        dd = (torch.reciprocal(depths_raw) - shift_es) / -scale_es * scale_gt + shift_gt

                    else:
                        # only estimate an naive shift of gt (max plus a small margin)
                        gt_disparity[torch.logical_not(valid_mask)] = -1e3
                        shift_gt = torch.max(gt_disparity) + 10
                        disp_diff = disp_raw / -(gt_disparity - shift_gt)
                        scale = torch.exp((torch.log(disp_diff) * valid_mask).sum(dim=[-1, -2])
                                          / valid_mask.sum(dim=[-1, -2]))
                        dd = torch.reciprocal(depths_raw * -scale) + shift_gt

                elif "gt_sparsedepth" in datas.keys():  # sparse depth point
                    raise NotImplementedError()
                else:
                    raise RuntimeError(f"Evaluator::Cannot find any depth ground truth")

            # ===========================================
            # render new view and quality
            # ===========================================
            if "in_poses" in datas.keys():  # arbitrary pose
                refpose = datas["in_poses"][frameidx]
                if not isinstance(datas["gt_poses"][frameidx], Sequence):
                    datas["gt_poses"][frameidx] = [datas["gt_poses"][frameidx]]
                    datas["gt_imgs"][frameidx] = [datas["gt_poses"][frameidx]]

                newviews_save.append([])
                for tarpose, tarviewgt in zip(datas["gt_poses"][frameidx], datas["gt_imgs"][frameidx]):
                    tarview = render_newview(mpi,
                                             refpose[0].unsqueeze(0).cuda(),
                                             tarpose[0].unsqueeze(0).cuda(),
                                             refpose[1].unsqueeze(0).cuda(),
                                             tarpose[1].unsqueeze(0).cuda(),
                                             depths_raw)

                    for metric_name in ["ssim", "psnr", "mse"]:
                        val = compute_img_metric(tarview, tarviewgt, metric_name, margin=0)
                        EVAL_DICT_PERSCENE.collect(metric_name, val)

                    newviews_save[-1].append(tarviewgt.cpu())
            elif "gt_disparity" in datas.keys():  # only shift left or right
                tarview = shift_newview(mpi, -dd)
                tarviewgt = datas["gt_imgs"][frameidx]
                for metric_name in ["ssim", "psnr", "mse"]:
                    val = compute_img_metric(tarview, tarviewgt, metric_name, margin=0)
                    EVAL_DICT_PERSCENE.collect(metric_name, val)

                newviews_save.append(tarviewgt.cpu())
            else:
                raise RuntimeError(f"Evaluator::Cannot find any gt views")

            # ===========================================
            # Depth quality
            # ===========================================
            disp_raw = estimate_disparity_torch(mpi, depths_raw).squeeze(0)
            disparitys_save.append(disp_raw)

            # ===========================================
            # Depth temporal consistency
            # ===========================================

        # end of per-scene processing
        # ===================================================================
        # Save informations
        # ===================================================================
        persceneinfo_path = os.path.join(saveroot, f"perscene_{scene_name}.txt")
        result_str = f"evaluation result of {scene_name}:\n" + EVAL_DICT_PERSCENE.make_table()
        print(result_str, flush=True)
        if save_persceneinfo:
            with open(persceneinfo_path, 'w') as f:
                f.writelines(result_str)

        saveroot_perscene = os.path.join(saveroot, f"{scene_name}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if save_tarviews and isinstance(newviews_save[0], np.ndarray):
            videoname = os.path.join(saveroot_perscene, f"novelviews.mp4")
            wido, heio = newviews_save[0].shape[:2]
            writer = cv2.VideoWriter()
            writer.open(videoname, fourcc, 15, (heio, wido), isColor=True)
            for img in newviews_save:
                writer.write(img)
            writer.release()
            print(f"{videoname} saved!")
        if save_disparity:
            videoname = os.path.join(saveroot_perscene, f"disparitymaps.mp4")
            wido, heio = disparitys_save[0].shape[:2]
            writer = cv2.VideoWriter()
            writer.open(videoname, fourcc, 15, (heio, wido), isColor=True)
            for img in disparitys_save:
                img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
                writer.write(img)
            writer.release()
            print(f"{videoname} saved!")

        # collect per-scene info
        EVAL_DICT = EVAL_DICT + EVAL_DICT_PERSCENE

    # end of the entire dataset processing
    print(f"Successfully processing all {len(dataset)} scenes/videos")
    info_path = os.path.join(saveroot, "all_results.txt")
    with open(info_path, 'w') as f:
        f.writelines(header)
        f.writelines(EVAL_DICT.make_table())
