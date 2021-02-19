import multiprocessing as mp
from utils import *
from models.mpi_utils import *
# from models.flow_utils import *
# from models.img_utils import *
from datetime import datetime
from dataset import COMPUTER_NAME

SCALE_AND_SHIFT_INVARIANT = False


class MetricCounter:
    def __init__(self):
        self.metrics = {}
        self.metrics_num = {}

    def collect(self, name, val, count=1.):
        if name in self.metrics.keys():
            self.metrics[name] += val
            self.metrics_num[name] += count
        else:
            self.metrics[name] = val
            self.metrics_num[name] = count

    def __add__(self, other: 'MetricCounter'):
        for k, v in other.metrics.items():
            self.collect(k, v, other.metrics_num[k])
        return self

    def make_table(self):
        s_ = f"\n{'metric': ^20}|{'value': ^10}|\tcount\n" \
             + '\n'.join([f"{k: ^20}|{v / self.metrics_num[k]: ^10.4f}|\t{self.metrics_num[k]}"
                          for k, v in self.metrics.items()])
        return s_


@torch.no_grad()
def evaluation_one(dataset, cudaid, pipeline, cfg):
    print(f"rank {cudaid} start working")
    num_process = cfg["num_process"]
    ret_cfg = cfg.pop("infer_cfg", "")
    eval_crop_margin = cfg.pop("eval_crop_margin", 0)

    saveroot = cfg["saveroot"]
    save_persceneinfo = cfg["save_perscene"]
    save_tarviews = cfg["save_tarviews"]
    save_disparity = cfg["save_disparity"]
    # For estimating the temporal performance
    torch.cuda.set_device(cudaid)

    pipeline = pipeline.cuda()
    if hasattr(pipeline, "flow_estim"):
        flow_estim = pipeline.flow_estim
    else:
        print(f"load RAFTNet")
        flow_estim = RAFTNet(False).cuda()
        flow_estim.eval()
        state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        flow_estim.load_state_dict(state_dict)
        for param in flow_estim.parameters():
            param.requires_grad = False

    SELF_EVAL_DICT = MetricCounter()
    idxes = np.arange(cudaid, len(dataset), num_process)
    for sceneid in idxes:
        datas = dataset[sceneid]
        scene_name = datas['scene_name']
        EVAL_DICT_PERSCENE = MetricCounter()
        newviews_save = []
        disparitys_save = []
        disparitys_out = []
        disparitys_gts = []
        depths_raw = make_depths(32)
        dd = None

        if hasattr(pipeline, "infer_multiple"):
            mpis = pipeline.infer_multiple(datas["in_imgs"], ret_cfg)
        else:
            mpis = []
            for refimg in datas["in_imgs"]:
                mpi = pipeline.infer_forward(refimg.unsqueeze(0).cuda(), ret_cfg)
                mpis.append(mpi.cpu())

        assert len(mpis) == len(datas["in_imgs"])
        for frameidx, mpi in enumerate(mpis):
            # ===========================================
            # estimate the scale & shift and estimate disparity map
            # ===========================================
            if frameidx == 0:  # only estimate scale in first frame
                disp_raw = estimate_disparity_torch(mpi, depths_raw).squeeze(0)

                if "gt_depth" in datas.keys():  # semi-dense depth map
                    gt_depth = datas["gt_depth"][frameidx]
                    valid_mask = torch.logical_and(-1e3 < gt_depth, gt_depth < 1e3)
                    gt_depth[torch.logical_not(valid_mask)] = 1e3

                    scale = torch.exp((torch.log(disp_raw * gt_depth) * valid_mask.type_as(disp_raw)).sum(dim=[-1, -2]) \
                                      / valid_mask.sum(dim=[-1, -2]))
                    dde = depths_raw * scale
                    dd = torch.reciprocal(dde)

                elif "gt_disparity" in datas.keys():  # semi-dense disparity map
                    gt_disparity = datas["gt_disparity"][frameidx]
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
                        if "stereoblur" in scene_name.lower():
                            shift_gt = 0
                        disp_diff = disp_raw / -(gt_disparity - shift_gt)
                        scale = torch.exp((torch.log(disp_diff) * valid_mask).sum(dim=[-1, -2])
                                          / valid_mask.sum(dim=[-1, -2]))
                        dd = torch.reciprocal(depths_raw * -scale) + shift_gt
                    dde = torch.reciprocal(dd)

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
                                             refpose[0].unsqueeze(0),
                                             tarpose[0].unsqueeze(0),
                                             refpose[1].unsqueeze(0),
                                             tarpose[1].unsqueeze(0),
                                             dde)
                    tarview = torch.clamp(tarview, 0, 1)
                    tarviewgt = torch.clamp(tarviewgt, 0, 1).unsqueeze(0)
                    for metric_name in ["ssim", "psnr", "mse", "lpips"]:
                        val = compute_img_metric(tarview, tarviewgt, metric_name, margin=eval_crop_margin)
                        EVAL_DICT_PERSCENE.collect(metric_name, float(val))

                    newviews_save[-1].append(tarviewgt.cpu())
            elif "gt_disparity" in datas.keys():  # only shift left or right
                tarview = shift_newview(mpi, -dd)
                tarviewgt = datas["gt_imgs"][frameidx].unsqueeze(0)
                tarview = torch.clamp(tarview, 0, 1)
                tarviewgt = torch.clamp(tarviewgt, 0, 1)

                for metric_name in ["ssim", "psnr", "mse", "lpips"]:
                    val = compute_img_metric(tarview, tarviewgt, metric_name, margin=eval_crop_margin)
                    EVAL_DICT_PERSCENE.collect(metric_name, float(val))

                newviews_save.append(tarview.cpu())
            else:
                raise RuntimeError(f"Evaluator::Cannot find any gt views")

            # ===========================================
            # Depth quality
            # ===========================================
            denorm = 1000.  # can be arbitrary, now the count is unit by K
            thresh1 = np.log10(1.25)  # log(1.25) ** 2
            thresh2 = np.log10(1.25 ** 2)  # log(1.25) ** 2
            thresh3 = np.log10(1.25 ** 3)  # log(1.25) ** 2
            disp_raw = estimate_disparity_torch(mpi, depths_raw).squeeze(0).cpu()
            disparitys_save.append(disp_raw)
            disp_cor = estimate_disparity_torch(mpi, dde).squeeze(0).cpu()
            disparitys_out.append(disp_cor)

            if "gt_depth" in datas.keys():  # semi-dense depth map
                gt_depth = datas["gt_depth"][frameidx]
                valid_mask = torch.logical_and(-1e3 < gt_depth, gt_depth < 1e3)
                valid_count = valid_mask.sum()
                gt_depth[torch.logical_not(valid_mask)] = 1e3
                gt_disparity = torch.reciprocal(gt_depth)
                disparitys_gts.append(gt_disparity)

                abs_rel = (gt_disparity - disp_cor).abs() / gt_disparity.clamp_min(np.finfo(np.float).eps)
                abs_rel = abs_rel[valid_mask].sum() / denorm

                log10 = torch.log10((disp_cor * gt_depth).abs()).abs()
                thr1 = torch.logical_and(log10 < thresh1, valid_mask).sum() / denorm
                thr2 = torch.logical_and(log10 < thresh2, valid_mask).sum() / denorm
                thr3 = torch.logical_and(log10 < thresh3, valid_mask).sum() / denorm
                log10 = log10[valid_mask].sum() / denorm
                # diff = (gt_disparity - disp_cor).abs()[valid_mask].mean()
                # diff_scale = torch.log10((disp_cor * gt_depth).abs()) ** 2
                # inliner_num = ((diff_scale < thresh) * valid_mask).sum()
                # diff_scale = diff_scale[valid_mask].mean()

                EVAL_DICT_PERSCENE.collect("disp_abs_rel", float(abs_rel), count=valid_count / denorm)
                EVAL_DICT_PERSCENE.collect("disp_log10", float(log10), count=valid_count / denorm)
                EVAL_DICT_PERSCENE.collect("disp_sigma1", float(thr1), count=valid_count / denorm)
                EVAL_DICT_PERSCENE.collect("disp_sigma2", float(thr2), count=valid_count / denorm)
                EVAL_DICT_PERSCENE.collect("disp_sigma3", float(thr3), count=valid_count / denorm)

            elif "gt_disparity" in datas.keys():  # semi-dense disparity map
                gt_disparity = datas["gt_disparity"][frameidx]
                valid_mask = torch.logical_and(-1e3 < gt_disparity, gt_disparity < 1e3)
                valid_count = valid_mask.sum()
                gt_disparity[torch.logical_not(valid_mask)] = 1e3
                disparitys_gts.append(gt_disparity)

                abs_rel = (gt_disparity - disp_cor).abs() / gt_disparity.abs().clamp_min(np.finfo(np.float).eps)
                abs_rel = abs_rel[valid_mask].sum() / denorm

                log10 = torch.log10(((disp_cor - shift_gt) / (gt_disparity - shift_gt)).abs()).abs()
                thr1 = torch.logical_and(log10 < thresh1, valid_mask).sum() / denorm
                thr2 = torch.logical_and(log10 < thresh2, valid_mask).sum() / denorm
                thr3 = torch.logical_and(log10 < thresh3, valid_mask).sum() / denorm
                log10 = log10[valid_mask].sum() / denorm

                # diff = ((disp_cor - gt_disparity).abs() * valid_mask).sum() / valid_mask.sum()
                # diff_scale = torch.log10(((disp_cor - shift_gt) / (gt_disparity - shift_gt)).abs()) ** 2
                # inliner_num = ((diff_scale < thresh) * valid_mask).sum()
                # diff_scale = (diff_scale * valid_mask).sum() / valid_mask.sum()

                EVAL_DICT_PERSCENE.collect("disp_abs_rel", float(abs_rel), count=valid_count / denorm)
                EVAL_DICT_PERSCENE.collect("disp_log10", float(log10), count=valid_count / denorm)
                EVAL_DICT_PERSCENE.collect("disp_sigma1", float(thr1), count=valid_count / denorm)
                EVAL_DICT_PERSCENE.collect("disp_sigma2", float(thr2), count=valid_count / denorm)
                EVAL_DICT_PERSCENE.collect("disp_sigma3", float(thr3), count=valid_count / denorm)

            elif "gt_sparsedepth" in datas.keys():  # sparse depth point
                raise NotImplementedError()
            else:
                raise RuntimeError(f"Evaluator::Cannot find any depth ground truth")

        # ===========================================
        # Depth temporal consistency
        # ===========================================
        # consistency for depth
        refimgs = [img.unsqueeze(0).cuda() for img in datas["in_imgs"]]
        flow_cache = Flow_Cacher(4)
        for frameidx in range(1, len(disparitys_out) - 1):
            idx0, idx1, idx2 = frameidx - 1, frameidx, frameidx + 1
            flow10 = flow_cache.estimate_flow(flow_estim, refimgs[idx1], refimgs[idx0])
            flow01 = flow_cache.estimate_flow(flow_estim, refimgs[idx0], refimgs[idx1])
            flow12 = flow_cache.estimate_flow(flow_estim, refimgs[idx1], refimgs[idx2])
            flow21 = flow_cache.estimate_flow(flow_estim, refimgs[idx2], refimgs[idx1])
            occ10 = torch.norm(warp_flow(flow01, flow10) + flow10, dim=1) < 1.5
            occ12 = torch.norm(warp_flow(flow21, flow12) + flow12, dim=1) < 1.5

            disp01 = warp_flow(disparitys_out[idx0].unsqueeze(0).unsqueeze(0), flow10)
            disp21 = warp_flow(disparitys_out[idx2].unsqueeze(0).unsqueeze(0), flow12)

            diff1 = ((disp01 - disparitys_out[idx1]) * occ10).abs().sum() / denorm
            occ02 = torch.logical_and(occ10, occ12)
            diff2 = ((disp01 + disp21 - 2 * disparitys_out[idx1]) * occ02).abs().sum() / denorm
            EVAL_DICT_PERSCENE.collect("temp_disp_o1", float(diff1), count=float(occ10.sum() / denorm))
            EVAL_DICT_PERSCENE.collect("temp_disp_o2", float(diff2), count=float(occ02.sum() / denorm))
        refimgs.clear()

        # Consistency for novel views
        if isinstance(newviews_save[0], torch.Tensor) \
                and "gt_imgs" in datas.keys() and len(datas["gt_imgs"]) == len(newviews_save):
            newviews_gt = [img.unsqueeze(0).cuda() for img in datas["gt_imgs"]]
            newviews_save_cuda = [i_.cuda() for i_ in newviews_save]
            flow_cache = Flow_Cacher(2)
            for frameidx in range(1, len(newviews_save)):
                flowgt10 = flow_cache.estimate_flow(flow_estim, newviews_gt[frameidx], newviews_gt[frameidx - 1])
                flowgt01 = flow_cache.estimate_flow(flow_estim, newviews_gt[frameidx - 1], newviews_gt[frameidx])
                occgt10 = torch.norm(warp_flow(flowgt01, flowgt10) + flowgt10, dim=1) < 1.5

                flowout10 = flow_cache.estimate_flow(flow_estim, newviews_save_cuda[frameidx],
                                                     newviews_save_cuda[frameidx - 1])

                newview01 = warp_flow(newviews_save[frameidx - 1], flowgt10)
                nvsdiff1 = ((newview01 - newviews_save[frameidx]) * occgt10).abs().sum() / denorm
                nvsepe = (torch.norm(flowout10 - flowgt10, dim=1) * occgt10).sum() / denorm
                EVAL_DICT_PERSCENE.collect("temp_nvs_o1", float(nvsdiff1), count=float(occgt10.sum() / denorm))
                EVAL_DICT_PERSCENE.collect("temp_nvs_epe", float(nvsepe), count=float(occgt10.sum() / denorm))

            newviews_gt.clear()
            newviews_save_cuda.clear()
        # end of per-scene processing
        # ===================================================================
        # Save informations
        # ===================================================================
        persceneinfo_path = os.path.join(saveroot, f"perscene_{scene_name}.txt")
        result_str = f"RANK_{cudaid}:evaluation result of {scene_name}:\n" + EVAL_DICT_PERSCENE.make_table()
        print(result_str, flush=True)
        if save_persceneinfo:
            with open(persceneinfo_path, 'w') as f:
                f.writelines(result_str)

        saveroot_perscene = os.path.join(saveroot, f"{scene_name}")
        mkdir_ifnotexist(saveroot_perscene)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if save_tarviews and isinstance(newviews_save[0], torch.Tensor):
            videoname = os.path.join(saveroot_perscene, f"novelviews.mp4")
            heio, wido = newviews_save[0].shape[-2:]
            writer = cv2.VideoWriter()
            writer.open(videoname, fourcc, 15, (wido, heio), isColor=True)
            for img in newviews_save:
                img = (img[0].permute(1, 2, 0) * 255).type(torch.uint8).numpy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                writer.write(img)
            writer.release()
            print(f"{videoname} saved!")
        if save_disparity:
            videoname = os.path.join(saveroot_perscene, f"disparitymaps.mp4")
            heio, wido = disparitys_save[0].shape[:2]
            writer = cv2.VideoWriter()
            writer.open(videoname, fourcc, 15, (wido, heio), isColor=True)
            for img in disparitys_save:
                img = (img * 255).type(torch.uint8).numpy()
                img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
                writer.write(img)
            writer.release()
            print(f"{videoname} saved!")

        # collect per-scene info
        SELF_EVAL_DICT = SELF_EVAL_DICT + EVAL_DICT_PERSCENE
    return SELF_EVAL_DICT


# warpper for error detection
def process_one(*args):
    try:
        ret = evaluation_one(*args)
    except Exception as e:
        print(f"Error!!!!!!!!!!!!!!!!!!!!!!!!!\n {e}", flush=True)
        ret = None

    return ret


# global dict store dataset-level metric
EVAL_DICT = MetricCounter()


def print_callback(ret_dict):
    global EVAL_DICT
    # collect per-scene info
    EVAL_DICT = EVAL_DICT + ret_dict


def evaluation(cfg):
    num_process = cfg["num_process"]
    datasetname = cfg["dataset"]
    datasetcfg = cfg["datasetcfg"]
    dataset = select_evalset(datasetname, **datasetcfg)

    checkpointname = cfg.pop("checkpoint", "")
    modelname = cfg.pop("model", "auto")
    pipelinename = cfg.pop("pipeline", "auto")
    ret_cfg = cfg.pop("infer_cfg", "")
    pipeline = smart_select_pipeline(
        checkpoint=checkpointname,
        force_modelname=modelname,
        force_pipelinename=pipelinename,
    ).cpu()
    checkpoint = torch.load(checkpointname)
    checkpointcfg = checkpoint["cfg"] if "cfg" in checkpoint.keys() else "no cfg"
    del checkpoint

    eval_crop_margin = cfg.pop("eval_crop_margin", 0)

    saveroot = cfg["saveroot"]
    if saveroot == "auto":
        saveroot = "/home/lmaag/xgpu-scratch/mali_data/Visual"
        if COMPUTER_NAME == "msi":
            saveroot = "Z:\\tmp\\Visual"
            if not os.path.exists(saveroot):
                saveroot = "D:\\MSI_NB\\source\\data\\Visual"
        saveroot = os.path.join(saveroot, f"{datasetname}_{os.path.basename(checkpointname).split('.')[0]}")
    mkdir_ifnotexist(saveroot)
    cfg["saveroot"] = saveroot

    header = f"""
+===================================
| Dataset: {dataset.name}
|       with config: {', '.join([f'{k}: {v}' for k, v in datasetcfg.items()])}
| Date: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
| Model: {modelname}
| Pipeline: {pipelinename}
| checkpoint: {checkpointname}
|       with config: {checkpointcfg}
| inference_cfg: {ret_cfg}
| evaluate_cfg: eval_crop_margin={eval_crop_margin}
+===================================================\n
    """
    print(header, flush=True)

    # start evalutate
    torch.cuda.empty_cache()
    mp.set_start_method('spawn')

    pool = mp.Pool(num_process)
    print(f"num_process, {num_process}")
    for i in range(num_process):
        print(f"launch process {i}")
        pool.apply_async(process_one,
                         (dataset, i, pipeline, cfg.copy()),
                         callback=print_callback)
    pool.close()
    pool.join()
    # end of the entire dataset processing
    global EVAL_DICT
    print(f"Successfully processing all {len(dataset)} scenes/videos")
    print(header)
    print(EVAL_DICT.make_table())
    info_path = os.path.join(saveroot, "all_results.txt")
    with open(info_path, 'w') as f:
        f.writelines(header)
        f.writelines(EVAL_DICT.make_table())
