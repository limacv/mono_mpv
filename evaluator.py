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

    def collect(self, name, val, count=1):
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
                    for metric_name in ["ssim", "psnr", "mse"]:
                        val = compute_img_metric(tarview, tarviewgt, metric_name, margin=eval_crop_margin)
                        EVAL_DICT_PERSCENE.collect(metric_name, float(val))

                    newviews_save[-1].append(tarviewgt.cpu())
            elif "gt_disparity" in datas.keys():  # only shift left or right
                tarview = shift_newview(mpi, -dd)
                tarviewgt = datas["gt_imgs"][frameidx].unsqueeze(0)
                tarview = torch.clamp(tarview, 0, 1)
                tarviewgt = torch.clamp(tarviewgt, 0, 1)

                for metric_name in ["ssim", "psnr", "mse"]:
                    val = compute_img_metric(tarview, tarviewgt, metric_name, margin=eval_crop_margin)
                    EVAL_DICT_PERSCENE.collect(metric_name, float(val))

                newviews_save.append(tarview.cpu())
            else:
                raise RuntimeError(f"Evaluator::Cannot find any gt views")

            # ===========================================
            # Depth quality
            # ===========================================
            disp_raw = estimate_disparity_torch(mpi, depths_raw).squeeze(0).cpu()
            disparitys_save.append(disp_raw)
            disp_cor = estimate_disparity_torch(mpi, dde).squeeze(0)
            if "gt_depth" in datas.keys():  # semi-dense depth map
                gt_depth = datas["gt_depth"][frameidx]
                valid_mask = torch.logical_and(-1e3 < gt_depth, gt_depth < 1e3)
                gt_depth[torch.logical_not(valid_mask)] = 1e3
                diff = (torch.reciprocal(gt_depth) - disp_cor).abs()[valid_mask].mean()
                diff_scale = torch.log10((disp_cor * gt_depth).abs()) ** 2
                thresh = np.log10(1.25) ** 2  # log(1.25) ** 2
                inliner_num = ((diff_scale < thresh) * valid_mask).sum()
                diff_scale = diff_scale[valid_mask].mean()

                EVAL_DICT_PERSCENE.collect("disp_mae", float(diff))
                EVAL_DICT_PERSCENE.collect("disp_msle", float(diff_scale))
                EVAL_DICT_PERSCENE.collect("disp_goodpct", float(inliner_num), int(valid_mask.sum()))

            elif "gt_disparity" in datas.keys():  # semi-dense disparity map
                gt_disparity = datas["gt_disparity"][frameidx]
                valid_mask = torch.logical_and(-1e3 < gt_disparity, gt_disparity < 1e3)
                gt_disparity[torch.logical_not(valid_mask)] = 1e3

                diff = ((disp_cor - gt_disparity).abs() * valid_mask).sum() / valid_mask.sum()
                diff_scale = torch.log10(((disp_cor - shift_gt) / (gt_disparity - shift_gt)).abs()) ** 2
                thresh = np.log10(1.25) ** 2  # log10(1.25) ** 2
                inliner_num = ((diff_scale < thresh) * valid_mask).sum()
                diff_scale = (diff_scale * valid_mask).sum() / valid_mask.sum()

                EVAL_DICT_PERSCENE.collect("disp_mae", float(diff))
                EVAL_DICT_PERSCENE.collect("disp_msle", float(diff_scale))
                EVAL_DICT_PERSCENE.collect("disp_goodpct", float(inliner_num), int(valid_mask.sum()))

            elif "gt_sparsedepth" in datas.keys():  # sparse depth point
                raise NotImplementedError()
            else:
                raise RuntimeError(f"Evaluator::Cannot find any depth ground truth")

            # ===========================================
            # Depth temporal consistency
            # ===========================================
            pass  # TODO

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
