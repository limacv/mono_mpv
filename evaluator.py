from utils import *
from models.mpi_utils import *


def evaluation(cfg):
    datasetname = cfg["dataset"]
    checkpointname = cfg.pop("checkpoint", "")
    modelname = cfg.pop("model", "auto")
    pipelinename = cfg.pop("pipeline", "auto")
    ret_cfg = cfg.pop("infer_cfg", {})

    dataset = select_evalset(datasetname, max_baseline=3)
    pipeline = smart_select_pipeline(
        checkpoint=checkpointname,
        force_modelname=modelname,
        force_pipelinename=pipelinename,
    )

    for datas in dataset:
        depth_map_list = []
        new_view_list = []
        scale = None
        depths = make_depths(32).cuda()
        for frameidx, refimg in enumerate(datas["in_imgs"]):
            mpi = pipeline.infer_forward(refimg.unsqueeze(0).cuda(), ret_cfg)

            gt_depth = datas["gt_depth"][frameidx].cuda()
            valid_mask = torch.logical_and(0 < gt_depth, gt_depth < 999)
            gt_depth[torch.logical_not(valid_mask)] = 999
            # estimate the scale so to scaling the depth to gt
            if scale is None:
                disp = estimate_disparity_torch(mpi, depths).squeeze(0)
                scale = torch.exp((torch.log(disp * gt_depth) * valid_mask.type_as(disp)).sum(dim=[-1, -2])
                                  / valid_mask.sum(dim=[-1, -2]))
                depths *= scale
                disp /= scale
            else:
                disp = estimate_disparity_torch(mpi, depths)

            refpose = datas["in_poses"][frameidx]
            for tarpose, tarviewgt in zip(datas["gt_poses"][frameidx], datas["gt_imgs"][frameidx]):
                tarview = render_newview(mpi,
                                         refpose[0].unsqueeze(0).cuda(),
                                         tarpose[0].unsqueeze(0).cuda(),
                                         refpose[1].unsqueeze(0).cuda(),
                                         tarpose[1].unsqueeze(0).cuda(),
                                         depths)
                mse = tarview - tarviewgt

