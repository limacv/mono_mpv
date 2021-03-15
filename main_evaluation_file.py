import evaluator
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate for fullv4")
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()

    checkpointpath = args.checkpoint
    if not os.path.exists(checkpointpath):
        raise RuntimeError(f"{checkpointpath} not exist")
    cfg = {
        # ======== dataset related =========
        "num_process": 2,
        "dataset": "StereoVideo",  # StereoVideo, NvidiaNovelView
        "datasetcfg": {
            "resolution": (448, 768),  # (540, 960)
            "max_baseline": 4,

            "seq_len": 20,
            "maxskip": 2
        },
        "const_scale": True,
        "scale_in_log": False,

        # ======== model and inference related =========
        "checkpoint": checkpointpath,
        "pipeline": "fullv4",
        "infer_cfg": "",

        # ======= computing error related ===========
        "eval_crop_margin": 0.1,

        # ======= saving results related ==========
        # actualsavepath = "saveroot/scenebasename/<item>
        # "auto" means saveroot = "datasetname_modelname_pipelinename/
        # set to "no" to disable saving
        "saveroot": "auto",
        "save_perscene": True,
        "save_tarviews": True,
        "save_disparity": True,
    }
    evaluator.evaluation(cfg)
