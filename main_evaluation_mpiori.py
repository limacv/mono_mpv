import evaluator


if __name__ == "__main__":
    cfg = {
        # ======== dataset related =========
        "num_process": 3,
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
        "checkpoint": "./log/checkpointsave/ablation00_svbase_r0_4.pth",
        "model": "MPINetv2",
        "pipeline": "disp_img",

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
