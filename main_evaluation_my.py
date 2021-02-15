import evaluator


if __name__ == "__main__":
    cfg = {
        # ======== dataset related =========
        "num_process": 3,
        "dataset": "StereoVideo",  # StereoVideo, NvidiaNovelView
        "datasetcfg": {
            "resolution": (448, 800),  # (540, 960)
            "max_baseline": 4,

            "seq_len": 20,
            "maxskip": 0
        },

        # ======== model and inference related =========
        "checkpoint": "./log/checkpoint/DispSpace_124113_r0.pth",
        "pipeline": "fullv4",
        "infer_cfg": "hardbw",

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
