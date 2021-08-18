import evaluator


if __name__ == "__main__":
    cfg = {
        # ======== dataset related =========
        "num_process": 1,
        "dataset": "StereoVideo",  # StereoVideo, NvidiaNovelView
        "datasetcfg": {
            "resolution": (448, 768),  # (540, 960)
            "max_baseline": 4,
            "proj_to_view1": True,

            "seq_len": 20,
            "maxskip": 2
        },
        "const_scale": True,
        "scale_in_log": False,

        # ======== model and inference related =========
        "checkpoint": "./log/checkpoint/LDI2MPI_newCEL5_r0.pth",
        "pipeline": "ldifilter",
        "pipelinecfg": {"winsz": 7, "forwardaf_idx": [0, 2, -1]},
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
