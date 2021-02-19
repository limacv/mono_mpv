import evaluator


if __name__ == "__main__":
    cfg = {
        # ======== dataset related =========
        "num_process": 2,
        "dataset": "StereoVideo",  # StereoVideo, NvidiaNovelView
        "datasetcfg": {
            "resolution": (448, 800),  # (540, 960)
            "max_baseline": 4,

            "seq_len": 15,
            "maxskip": 0
        },

        # ======== model and inference related =========
        "checkpoint": "./log/checkpoint/raSV_scratch_adapts_122129_r0.pth",
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
