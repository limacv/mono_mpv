import evaluator


if __name__ == "__main__":
    cfg = {
        # ======== dataset related =========
        "dataset": "NvidiaNovelView",  # StereoVideo
        "datasetcfg": {
            "resolution": (448, 800),  # (540, 960)
            "max_baseline": 4,
            
            "seq_len": 20,
            "maxskip": 0
        },

        # ======== model and inference related =========
        # "checkpoint": "./log/checkpoint/raSV_scratch_s103_040031_r0.pth",
        # "model": "MPINetv2",
        # "pipeline": "disp_img",

        # "checkpoint": "./log/checkpoint/mpinet_ori.pth",
        # "model": "MPINetv2",
        # "pipeline": "disp_img",

        "checkpoint": "./log/checkpoint/V52setcnn_121011_r0.pth",
        # "model": "MPINetv2",
        "pipeline": "fullv4",
        "infer_cfg": {},

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
