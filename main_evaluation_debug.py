import evaluator


if __name__ == "__main__":
    cfg = {
        # ======== dataset related =========
        "num_process": 1,
        "dataset": "StereoVideo",  # StereoVideo, NvidiaNovelView
        "datasetcfg": {
            "resolution": (448, 768),  # (540, 960)
            "max_baseline": 3,
            
            "seq_len": 7,
            "maxskip": 0
        },
        "const_scale": True,
        "scale_in_log": False,

        # ======== model and inference related =========
        # "checkpoint": "./log/checkpoint/raSV_scratch_adapts_122129_r0.pth",
        # "model": "MPINetv2",
        # "pipeline": "disp_img",

        "checkpoint": "./log/checkpoint/mpinet_ori.pth",
        "model": "MPINetv2",
        "pipeline": "disp_img",

        # "checkpoint": "./log/checkpoint/V52setcnn_adapts_121814_r0.pth",
        # "pipeline": "fullv4",
        # "infer_cfg": "hardbw",

        # ======= computing error related ===========
        "eval_crop_margin": 0.1,

        # ======= saving results related ==========
        # actualsavepath = "saveroot/scenebasename/<item>
        # "auto" means saveroot = "datasetname_modelname_pipelinename/
        # set to "no" to disable saving
        "saveroot": "auto",
        "save_perscene": True,
        "save_tarviews": False,
        "save_disparity": True,
    }
    evaluator.evaluation(cfg)
