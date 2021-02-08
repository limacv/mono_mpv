import evaluator


if __name__ == "__main__":
    cfg = {
        # ======== dataset related =========
        "dataset": "StereoVideo",  # StereoVideo
        "datasetcfg": {
            "resolution": (448, 800),
            "max_baseline": 4,

            "seq_len": 20,
            "maxskip": 0
        },

        # ======== model and inference related =========
        "checkpoint": "./log/checkpoint/raSV_finetune_251549_r0.pth",
        "model": "MPINetv2",
        "pipeline": "disp_img",

        # "checkpoint": "./log/checkpoint/mpinet_ori.pth",
        # "model": "MPINetv2",
        # "pipeline": "disp_img",

        # "checkpoint": "./log/checkpoint/V52setcnn_121011_r0.pth",
        # "model": "MPINetv2",
        # "pipeline": "fullv4",
        # "infer_cfg": {},

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
