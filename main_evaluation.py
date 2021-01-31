import evaluator


if __name__ == "__main__":
    cfg = {
        "dataset": "NvidiaNovelView",
        "checkpoint": "./log/checkpoint/mpinet_ori.pth",
        "model": "MPINetv2",
        "pipeline": "disp_img",
        "infer_cfg": {}
    }
    evaluator.evaluation(cfg)
