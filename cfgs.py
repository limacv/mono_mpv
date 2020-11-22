single_view_cfg = {
    # const configuration <<<<<<<<<<<<<<<<
    "cuda_device": 0,
    "log_prefix": "./log/",
    "tensorboard_logdir": "run/",
    "mpi_outdir": "mpi/",
    "checkpoint_dir": "checkpoint/",

    "write_validate_result": True,
    "validate_num": 32,
    "valid_freq": 500,
    "train_report_freq": 1,  # 10

    # about training <<<<<<<<<<<<<<<<
    # comment of current epoch, will print on config.txt
    "comment": "<Please add comment in experiments>",
    "model_name": "MPINet",
    "batch_size": 2,
    "num_epoch": 1000,
    "savepth_iter_freq": 500,
    "sample_num_per_epoch": -1,  # < 0 means randompermute
    "lr": 5e-5,
    "check_point": "MPINet/mpinet_ori.pth",  # relative to log_prefix
    "loss_weights": {
        "pixel_loss_cfg": 'l1',
        "pixel_loss": 1,
        "smooth_loss": 0.5,
        "depth_loss": 0.1,
        "sparse_loss": 0,
    },
}