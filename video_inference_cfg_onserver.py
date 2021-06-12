from utils import *
from models.ModelWithLoss import *
from models.loss_utils import *
from models.mpi_utils import *
from models.flow_utils import *
from testposes import *
import traceback

import torch.backends.cudnn
import os
# Adjust configurations here ########################################################################################
# video_path = "/home/lmaag/xgpu-scratch/mali_data/StereoVideoFinalv3/videos/StereoBlur_HD720-02-15-49-26_0.mp4"
video_path_list = [
    "/d1/scratch/PI/psander/mali_data/StereoVideoFinalv3/videos/testHD720-04-15-33-25.mp4",
    "/home/lmaag/xgpu-scratch/mali_data/StereoVideoFinalv3/videos/StereoBlur_HD720-02-15-49-26_0.mp4",
    "/home/lmaag/xgpu-scratch/mali_data/StereoVideoFinalv3/videos/StereoBlur_HD720-02-15-34-24_1.mp4",
    "/d1/scratch/PI/psander/mali_data/DAVIS_VID/video/blackswan.mp4",
    "/d1/scratch/PI/psander/mali_data/DAVIS_VID/video/bmx-bumps.mp4",
    "/d1/scratch/PI/psander/mali_data/DAVIS_VID/video/camel.mp4",
    "/d1/scratch/PI/psander/mali_data/DAVIS_VID/video/car-roundabout.mp4",
    "/d1/scratch/PI/psander/mali_data/DAVIS_VID/video/soapbox.mp4"
]

pose_list = [
    target_pose1,
    target_pose4,
    target_pose5,
    target_pose_blackswan,
    target_pose_bmx_bumps,
    target_pose_camel,
    target_pose_carroundabout,
    target_pose_soapbox
]

for video_path, newview_pose in zip(video_path_list, pose_list):
    outputres = (848, 480)  # (960, 540)
    framestart = 0
    frameend = 120

    save_disparity = True
    save_newview = True
    save_mpv = True
    save_net = True
    reply_flag = False

    out_prefix = "/home/lmaag/xgpu-scratch/mali_data/VisualNeat/"


    try:
        exec(''.join(open("video_inferencev1.py").read().splitlines(True)[1:]))
        exec(''.join(open("video_inferencev2.py").read().splitlines(True)[1:]))
        exec(''.join(open("video_inferencev3.py").read().splitlines(True)[1:]))
        exec(''.join(open("video_inferencev4.py").read().splitlines(True)[1:]))
        exec(''.join(open("video_inferencelbtc.py").read().splitlines(True)[1:]))
    except Exception as e:
        print(e)
        print(traceback.format_exc())
