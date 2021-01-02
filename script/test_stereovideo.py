import os
import cv2
from glob import glob
import numpy as np

root = "/home/lmaag/xgpu-scratch/mali_data/StereoVideoFinal"
videos = sorted(glob(os.path.join(root, "videos/*.mp4")))

with open(os.path.join(root, "failed.txt"), 'w') as f:
    for video in videos:
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 15:
            print(f"{video} too less frame {frame_count}", flush=True)

        disp_path = video.replace("videos/", "disparities/").split('.')[0]
        try:
            displ = np.load(os.path.join(disp_path, "left", f"{frame_count // 2:06d}.npy"))
            # dispr = np.load(os.path.join(disp_path, "right", f"{frame_count // 2:06d}.npy"))
        except FileNotFoundError:
            print(f"{video} cannot find disparity", flush=True)
            # f.writelines(video)

        displ[displ == np.finfo(np.half).min] = 0
        print(f"min={displ.min():.2f}, max={displ.max():.2f}:{video}", flush=True)

print("-------------end-------------", flush=True)
