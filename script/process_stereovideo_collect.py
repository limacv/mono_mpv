# pre-process the stereoblur && collect all the clip name to WSVD_list.txt and Youtube_list.txt

from glob import glob
import os
import cv2

# StereoBlur_SRC = "/home/lmaag/xgpu-scratch/mali_data/StereoVideoGood/StereoBlur"
# StereoBlur_DST = "/home/lmaag/xgpu-scratch/mali_data/StereoVideo_stage1v2/StereoBlur"
#
# videonames = glob(os.path.join(StereoBlur_SRC, "*.mp4"))
# for videoname in videonames:
#     print(videoname)
#     cap = cv2.VideoCapture(videoname)
#     writer = cv2.VideoWriter()
#     frameend = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     scene_count = 0
#     for frameidx in range(frameend):
#         ret, frame = cap.read()
#         if frameidx % 2 == 0:
#             continue
#         if writer.isOpened():
#             writer.write(frame)
#         else:
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             writer.open(os.path.join(StereoBlur_DST,
#                                      f"{os.path.basename(videoname).split('.')[0]}_{scene_count}.mp4"),
#                         fourcc, 30, (2560, 720), True)
#
#         if (frameidx + 1) % 300 == 0:
#             scene_count += 1
#             writer.release()
#             if frameend - frameidx < 300:
#                 break
#
# print("finish process stereoblur")

dataset_root = "Z:\\dataset\\StereoVideo_stage1v2_vis\\"
out_root = "Z:\\dataset\\StereoVideo_stage1v2"

wsvd_list = glob(os.path.join(dataset_root, "WSVD", "*.mp4"))
ytb_list = glob(os.path.join(dataset_root, "Youtube", "*.mp4"))
stereoblur_list = glob(os.path.join(out_root, "StereoBlur", "*.mp4"))

wsvd_list = [os.path.basename(i) for i in wsvd_list]
ytb_list = [os.path.basename(i) for i in ytb_list]
stereoblur_list = [os.path.basename(i) for i in stereoblur_list]

with open(os.path.join(out_root, "WSVD_list.txt"), 'w') as f:
    for line in wsvd_list:
        f.writelines(line + '\n')

with open(os.path.join(out_root, "Youtube_list.txt"), 'w') as f:
    for line in ytb_list:
        f.writelines(line + '\n')

with open(os.path.join(out_root, "StereoBlur_list.txt"), 'w') as f:
    for line in stereoblur_list:
        f.writelines(line + '\n')
