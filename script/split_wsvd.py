import os
import cv2
from glob import glob
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import subprocess
import multiprocessing
import shutil
import imageio
import pickle

ffmpeg_exe = "/home/lmaag/miniconda3/bin/ffmpeg"
WSVD_root = "/home/lmaag/xgpu-scratch/mali_data/WSVD/"
WSVD_out_root = "/home/lmaag/xgpu-scratch/mali_data/WSVD_processed"
WSVD_tumbnail = "/home/lmaag/xgpu-scratch/mali_data/WSVD_tumbnail"
Error_list = "/home/lmaag/xgpu-scratch/mali_data/WSVD_processed/error_list.txt"

# ffmpeg_exe = "D:\\MSI_NB\\source\\util\\ffmpeg\\bin\\ffmpeg.exe"
# WSVD_root = "D:\\MSI_NB\\source\\data\\WSVD\\"
# WSVD_tmp_root = "D:\\MSI_NB\\source\\data\\WSVD_tmp\\"
# WSVD_out_root = "D:\\MSI_NB\\source\\data\\WSVD_processed\\"
# WSVD_tumbnail = "D:\\MSI_NB\\source\\data\\WSVD_thumbnail\\"
# Error_list = "D:\\MSI_NB\\source\\data\\WSVD_processed\\error_list.txt"
if not os.path.exists(os.path.join(WSVD_out_root, "train")):
    os.makedirs(os.path.join(WSVD_out_root, "train"))
if not os.path.exists(os.path.join(WSVD_out_root, "test")):
    os.makedirs(os.path.join(WSVD_out_root, "test"))
if not os.path.exists(os.path.join(WSVD_tumbnail, "train")):
    os.makedirs(os.path.join(WSVD_tumbnail, "train"))
if not os.path.exists(os.path.join(WSVD_tumbnail, "test")):
    os.makedirs(os.path.join(WSVD_tumbnail, "test"))
output_tumbnail = True

print("loading frameids", flush=True)
with open(os.path.join(WSVD_root, "wsvd_train_clip_frame_ids.pkl"), 'rb') as f:
    train_clips = pickle.load(f)
    train_clips = {vid["name"].split('.')[0]: vid["clips"] for vid in train_clips}

with open(os.path.join(WSVD_root, "wsvd_test_clip_frame_ids.pkl"), 'rb') as f:
    test_clips = pickle.load(f)
    test_clips = {vid["name"].split('.')[0]: vid["clips"] for vid in test_clips}


def runffmpeg(args):
    try:
        subprocess.check_output(args)
    except Exception as e:
        print(e)
        return False
    return True


def split_one_video(video_file: str, a, b):
    print(f"({a}/{b} {100 * a / b:.2f}%)processing video {video_file}", flush=True)
    basename = os.path.splitext(os.path.basename(video_file))[0]
    trainstr = "train"
    try:
        clips = train_clips[basename]
    except KeyError:
        try:
            clips = test_clips[basename]
            trainstr = "test"
        except KeyError:
            print(f"cannot find {basename} in pickle file")
            with open(Error_list, 'a') as errfile:
                errfile.write(f"{basename}\tall  \tcannot find in pickle file\n")
            return None

    # first save to images
    cap = cv2.VideoCapture(video_file)
    ret, img = cap.read()
    if not ret:
        print(f"!error when open {video_file}")
        with open(Error_list, 'a') as errfile:
            errfile.write(f"{basename}\tall  \tcannot open video file\n")
        return None

    cap.release()
    for clipid, clip in enumerate(clips):
        min_frameidx = clip["frames"].min()
        max_frameidx = clip["frames"].max()
        need_lrswarp = (clip["label"] == -1)
        output_file = os.path.join(WSVD_out_root, trainstr, f"{basename}_clip{clipid}.mp4")
        if not need_lrswarp:
            ret = runffmpeg([
                ffmpeg_exe,
                '-y', '-hide_banner', '-loglevel', 'panic',
                '-i', video_file,
                '-vf', f'trim=start_frame={min_frameidx}:end_frame={max_frameidx + 1}, '
                       f'scale=iw*2:ih,setsar=1',
                output_file
            ])
        else:
            ret = runffmpeg([
                ffmpeg_exe,
                '-y', '-hide_banner', '-loglevel', 'panic',
                '-i', video_file,
                '-filter_complex', f'[0:v]trim=start_frame={min_frameidx}:end_frame={max_frameidx + 1}[in];'
                                   f'[in]split[inleft][inright];'
                                   f'[inleft]crop=iw/2:ih:0:0, scale=iw*2:ih,setsar=1[left];'
                                   f'[inright]crop=iw/2:ih:iw/2:0, scale=iw*2:ih,setsar=1[right];'
                                   f'[right][left]hstack',
                output_file
            ])

        if not ret:
            with open(Error_list, 'a') as errfile:
                errfile.write(f"{basename}\tclip {clipid}\tffmpeg error\n")
            continue

        if output_tumbnail:
            tumbnail_file = os.path.join(WSVD_tumbnail, trainstr, f"{basename}_clip{clipid}.mp4")
            runffmpeg([
                ffmpeg_exe, '-hide_banner', '-loglevel', 'panic',
                '-y',
                '-i', output_file,
                '-vf', 'scale=640:-1, setpts=0.25*PTS',
                tumbnail_file
            ])
            if not ret:
                with open(Error_list, 'a') as errfile:
                    errfile.write(f"{basename}\tclip {clipid}\tffmpeg error when tumbnailing\n")
                continue


if __name__ == "__main__":
    video_list = glob(os.path.join(WSVD_root, "videos", "*.mp4"))
    with open(Error_list, 'a') as errfile:
        errfile.write(f"=================newrun======================")
    po = multiprocessing.Pool(16)
    for i, video_file in enumerate(video_list):
        po.apply_async(split_one_video, [video_file, i, len(video_list)])
        # split_one_video(video_file, i)

    po.close()
    po.join()
    print("------end------")
