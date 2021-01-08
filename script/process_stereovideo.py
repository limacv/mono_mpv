"""
Final step of processing StereoVideo
    Rescale and Compute and save final disparity
"""

import sys

sys.path.append('../')
from models.LEAStereo_network import LEAStereo
from models.flow_utils import RAFTNet, RAFT_path, downflow8
from models.mpi_utils import *
from dataset import StereoVideo_root
import os
import cv2
import torch
from glob import glob
from collections import namedtuple
from torchvision.transforms import ToTensor
import numpy as np
import multiprocessing
import argparse


width_2_list = [
    "1j1-9-CWPAk",
    "VAp_zbH2mMo",
    "YQgufPXF31g",
    "9izczuHy5I8",
    "lN1iEttFjbg",
    "OvuEZMQptLI",
    "0LDAAXTmCnU",

    "5J53betZS-Q",
    "KjdFoJI3Bbw",
]

width_0_5_list = [
    "_fjzc3Aa5Zk",
    "Q4nn6AflK54",
]

swap_list = [
    "fqi0U1NlfwY",
    "GpZK3WWu71I",
    "Q4nn6AflK54",
    "SIIhKaNh7Qg",
    "V62bt7y49D8",
    "WSAcvmGZkY8",
    "6Zn1JJOdbRg",
    "FDm66P63tkY",
    "DQDfm_1wfPk",
]

unintentional_scene_change_list = [
    # "hau6A7j-z4M_0 -> 1",
    # "bfTuSLFzh2M_3 -> 0",
    # "fjTpY5ZanGE_8 -> 11,12",
    # "fjTpY5ZanGE_3 -> 13,14",
    # "3b11LgJpWSQ_0/2/4/5 -> 6/7/8/9",
    # "adFfSYM5J7M_7 -> 20,21",
    # "adFfSYM5J7M_8 -> 22,23",
    # "adFfSYM5J7M_12 -> 24,25",
    # "BZ-WNxyPE-4_8/9/10 -> 11/12/13",
    # "IG6VsR61P3A_2/3/4/5 -> 20/21,22/23/24",
    # "cZJhMUPPWnE_1/2 -> 40/41",
    # "J6-girGi-HM_0 -> 5",
    # "ka3RqjPZLSA_7 -> 12",
    # "lyUhAUjw-PU_15 -> 30",
    # "NOxoJPhj8Hk_1 -> 10"
]

fpsx2_list = [
    "_yB7Q_2rGJo",
    "sh7RPf7YuEE",
    "M2cmtzyYIbc"
]


"""
=======Requirement=======
StereoVideo_stage1/
    StereoBlur/
        *.mp4
    WSVD/
        *.mp4
    Youtube/
        *.mp4

// containing the video list after maunally filter bad video clip
WSVD_list.txt
Youtube_list.txt

=========output==========
StereoVideoFinal/
    videos/
        <Youtube/WSVD/StereoBlur>_name.mp4
    disparitys/
        <Youtube/WSVD/StereoBlur>_name/
            left/
                00000.npy
            right/
                00000.npy
        
"""

flow_bidirect_thresh = 1.5
vertical_disparity_torl = 1.5


def mkdir_ifnotexist(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def print_(*args, **kwargs):
    print(*args, **kwargs)
    pass


def prepareoffset(wid, hei):
    offsety, offsetx = torch.meshgrid([
        torch.linspace(0, hei - 1, hei),
        torch.linspace(0, wid - 1, wid)
    ])
    offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).float().cuda()
    return offset.permute(0, 2, 3, 1)


def detect_black_border(img: np.ndarray):
    img = img.sum(axis=-1).astype(np.float32)
    img_h = np.maximum(img.mean(axis=1) - 20, 0) > 0
    img_w = np.maximum(img.mean(axis=0) - 20, 0) > 0
    top, bottom = np.argmax(img_h), np.argmax(img_h[::-1])
    left, right = np.argmax(img_w), np.argmax(img_w[::-1])
    if top != 0 and bottom != 0:
        top += 5
        bottom += 5
    if left != 0 and right != 0:
        left += 5
        right += 5
    return left, right, top, bottom


def process_clip(videofile, model, scale_func: Callable, fpsx2=False, needswarp=False):
    totensor = ToTensor()
    basename = os.path.basename(videofile).split('.')[0]
    classname = videofile.split(os.sep)[-2]
    video_out_path = os.path.join(Output_prefix, "videos", f"{classname}_{basename}.mp4")
    displ_out_path = os.path.join(Output_prefix, "disparities", f"{classname}_{basename}", "left")
    dispr_out_path = os.path.join(Output_prefix, "disparities", f"{classname}_{basename}", "right")
    visual_out_path = os.path.join(Output_prefix, "visualize", f"{classname}_{basename}.mp4")

    mkdir_ifnotexist(os.path.dirname(video_out_path))
    mkdir_ifnotexist(os.path.dirname(visual_out_path))
    mkdir_ifnotexist(displ_out_path)
    mkdir_ifnotexist(dispr_out_path)

    cap = cv2.VideoCapture(videofile)
    writer = cv2.VideoWriter()
    writer_vis = cv2.VideoWriter()
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"process {classname}_{basename}, total {framecount} frames", flush=True)
    # imglr_last = None
    # flowlr_last = None
    offset = None

    vis_disp_min, vis_disp_max = None, None
    blackl, blackr, blackt, blackb = -1, -1, -1, -1
    for frameidx in range(framecount):
        print(f"{frameidx}", end='.')
        ret, img = cap.read()
        if not ret:
            print(f"{basename}::is not normal regarding frame count, ending at frame {frameidx}")
            break
        if fpsx2:
            if frameidx % 2 == 0:
                frameidx //= 2
            else:
                continue
        hei, wid = img.shape[:2]
        wid = wid // 2
        if not needswarp:
            imgl, imgr = img[:, :wid], img[:, wid:]
        else:
            imgr, imgl = img[:, :wid], img[:, wid:]
        # correct scale
        hei, wid = scale_func(hei, wid)
        if wid > 1280:
            hei = int(1280 * hei / wid)
            wid = 1280
        imgl = cv2.resize(imgl, (wid, hei))
        imgr = cv2.resize(imgr, (wid, hei))

        # detect black border
        if blackl < 0:
            blackl, blackr, blackt, blackb = detect_black_border(imgl)
            blackl1, blackr1, blackt1, blackb1 = detect_black_border(imgr)
            blackl, blackt = max(blackl, blackl1), max(blackt, blackt1)
            blackr, blackb = min(blackr, blackr1), min(blackb, blackb1)
            if blackl + blackr + blackt + blackb > 0:
                print(f"{basename}::black border detected with l{blackl}, r{blackr}, t{blackt}, b{blackb}")

        imgl = imgl[blackt: hei-blackb, blackl: wid-blackr]
        imgr = imgr[blackt: hei-blackb, blackl: wid-blackr]
        hei, wid, _ = imgl.shape

        imglt, imgrt = totensor(cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda(), \
                       totensor(cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
        offset = prepareoffset(wid, hei) if offset is None else offset
        imglrt = torch.cat([imglt, imgrt])

        flowlr_ini = None
        # if imglr_last is None:
        #     flowlr_ini = None
        # else:
        #     with torch.no_grad():
        #         flowlrnowlast = model(imglrt, imglr_last)
        #         flowlr_ini = warp_flow(flowlr_last, flowlrnowlast, offset=offset)
        #         flowlr_ini = downflow8(flowlr_ini)

        with torch.no_grad():
            flowl2r, flowr2l = model(imglrt, torch.cat([imgrt, imglt]),
                                     init_flow=flowlr_ini,
                                     iters=20 if flowlr_ini is None else 12)
            flowl2r, flowr2l = flowl2r.unsqueeze(0), flowr2l.unsqueeze(0)
            occl = warp_flow(flowr2l, flowl2r, offset=offset) + flowl2r
            occl = (torch.norm(occl, dim=1) < flow_bidirect_thresh)[0]
            vflowl = (flowl2r[0, 1].abs() < vertical_disparity_torl)
            invalidl_ma = torch.logical_not(torch.logical_and(occl, vflowl)).cpu().numpy()
            displ = flowl2r[0, 0].cpu().numpy().astype(np.half)

            occr = warp_flow(flowl2r, flowr2l, offset=offset) + flowr2l
            occr = (torch.norm(occr, dim=1) < flow_bidirect_thresh)[0]
            vflowr = (flowr2l[0, 1].abs() < vertical_disparity_torl)
            invalidr_ma = torch.logical_not(torch.logical_and(occr, vflowr)).cpu().numpy()
            dispr = flowr2l[0, 0].cpu().numpy().astype(np.half)

        disp_min = np.min(dispr)
        displ[invalidl_ma] = np.finfo(np.half).min
        dispr[invalidr_ma] = np.finfo(np.half).min

        # for visualization
        vis_disp = np.where(dispr == np.finfo(np.half).min, disp_min, dispr)
        if vis_disp_min is None:
            vis_disp_min = np.min(vis_disp)
            vis_disp_max = np.max(vis_disp)
        vis_disp = (vis_disp - vis_disp_min) / (vis_disp_max - vis_disp_min)
        vis_disp = np.clip(vis_disp, 0, 1)
        vis_disp = (vis_disp * 255).astype(np.uint8)
        vis_disp = cv2.applyColorMap(vis_disp, cv2.COLORMAP_HOT)
        vis_disp[invalidr_ma] = [0, 0, 0]

        scale = 640 / wid
        vis_hei = int(hei * scale)
        vis_img = cv2.resize(imgr, (640, vis_hei), None)
        vis_disp = cv2.resize(vis_disp, (640, vis_hei), None)
        vis_wid = 2 * 640

        # write all the things
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if not writer.isOpened():
            writer.open(video_out_path, fourcc, 30, (wid * 2, hei), isColor=True)
            writer_vis.open(visual_out_path, fourcc, 30, (vis_wid, vis_hei), isColor=True)

        stereoimg = np.hstack([imgl, imgr])
        stereovis = np.hstack([vis_img, vis_disp])
        writer.write(stereoimg)
        writer_vis.write(stereovis)

        np.save(os.path.join(displ_out_path, f"{frameidx:06d}.npy"), displ)
        np.save(os.path.join(dispr_out_path, f"{frameidx:06d}.npy"), dispr)

        # imglr_last = imglrt
        # flowlr_last = torch.cat([flowl2r, flowr2l])

    writer.release()
    writer_vis.release()
    print(f"{basename} OK", flush=True)


def processvideos(video_list, cudaid):
    try:
        torch.cuda.set_device(cudaid)
        model = RAFTNet().eval().cuda()

        state_dict = torch.load(RAFT_path["sintel"], map_location=f'cuda:{cudaid}')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        for video_path in video_list:
            # decide special config based on name
            base_name = os.path.basename(video_path).split('.')[0]
            print(f"||{cudaid}||::process {base_name}", flush=True)
            base_name = base_name.rsplit('_', 1)[0]

            scale_func = (lambda h, w: (h, w))
            fpsx2 = False
            need_swap = False

            if base_name in fpsx2_list:
                # for stereoblur, need to turn 60fps to 30fps
                fpsx2 = True

            if base_name in width_2_list:
                scale_func = (lambda h, w: (h, w * 2))
            elif base_name in width_0_5_list:
                scale_func = (lambda h, w: (h, w // 2))

            if base_name in swap_list:
                need_swap = True

            process_clip(video_path, model,
                         scale_func=scale_func,
                         fpsx2=fpsx2,
                         needswarp=need_swap)
    except Exception as e:
        print(f"Error occur!!! {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_id', dest='work_id', type=int, help="index of num_worker")
    parser.add_argument('--num_worker', dest='num_worker', type=int, default=3, help="total number of Nodes used")
    args = parser.parse_args()

    datasetroot = "/home/lmaag/xgpu-scratch/mali_data"
    num_process = 10
    Source_midfix = "StereoVideo_stage1v2"
    Dest_midfix = "StereoVideoFinalv2"
    Source_prefix = os.path.join(datasetroot, Source_midfix)
    Output_prefix = os.path.join(datasetroot, Dest_midfix)

    # stereo_blur_videos = glob(os.path.join(Source_prefix, "StereoBlur", "*.mp4"))
    # wsvd_videos = glob(os.path.join(Source_prefix, "WSVD", "*.mp4"))
    # youtube_videos = glob(os.path.join(Source_prefix, "Youtube", "*.mp4"))
    with open(os.path.join(Source_prefix, "StereoBlur_list.txt"), 'r') as f:
        lines = f.readlines()
        stereo_blur_videos = [os.path.join(Source_prefix, "StereoBlur", l_.strip('\n')) for l_ in lines]
    with open(os.path.join(Source_prefix, "WSVD_list.txt"), 'r') as f:
        lines = f.readlines()
        wsvd_videos = [os.path.join(Source_prefix, "WSVD", l_.strip('\n')) for l_ in lines]
    with open(os.path.join(Source_prefix, "Youtube_list.txt"), 'r') as f:
        lines = f.readlines()
        youtube_videos = [os.path.join(Source_prefix, "Youtube", l_.strip('\n')) for l_ in lines]

    all_videos = sorted(stereo_blur_videos + wsvd_videos + youtube_videos)
    all_videos = all_videos[args.work_id::args.num_worker]

    po = multiprocessing.Pool(num_process)
    for i in range(num_process):
        po.apply_async(processvideos, [all_videos[i::num_process], i])

    po.close()
    po.join()
    print("------end------")
