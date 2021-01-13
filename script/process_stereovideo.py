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
from collections import deque


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


class Flow_Cacher:
    def __init__(self, max_sz):
        self.sz = max_sz
        self.flow_cache = {}
        self.history_que = deque()

    def estimate_flow(self, model, im0, im1):
        key = f"{id(im0)}_{id(im1)}"
        if key in self.flow_cache:
            # print("cache hit!")
            return self.flow_cache[key]
        else:
            with torch.no_grad():
                flow = model(im0, im1)
            self.flow_cache[key] = flow
            self.history_que.append(key)
            if len(self.history_que) > self.sz:
                self.flow_cache.pop(self.history_que.popleft())
            return flow


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
    imgl_window, imgr_window = deque(), deque()
    imglt_window, imgrt_window = deque(), deque()
    displ_window, dispr_window = deque(), deque()
    flow_cachel = Flow_Cacher(100)
    flow_cacher = Flow_Cacher(100)

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
        offset = prepareoffset(wid, hei) if offset is None else offset

        imglt, imgrt = totensor(cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda(), \
                       totensor(cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
        # ==============================================
        # predicting the current disparity map
        # ==============================================
        val_ninf = torch.tensor(float('-inf')).type_as(imglt)
        val_inf = torch.tensor(float('inf')).type_as(imglt)
        val_false = torch.tensor(False).to(imglt.device)
        with torch.no_grad():
            flowl2r, flowr2l = model(torch.cat([imglt, imgrt]), torch.cat([imgrt, imglt]),
                                     iters=20)
            flowl2r, flowr2l = flowl2r.unsqueeze(0), flowr2l.unsqueeze(0)
            occl = warp_flow(flowr2l, flowl2r, offset=offset) + flowl2r
            occl = (torch.norm(occl, dim=1, keepdim=True) < flow_bidirect_thresh)
            vflowl = (flowl2r[:, 1:].abs() < vertical_disparity_torl)
            invalidl_ma = torch.logical_not(torch.logical_and(occl, vflowl))
            displ = torch.where(invalidl_ma, val_ninf, flowl2r[:, 0:1])

            occr = warp_flow(flowl2r, flowr2l, offset=offset) + flowr2l
            occr = (torch.norm(occr, dim=1, keepdim=True) < flow_bidirect_thresh)
            vflowr = (flowr2l[0, 1:].abs() < vertical_disparity_torl)
            invalidr_ma = torch.logical_not(torch.logical_and(occr, vflowr))
            dispr = torch.where(invalidr_ma, val_ninf, flowr2l[:, 0:1])

        # ==========================================
        # filtering
        # ==========================================
        imgl_window.append(imgl)
        imgr_window.append(imgr)
        imglt_window.append(imglt)
        imgrt_window.append(imgrt)
        displ_window.append(displ)
        dispr_window.append(dispr)
        if len(imgl_window) < Filter_WinSz:
            continue
        elif len(imgl_window) > Filter_WinSz:
            imgl_window.popleft()
            imgr_window.popleft()
            imglt_window.popleft()
            imgrt_window.popleft()
            displ_window.popleft()
            dispr_window.popleft()

        mididx = Filter_WinSz // 2

        def filter_depth(cache: Flow_Cacher, img_win, disp_win):
            disp_list = []
            for i in range(Filter_WinSz):
                if i == mididx:
                    disp_list.append(disp_win[i])
                else:
                    flowf = cache.estimate_flow(model, img_win[mididx], img_win[i])
                    flowb = cache.estimate_flow(model, img_win[i], img_win[mididx])
                    dispwarp = warp_flow(disp_win[i], flowf, offset=offset, mode='nearest')
                    occ = warp_flow(flowb, flowf, offset=offset) + flowf
                    occ = torch.norm(occ, dim=1, keepdim=True)
                    disp = torch.where(occ < 2, dispwarp, val_ninf)
                    disp_list.append(disp)

            disp_all = torch.cat(disp_list, dim=0)
            good_ma = (disp_all > -9999)
            condition1 = good_ma.type(torch.float).sum(dim=0, keepdim=True) >= Filter_good_time_min
            good_ma = torch.where(condition1, good_ma, val_false)

            maxval = disp_all.max(dim=0, keepdim=True)[0]
            minval = torch.where(good_ma, disp_all, val_inf).min(dim=0, keepdim=True)[0]
            condition2 = (maxval - minval) < Filter_max_diff_tol
            good_ma = torch.where(condition2, good_ma, val_false)
            condition = torch.logical_and(condition1, condition2)

            disp_all = torch.where(good_ma, disp_all, torch.tensor(0.).cuda())
            weight = torch.tensor(Filter_weight).reshape(-1, 1, 1, 1).type_as(disp_all)
            weight = torch.where(good_ma, weight, torch.tensor(0.).cuda())
            weightsum = weight.sum(dim=0, keepdim=True) + 0.0000001
            disp = (disp_all * weight).sum(dim=0, keepdim=True) / weightsum
            disp = torch.where(condition, disp, val_ninf)
            return disp.squeeze(0).squeeze(0)

        imgl = imgl_window[mididx]
        imgr = imgr_window[mididx]

        displ = filter_depth(flow_cachel, imglt_window, displ_window)
        dispr = filter_depth(flow_cacher, imgrt_window, dispr_window)
        displ = displ.cpu().numpy()
        dispr = dispr.cpu().numpy()

        # ==============================================
        # visualization and save
        # ==============================================
        invalidr_ma = dispr < -99999
        vis_disp = np.where(invalidr_ma, dispr.max(), dispr)
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
            outframeidx = 0

        stereoimg = np.hstack([imgl, imgr])
        stereovis = np.hstack([vis_img, vis_disp])
        writer.write(stereoimg)
        writer_vis.write(stereovis)

        np.save(os.path.join(displ_out_path, f"{outframeidx:06d}.npy"), displ)
        np.save(os.path.join(dispr_out_path, f"{outframeidx:06d}.npy"), dispr)
        outframeidx += 1

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
    # datasetroot = "/home/lmaag/xgpu-scratch/mali_data"
    # num_process = 10
    # Source_midfix = "StereoVideo_stage1"
    # Dest_midfix = "StereoVideoFinal"
    # Source_prefix = os.path.join(datasetroot, Source_midfix)
    # Output_prefix = os.path.join(datasetroot, Dest_midfix)
    # with open(os.path.join(datasetroot, "StereoVideo_stage1v2", "v3_list.txt"), 'r') as f:
    #     lines = f.readlines()
    # processlist = [os.path.join(Source_prefix, "WSVD", l_.strip('\n').split('_', 1)[1] + ".mp4") for l_ in lines]
    # po = multiprocessing.Pool(num_process)
    # for i in range(num_process):
    #     po.apply_async(processvideos, [processlist[i::num_process], i])
    # exit(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--work_id', dest='work_id', type=int, help="index of num_worker")
    parser.add_argument('--num_worker', dest='num_worker', type=int, default=3, help="total number of Nodes used")
    args = parser.parse_args()

    datasetroot = "/home/lmaag/xgpu-scratch/mali_data"
    num_process = 10
    Source_midfix = "StereoVideo_stage1v2"
    Dest_midfix = "StereoVideoFinalv3"
    Filter_WinSz = 7
    Filter_weight = (0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05)
    Filter_max_diff_tol = 8
    Filter_good_time_min = 3
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
