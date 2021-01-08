"""
Stage1 of processing StereVideo
    Split Scene, pre-filter, and visualize the Scene
    Source folder: <Dataset_path>/StereVideoGood
    Destination folder: <Dataset_path>/StereVideo_stage1  &  <Dataset_path>/StereVideo_stage1_vis
"""
import sys
sys.path.append('../')
from models.LEAStereo_network import LEAStereo
from models.flow_utils import RAFTNet, RAFT_path
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


print(f"Process the StereoVideo dataset")


def mkdir_ifnotexist(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def print_(*args, **kwargs):
    # print(*args, **kwargs)
    pass


def prepareoffset(wid, hei):
    offsety, offsetx = torch.meshgrid([
        torch.linspace(0, hei - 1, hei),
        torch.linspace(0, wid - 1, wid)
    ])
    offset = torch.stack([offsetx, offsety], 0).unsqueeze(0).float().cuda()
    return offset.permute(0, 2, 3, 1)


def processvideo(videofile, model):
    outprefix = videofile.replace(Source_midfix, Target_midfix).split('.')[0]
    if os.path.exists(f"{outprefix}_0.mp4"):
        print(f"GoodNews!!! {videofile} has already be processed!")
        return
    print(f"process {videofile}", flush=True)

    def isnewscene(img_last, img_now):
        # scene detection, always try to promote new scene easily
        if len(scene_img_list) > scene_max_frame_num:
            return True
        if img_now.mean() < black_frame_threshold:
            print(f"New Scene detected due to black frame", end=' ', flush=True)
            return True
        with torch.no_grad():
            flowlastnow = model(img_last, img_now)
            flownowlast = model(img_now, img_last)
            occlast = warp_flow(flownowlast, flowlastnow, offset=offset) + flowlastnow
            occlast = (torch.norm(occlast, dim=1) < flow_bidirect_thresh).type(torch.float)
            occ_pct = 1. - occlast.sum() / occlast.nelement()
            print_(f"occ_pct={float(occ_pct):.3f}")
            if occ_pct > 0.7:
                print(f"New Scene detected due to large occ value", end=' ', flush=True)
                return True
            warp = warp_flow(img_now, flowlastnow, offset=offset)
            l1 = torch.sum((warp - img_last).abs().sum(dim=1) * occlast) / occlast.sum()
            print_(f"l1_value={float(l1):.3f}")
            if l1 > scene_max_l1diff:
                print(f"New Scene detected due to large l1 value", end=' ', flush=True)
                return True
            return False

    totensor = ToTensor()

    outprefix = videofile.replace(Source_midfix, Target_midfix).split('.')[0]
    visprefix = videofile.replace(Source_midfix, f"{Target_midfix}_vis").split('.')[0]
    mkdir_ifnotexist(os.path.dirname(outprefix))
    mkdir_ifnotexist(os.path.dirname(visprefix))

    cap = cv2.VideoCapture(videofile)
    writer = cv2.VideoWriter()
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_last = None
    offset = None

    scene_idx = 0
    scene_img_list = []
    scene_dispvis_list = []
    vis_disp_min, vis_disp_max = None, None
    for frameidx in range(1, framecount):
        print(f"{frameidx}.", end='', flush=True)
        ret, img = cap.read()
        if not ret:
            print(f"{videofile} is not normal regarding frame count, ending at frame {frameidx}")
            break
        hei, wid = img.shape[:2]
        imgl, imgr = img[:, :wid//2], img[:, wid//2:]
        if wid < 2 * hei:
            imgl = cv2.resize(imgl, (wid, hei))
            imgr = cv2.resize(imgr, (wid, hei))
        else:
            wid = wid // 2
        if wid >= 960:
            hei = int(hei / wid * 960)
            wid = 960
            imglt = cv2.resize(imgl, (wid, hei))
            imgrt = cv2.resize(imgr, (wid, hei))
        else:
            imglt, imgrt = imgl, imgr
        img_vis = imglt
        imglt, imgrt = totensor(cv2.cvtColor(imglt, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda(), \
                       totensor(cv2.cvtColor(imgrt, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
        offset = prepareoffset(wid, hei) if offset is None else offset
        if img_last is None:
            img_last = imglt
            continue

        # ====================================================================
        # Start main logic
        # ===================================================================
        print_(f'black_frame_value={imglt.mean():.2f}')
        if imglt.mean() < black_frame_threshold:  # black frame
            scene_img_list.clear()
            scene_dispvis_list.clear()
            img_last = None
            continue

        if isnewscene(img_last, imglt):
            video_file_name = f"{outprefix}_{scene_idx}.mp4"
            vis_file_name = f"{visprefix}_{scene_idx}.mp4"

            if len(scene_img_list) < scene_least_frame_num:
                print(f"    However length {len(scene_img_list)} is too short, will skip and throw all frames", flush=True)
                scene_img_list.clear()
                scene_dispvis_list.clear()
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                wido, heio = scene_img_list[0].shape[:2]
                writer.open(video_file_name, fourcc, 30, (heio, wido), isColor=True)
                for img in scene_img_list:
                    writer.write(img)
                writer.release()

                wido, heio = scene_dispvis_list[0].shape[:2]
                writer.open(vis_file_name, fourcc, 30, (heio, wido), isColor=True)
                for img in scene_dispvis_list:
                    writer.write(img)
                writer.release()
                print(f"    {video_file_name} successfully written")
                scene_img_list.clear()
                scene_dispvis_list.clear()
                scene_idx += 1

        # check whether a bad frame.
        with torch.no_grad():
            flowlr = model(imglt, imgrt)
            flowrl = model(imgrt, imglt)

            occl = warp_flow(flowrl, flowlr, offset=offset) + flowlr
            occl = (torch.norm(occl, dim=1) < flow_bidirect_thresh).type(torch.float)
            noccl_pct = occl.sum() / occl.nelement()
            # condition1: occl_pct should not be too high
            print_(f"noccl_pct={noccl_pct}")
            if noccl_pct < nocc_pct_min:
                print(f"    bad frame due to large occlusion pct")
                img_last = None
                scene_img_list.clear()
                scene_dispvis_list.clear()
                continue

            vflowl = (flowlr[:, 1].abs() < vertical_disparity_torl).type(torch.float)
            vflow_occ_l = vflowl * occl
            vflow_pct = vflow_occ_l.sum() / occl.sum()
            # condition2: vflow_pct should be relatively low
            print_(f"vertical_disparity_pct={vflow_pct}")
            if vflow_pct < vflow_inlier_pct_min:  # bad frame detect
                print(f"    bad frame due to lots of vertical flow detected")
                img_last = None
                scene_img_list.clear()
                scene_dispvis_list.clear()
                continue

            # condition3: disparity shouldn't be too flat
            displ = flowlr[:, 0]
            vmax = torch.kthvalue(displ.reshape(-1), int(displ.nelement() * 0.98))[0]
            vmin = torch.kthvalue(displ.reshape(-1), int(displ.nelement() * 0.02))[0]
            disp_std = vmax - vmin
            # condition3: disparity std should not be too low
            print_(f"disparity_std={disp_std}")
            if disp_std < disparity_std_min:
                print(f"    bad frame due to small disparity std")
                img_last = None
                scene_img_list.clear()
                scene_dispvis_list.clear()
                continue

        # we won't correct the left-right relation even if it's reversed or shifted. We will rely on the
        # scale and shift-invariant disparity loss during training
        if vis_disp_max is None:
            vis_disp_min, vis_disp_max = torch.min(displ) - 30, torch.max(displ) + 30

        vis_disp = (displ - vis_disp_min) / (vis_disp_max - vis_disp_min)
        vis_disp = vis_disp * vflowl * occl
        vis_disp = (vis_disp * 255).type(torch.uint8)
        vis_disp = cv2.applyColorMap(vis_disp.cpu().numpy()[0], cv2.COLORMAP_HOT)
        scene_img_list.append(np.hstack([imgl, imgr]))
        scene_dispvis_list.append(np.hstack([vis_disp, img_vis]))
        img_last = imglt

    print(f"------------{videofile} finishes!!-----------------")


def process_list(video_list, cudaid):
    torch.cuda.set_device(cudaid)
    model = RAFTNet().eval().cuda()

    state_dict = torch.load(RAFT_path["sintel"], map_location=f'cuda:{cudaid}')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    for video_path in video_list:
        # decide special config based on name
        print(f"||{cudaid}||::process {video_path}", flush=True)
        processvideo(video_path, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_id', dest='work_id', type=int, help="index of num_worker")
    parser.add_argument('--num_worker', dest='num_worker', type=int, default=3, help="total number of Nodes used")
    args = parser.parse_args()

    dataset_root = "/home/lmaag/xgpu-scratch/mali_data"

    Source_midfix = "StereoVideoGood"
    Target_midfix = "StereoVideo_stage1v2"

    Source_root = os.path.join(dataset_root, Source_midfix)
    stereo_blur_videos = glob(os.path.join(Source_root, "StereoBlur", "*.mp4"))
    wsvd_videos = glob(os.path.join(Source_root, "WSVD", "*.mp4"))
    youtube_videos = glob(os.path.join(Source_root, "Youtube", "*.mp4"))
    scene_least_frame_num = 30
    scene_max_frame_num = 330
    scene_max_l1diff = 0.25
    black_frame_threshold = 0.15
    flow_bidirect_thresh = 1.5
    nocc_pct_min = 0.65  # smaller than this is outlier
    vertical_disparity_torl = 1.5
    vflow_inlier_pct_min = 0.7  # smaller than this is outlier
    disparity_std_min = 10

    num_thread = 10
    video_list = sorted(wsvd_videos + youtube_videos + stereo_blur_videos)
    video_list = video_list[args.work_id::args.num_worker]

    po = multiprocessing.Pool(num_thread)
    for i in range(num_thread):
        po.apply_async(process_list, [video_list[i::num_thread], i])

    po.close()
    po.join()
    print("------end------")
