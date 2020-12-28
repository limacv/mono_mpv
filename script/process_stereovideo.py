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


def mkdir_ifnotexist(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


StereoVideo_stage1_root = "Z:\\dataset\\StereoVideo_processed\\"
Output_midfix = "StereoVideoFinal"
stereo_blur_videos = glob(os.path.join(StereoVideo_root, "StereoBlur", "*.mp4"))
wsvd_videos = glob(os.path.join(StereoVideo_stage1_root, "WSVD", "*.mp4"))
youtube_videos = glob(os.path.join(StereoVideo_stage1_root, "Youtube", "*.mp4"))


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


def process_clip(videofile, model):
    outprefix = videofile.replace("StereoVideoGood", "StereoVideo_processed").split('.')[0]
    visprefix = videofile.replace("StereoVideoGood", "StereoVideo_visualization").split('.')[0]
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
        print(f"{frameidx}...", end='', flush=True)
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
        if wid >= 1920:
            imglt = cv2.resize(imgl, None, None, 0.5, 0.5)
            imgrt = cv2.resize(imgr, None, None, 0.5, 0.5)
            hei, wid = imglt.shape[:2]
        else:
            imglt, imgrt = imgl, imgr
        img_vis = imglt
        imglt, imgrt = totensor(cv2.cvtColor(imglt, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda(), \
                       totensor(cv2.cvtColor(imgrt, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
        offset = prepareoffset(wid, hei) if offset is None else offset
        if img_last is None:
            img_last = imglt
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

        print_(f'black_frame_value={imglt.mean():.2f}')
        if imglt.mean() < black_frame_threshold:  # black frame
            scene_img_list.clear()
            scene_dispvis_list.clear()
            img_last = None
            continue

        # check whether a bad frame.
        with torch.no_grad():
            flowlr = model(imglt, imgrt)
            vflowl = (flowlr[:, 1].abs() < vertical_disparity_torl).type(torch.float)
            vflow_pct = 1 - vflowl.sum() / vflowl.nelement()
            # condition1: vflow_pct should be relatively low
            print_(f"vertical_disparity_pct={vflow_pct}")
            if vflow_pct > vertical_disparity_pct_max:  # bad frame detect
                print(f"    bad frame due to lots of vertical flow detected")
                img_last = None
                scene_img_list.clear()
                scene_dispvis_list.clear()
                continue

            flowrl = model(imgrt, imglt)
            occl = warp_flow(flowrl, flowlr, offset=offset) + flowlr
            occl = (torch.norm(occl, dim=1) < flow_bidirect_thresh).type(torch.float)
            occl_pct = 1 - occl.sum() / occl.nelement()
            # condition2: occl_pct should not be too high
            print_(f"occl_pct={occl_pct}")
            if occl_pct > occ_pct_max:
                print(f"    bad frame due to large occlusion pct")
                img_last = None
                scene_img_list.clear()
                scene_dispvis_list.clear()
                continue

            # condition3: disparity shouldn't be too flat
            displ = flowlr[:, 0]
            disp_std = torch.std(displ)
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


def processvideos(video_list, cudaid):
    torch.cuda.set_device(cudaid)
    model = RAFTNet().eval().cuda()

    state_dict = torch.load(RAFT_path["sintel"], map_location=f'cuda:{cudaid}')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    totensor = ToTensor()

    for video_path in video_list:
        process_clip()


if __name__ == "__main__":
    processvideos(youtube_videos[:,:,2])

    po = multiprocessing.Pool(10)
    for i, video_file in enumerate(wsvd_videos):
        po.apply_async(processvideos, [video_file, i % 10])

    po.close()
    po.join()
    print("------end------")
