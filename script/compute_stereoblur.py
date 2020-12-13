import sys
sys.path.append('../')
from models.LEAStereo_network import LEAStereo
from dataset import StereoBlur_root
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


Opt = namedtuple("opt", ["cell_arch_fea", "cell_arch_mat", "net_arch_fea", "net_arch_mat",
                         "fea_block_multiplier", "fea_filter_multiplier", "fea_num_layers",
                         "fea_step", "mat_block_multiplier", "mat_filter_multiplier", "mat_num_layers",
                         "mat_step", "maxdisp"])
opt = Opt(cell_arch_fea="../weights/LEAStereo/architecture/feature_genotype.npy",
          cell_arch_mat="../weights/LEAStereo/architecture/matching_genotype.npy",
          net_arch_fea="../weights/LEAStereo/architecture/feature_network_path.npy",
          net_arch_mat="../weights/LEAStereo/architecture/matching_network_path.npy",
          fea_block_multiplier=4,
          fea_filter_multiplier=8,
          fea_num_layers=6,
          fea_step=3,
          mat_block_multiplier=4,
          mat_filter_multiplier=8,
          mat_num_layers=12,
          mat_step=3,
          maxdisp=192)

videos = glob(os.path.join(StereoBlur_root, "train", "*.mp4"))


def processvideo(videofile, cudaid):
    torch.cuda.set_device(cudaid)
    model = LEAStereo(opt).cuda()
    checkpoint = torch.load("../weights/LEAStereo/middleburry.pth", map_location=f"cuda:{cudaid}")
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    totensor = ToTensor()

    outroot = videofile.replace("train", "train_disp").split('.')[0]
    visoutpath = outroot
    leftout = os.path.join(outroot, "left")
    rightout = os.path.join(outroot, "right")

    mkdir_ifnotexist(leftout)
    mkdir_ifnotexist(rightout)

    cap = cv2.VideoCapture(videofile)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frameidx in range(framecount):
        print(f"\r{frameidx}", end='')
        ret, img = cap.read()
        if not ret:
            print(f"{videofile} is not normal regarding frame count")
            break
        wid = img.shape[1]
        imgl, imgr = img[:, :wid//2], img[:, wid//2:]
        imgl = cv2.resize(imgl, (1248, 720))
        imgr = cv2.resize(imgr, (1248, 720))
        hei, wid, cnl = imgl.shape
        imglt, imgrt = totensor(imgl).unsqueeze(0).cuda(), totensor(imgr).unsqueeze(0).cuda()

        # left view
        with torch.no_grad():
            disp = model(imglt, imgrt)
        dispnp = disp[0].cpu().numpy().astype(np.float16)
        np.save(os.path.join(leftout, f"{frameidx:05d}.npy"), dispnp)

        # right view
        imglt = torch.flip(imglt, dims=[-1])
        imgrt = torch.flip(imgrt, dims=[-1])
        with torch.no_grad():
            disp = model(imgrt, imglt)
            disp = torch.flip(disp, dims=[-1])
        dispnp = disp[0].cpu().numpy().astype(np.float16)
        np.save(os.path.join(rightout, f"{frameidx:05d}.npy"), dispnp)

        dispvis = (dispnp / (120 / 255)).astype(np.uint8)
        dispvis = cv2.applyColorMap(dispvis, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(visoutpath, f"{frameidx:05d}.jpg"), dispvis)


if __name__ == "__main__":
    videos = glob(os.path.join(StereoBlur_root, "train", "*.mp4"))
    # for i, videofile in enumerate(videos):
    #     processvideo(videofile, 0)
    po = multiprocessing.Pool(8)
    for i, video_file in enumerate(videos):
        po.apply_async(processvideo, [video_file, i % 8])

    po.close()
    po.join()
    print("------end------")