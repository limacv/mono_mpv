import cv2
import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mpi_utils import *

lastposx, lastposy = 0, 0
currentdx, currentdy = 0, 0


def click_callback(event, x, y, flags, param):
    global lastposx, lastposy, currentdx, currentdy
    if event == cv2.EVENT_LBUTTONDOWN:
        lastposx, lastposy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        currentdx = (x - lastposx) * 0.001
        currentdy = (y - lastposy) * 0.001


videopath = "./"

parser = argparse.ArgumentParser(description="visualize the mpi")
parser.add_argument("videoname")
args = parser.parse_args()
videoname = args.videoname

if __name__ == "__main__":
    alphaname = videoname.replace(".mp4", "_alpha.mp4")
    caprgb = cv2.VideoCapture(os.path.join(videopath, videoname))
    capa = cv2.VideoCapture(os.path.join(videopath, alphaname))
    if not caprgb.isOpened() or not capa.isOpened():
        raise RuntimeError(f"cannot open video {videoname}")

    gnwid, gnhei = 8, 3
    inwid = int(caprgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    inhei = int(caprgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    imwid, imhei = inwid // gnwid, inhei // gnhei
    layernum = gnwid * gnhei
    depthes = make_depths(layernum).cuda()
    target_pose = torch.tensor(
        [[1.0, 0.0, 0.0, 0],
         [0.0, 1.0, 0.0, 0],
         [0.0, 0.0, 1.0, 0]]
    ).type(torch.float32).cuda().unsqueeze(0)
    source_pose = torch.tensor(
        [[1.0, 0.0, 0.0, 0],
         [0.0, 1.0, 0.0, 0],
         [0.0, 0.0, 1.0, 0]]
    ).type(torch.float32).cuda().unsqueeze(0)
    intrin = torch.tensor(
        [[imwid / 2, 0.0, imwid / 2],
         [0.0, imhei / 2, imhei / 2],
         [0.0, 0.0, 1.0]]
    ).type(torch.float32).cuda().unsqueeze(0)

    mpilist = []
    mode = "mpi"

    frames = []
    print(f"Loading mpis of {imwid}x{imhei}")
    frameidx = 0
    while True:
        print(f'\r{frameidx}', end='')
        ret, rgb = caprgb.read()
        _, alpha = capa.read()
        if not ret:
            break
        frameidx += 1
        frame = np.concatenate([rgb, alpha[:, :, 0:1]], axis=-1)
        frames.append(torch.tensor(frame).cuda())

    cv2.namedWindow("mpi")
    cv2.setMouseCallback("mpi", click_callback)
    update = True
    with torch.no_grad():
        while True:
            for i in range(len(frames)):
                if update:
                    rgba = torch.tensor(frames[i]).cuda().type(torch.float32) / 255
                    mpi = rgba.reshape(gnhei, imhei, gnwid, imwid, 4)\
                        .permute(0, 2, 4, 1, 3)\
                        .reshape(gnhei * gnwid, 4, imhei, imwid)
                if mode == "mpi":
                    cosx, sinx, cosy, siny = np.cos(currentdx), np.sin(currentdx), np.cos(currentdy), np.sin(currentdy)
                    target_pose = torch.tensor(
                        [[cosx, -sinx*siny, -sinx * cosy, currentdx],
                         [0.0, cosy, -siny, currentdy],
                         [sinx, cosx * siny, cosx * cosy, 0]]
                    ).type(torch.float32).cuda().unsqueeze(0)
                    view = render_newview(mpi.unsqueeze(0), source_pose, target_pose, intrin, depthes)[0]

                elif mode == "disp":
                    estimate_disparity_torch(mpi.unsqueeze(0), depthes)

                vis = (view * 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
                cv2.imshow("mpi", vis)
                key = cv2.waitKey(1)
                if key == 27:
                    exit(0)
                elif key == ord(' '):
                    update = not update

