import cv2
import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mpi_utils import *

lastposx, lastposy = 0, 0
currentdx, currentdy = 0, 0
load_partial_raw = True


def click_callback(event, x, y, flags, param):
    global lastposx, lastposy, currentdx, currentdy
    if event == cv2.EVENT_LBUTTONDOWN:
        lastposx, lastposy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        currentdx = (x - lastposx) * 0.001
        currentdy = (y - lastposy) * 0.001


def square2alpha(x, layer, planenum):
    denorm = planenum - 1
    scale, thick, depth = layer
    n = denorm * scale * (x - depth + thick)
    n = torch.max(n, torch.zeros(1).type_as(n))
    n = torch.min(n, denorm * scale * thick)
    return n[1:] - n[:-1]


class MPIS:
    def __init__(self):
        self.raw_list = []
        self.hei, self.wid = 0, 0
        self.layernum = 0
        self.frameidx = 0
        self.depthes = None

    def __next__(self):
        self.frameidx = (self.frameidx + 1) % len(self.raw_list)
        return self.getmpi(self.frameidx)

    def getmpi(self, i):
        raise NotImplementedError


class MPRGBA(MPIS):
    def __init__(self):
        super().__init__()
        self.gnwid, self.gnhei = 8, 4

    def load(self, videoname):
        alphaname = videoname.replace(".mp4", "_alpha.mp4")
        caprgb = cv2.VideoCapture(os.path.join(videopath, videoname))
        capa = cv2.VideoCapture(os.path.join(videopath, alphaname))
        if not caprgb.isOpened() or not capa.isOpened():
            raise RuntimeError(f"cannot open video {videoname}")

        wid = int(caprgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        hei = int(caprgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.wid, self.hei = wid // self.gnwid, hei // self.gnhei
        self.layernum = self.gnwid * self.gnhei

        print(f"Loading mpis of {self.wid}x{self.hei}")
        frameidx = 0
        while True:
            print(f'\r{frameidx}', end='', flush=True)
            ret, rgb = caprgb.read()
            _, alpha = capa.read()
            if not ret:
                break
            frameidx += 1
            frame = np.concatenate([rgb, alpha[:, :, 0:1]], axis=-1)
            self.raw_list.append(torch.tensor(frame).cuda())
        print("\nSuccess")
        self.depthes = make_depths(self.layernum).cuda()

    def getmpi(self, i):
        rgba = self.raw_list[i].type(torch.float32) / 255
        mpi = rgba.reshape(self.gnhei, self.hei, self.gnwid, self.wid, 4) \
            .permute(0, 2, 4, 1, 3) \
            .reshape(1, self.layernum, 4, self.hei, self.wid)
        return mpi


class MPNet(MPIS):
    def __init__(self):
        super().__init__()
        self.layernum = planenum
        self.alphax = None

    def load(self, videoname):
        cap = cv2.VideoCapture(os.path.join(videopath, videoname))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {videoname}")

        wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.wid, self.hei = wid // 2, hei // 2

        print(f"Loading mpis of {self.wid}x{self.hei}")
        frameidx = 0
        while True:
            print(f'\r{frameidx}', end='', flush=True)
            ret, net = cap.read()
            if not ret:
                break
            frameidx += 1
            net = (torch.tensor(net).cuda().type(torch.float32) / 255).permute(2, 0, 1)
            layer1, layer2, fg, bg = net[:, :self.hei, :self.wid], \
                                     net[:, :self.hei, self.wid:], \
                                     net[:, self.hei:, :self.wid], \
                                     net[:, self.hei:, self.wid:]
            net = torch.stack([layer1, layer2, fg, bg]).contiguous()
            self.raw_list.append(net)
        print("\nSuccess")
        self.alphax = torch.linspace(0, 1, self.layernum).reshape(self.layernum, 1, 1).type_as(self.raw_list[0])
        self.depthes = make_depths(self.layernum).cuda()

    def getmpi(self, i):
        layer1, layer2, imfg, imbg = self.raw_list[i]
        alpha1 = square2alpha(self.alphax, layer1, self.layernum)
        alpha2 = square2alpha(self.alphax, layer2, self.layernum)
        alpha = alpha1 + alpha2
        alpha = torch.clamp(alpha, 0, 1)
        alpha = torch.cat([torch.ones([1, self.hei, self.wid]).type_as(alpha), alpha], dim=0)
        mpi = alpha2mpi(alpha.unsqueeze(0), imfg.unsqueeze(0), imbg.unsqueeze(0))
        return mpi


videopath = "./"

parser = argparse.ArgumentParser(description="visualize the mpi")
parser.add_argument("videoname")
parser.add_argument("--isnet", default="auto", help="choose among 'mpi', 'net', 'auto', default is auto"
                                                    "select the video type")
parser.add_argument("--planenum", default=32, help="the number of planes")
args = parser.parse_args()
videoname = args.videoname
planenum = args.planenum
isnet = True if args.isnet.lower() == "net" or (args.isnet.lower() == "auto" and videoname.endswith("_net.mp4")) \
    else False

if __name__ == "__main__":
    repos = MPNet() if isnet else MPRGBA()
    repos.load(videoname)
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
        [[repos.wid / 2, 0.0, repos.wid / 2],
         [0.0, repos.hei / 2, repos.hei / 2],
         [0.0, 0.0, 1.0]]
    ).type(torch.float32).cuda().unsqueeze(0)

    mode = "mpi"

    cv2.namedWindow("mpi")
    cv2.setMouseCallback("mpi", click_callback)
    update = True
    curmpi = None
    with torch.no_grad():
        while True:
            if update:
                curmpi = next(repos)
            if mode == "mpi":
                cosx, sinx, cosy, siny = np.cos(currentdx), np.sin(currentdx), np.cos(currentdy), np.sin(currentdy)
                target_pose = torch.tensor(
                    [[cosx, -sinx*siny, -sinx * cosy, currentdx],
                     [0.0, cosy, -siny, currentdy],
                     [sinx, cosx * siny, cosx * cosy, 0]]
                ).type(torch.float32).cuda().unsqueeze(0)
                view = render_newview(curmpi, source_pose, target_pose, intrin, intrin, repos.depthes)[0]

            elif mode == "disp":
                estimate_disparity_torch(curmpi.unsqueeze(0), repos.depthes)

            vis = (view * 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
            cv2.imshow("mpi", vis)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
            elif key == ord(' '):
                update = not update

