import cv2
import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mpi_utils import *

lastposx, lastposy = 0, 0
currentdx, currentdy, currentdz = 0, 0, 0
focal = 1
disp_min, disp_max = 1 / default_d_far, 1 / default_d_near
load_partial_raw = True


def click_callback(event, x, y, flags, param):
    global lastposx, lastposy, currentdx, currentdy, currentdz, focal
    if event == cv2.EVENT_LBUTTONDOWN:
        lastposx, lastposy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        currentdx = (x - lastposx) * 0.001
        currentdy = (y - lastposy) * 0.001
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            focal += 0.04
            currentdz += 0.04
        else:
            focal -= 0.04
            currentdz -= 0.04


def square2alpha(x, layer, planenum):
    denorm = planenum - 1
    scale, thick, depth = layer
    n = denorm * scale * (x - depth + thick)
    n = torch.max(n, torch.zeros(1).type_as(n))
    n = torch.min(n, denorm * scale * thick)
    return n[1:] - n[:-1]


def nsets2alpha(x, ds, th, sig):
    th = torch.min(ds - 1. / default_d_far, th)
    t = x - ds + th
    t = torch.relu(t)
    t = torch.min(t, th)
    dt = t[:, :, 1:] - t[:, :, :-1]

    expo = -(dt * sig).sum(dim=1)
    return -torch.exp(expo * 20) + 1


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
        self.x = torch.reciprocal(make_depths(planenum)).reshape(1, 1, planenum, 1, 1).cuda()
        self.dx = float(self.x[0, 0, 2, 0, 0] - self.x[0, 0, 1, 0, 0])
        self.depthes = make_depths(self.layernum)[1:].cuda()
        self.zeros = None

    def load(self, videoname):
        cap = cv2.VideoCapture(os.path.join(videopath, videoname))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {videoname}")

        wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.wid, self.hei = wid // 2, hei // 3

        print(f"Loading mpis of {self.wid}x{self.hei}")
        frameidx = 0
        while True:
            print(f'\r{frameidx}', end='', flush=True)
            ret, net = cap.read()
            if not ret:
                break
            frameidx += 1
            net = (torch.tensor(net).type(torch.float32) / 255).permute(2, 0, 1)
            fg, bg, dfg, dbg, tfg, tbg = net[:, :self.hei, :self.wid], net[:, :self.hei, self.wid:], \
                                         net[:, self.hei:2*self.hei, :self.wid], net[:, self.hei:2*self.hei, self.wid:], \
                                         net[:, 2*self.hei:, :self.wid], net[:, 2*self.hei:, self.wid:],
            disp = torch.cat([dfg[0:1], dbg[0:1]], dim=0).unsqueeze(0)
            disp = disp * (disp_max - disp_min) + disp_min
            thick = torch.cat([tfg[0:1] / 10, tbg[0:1]], dim=0).unsqueeze(0)
            self.raw_list.append((disp.contiguous().cuda(), thick.contiguous().cuda(),
                                  fg.unsqueeze(0).contiguous().cuda(), bg.unsqueeze(0).contiguous().cuda()))
        print("\nSuccess")
        self.zeros = torch.zeros_like(tfg[0:1].unsqueeze(0)).cuda()

    def getmpi(self, i):
        disp, thick, imfg, imbg = self.raw_list[i]

        alpha = nsets2alpha(self.x,
                            disp.unsqueeze(2),
                            thick.unsqueeze(2) * 2,
                            5)
        alpha = dilate(erode(alpha))
        # compute blend weight
        fg, bg = disp.split(1, dim=1)

        tfg = thick[:, 0:1]
        # turning = (fg + bg) / 2.
        # scaling = - torch.exp(-tfg * 100) + 1
        turning = (dilate(dilate(fg)) + bg) / 2.
        scaling = - torch.exp(-dilate(dilate(tfg)) * 100) + 1
        blend_weight = torch.relu(torch.min(turning, self.x[:, 0, 1:]) - self.x[:, 0, :-1]) / self.dx
        blend_weight = - blend_weight * scaling + 1

        mpi = alpha2mpi(alpha, imfg, imbg, blend_weight=blend_weight)
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
isnet = True if args.isnet.lower() == "net" or (args.isnet.lower() == "auto" and '_net' in videoname) \
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
    tarintrin = torch.tensor(
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
                     [sinx, cosx * siny, cosx * cosy, currentdz]]
                ).type(torch.float32).cuda().unsqueeze(0)
                tarintrin[0, 0, 0] = repos.wid / 2 * focal
                tarintrin[0, 1, 1] = repos.hei / 2 * focal
                view = render_newview(curmpi, source_pose, target_pose, intrin, tarintrin, repos.depthes)[0]

            elif mode == "disp":
                estimate_disparity_torch(curmpi.unsqueeze(0), repos.depthes)

            vis = (view * 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
            cv2.imshow("mpi", vis)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
            elif key == ord(' '):
                update = not update
            elif key == ord('p'):
                print("target pose:")
                print(target_pose)
                print("focal length:")
                print(focal)

