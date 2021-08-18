import cv2
import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mpi_utils import *
from models.flow_utils import forward_scatter_fusebatch

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


def ldiad2mpia(ldiad):
    plane_disps = torch.reciprocal(make_depths(planenum)).reshape(1, 1, planenum, 1, 1).type_as(ldiad)
    disp_min, disp_max = 1 / default_d_far, 1 / default_d_near

    d = (ldiad[:, :, 1:2] - plane_disps) * (planenum - 1.) / (disp_max - disp_min)
    d = - torch.abs(torch.clamp(d, -1, 1)) + 1.
    mpialpha = - torch.pow(- ldiad[:, :, 0:1] + 1., d).prod(dim=1) + 1.
    mpialpha = torch.clamp(mpialpha, 0, 1)
    mpialpha[:, 0] = 1.  # the background should be all 1

    d = d + 0.0001
    blend_weight = d / d.sum(dim=1, keepdim=True)

    return mpialpha, blend_weight


def ldi2mpi(ldi):
    mpia, bw = ldiad2mpia(ldi[:, :, -2:])
    rgbs = ldi[:, :, :3]
    mpirgb = (rgbs.unsqueeze(2) * bw.unsqueeze(3)).sum(dim=1)
    mpi = torch.cat([mpirgb, mpia.unsqueeze(2)], dim=2)
    return mpi


class MPRGBA:
    def __init__(self):
        self.layernum = planenum
        self.depthes = make_depths(self.layernum).cuda()
        self.zeros = None
        self.raw_list = []
        self.hei, self.wid = 0, 0
        self.coord = None
        self.frameidx = 0

    def __next__(self):
        self.frameidx = (self.frameidx + 1) % len(self.raw_list)
        return self.getmpi(self.frameidx)

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
            fg, bg, dfg, dbg, afg, abg = net[:, :self.hei, :self.wid], net[:, :self.hei, self.wid:], \
                                         net[:, self.hei:2*self.hei, :self.wid], net[:, self.hei:2*self.hei, self.wid:], \
                                         net[:, 2*self.hei:, :self.wid], net[:, 2*self.hei:, self.wid:],
            disp = torch.cat([dbg[0:1], dfg[0:1]], dim=0)[None, :, None, ...]
            disp = dilate(dilate(disp.squeeze(0), 3, 2)).unsqueeze(0)
            alpha = torch.cat([abg[0:1], afg[0:1]], dim=0)[None, :, None, ...]  # 1, 2, 1, H, W
            rgb = torch.stack([bg, fg])[None, ...]
            ldi = torch.cat([rgb, alpha, disp], dim=2)

            self.raw_list.append(ldi.contiguous().cuda())

        print("\nSuccess")
        self.zeros = torch.zeros_like(fg[0:1].unsqueeze(0)).cuda()

    def getmpi(self, i):
        return ldi2mpi(self.raw_list[i])


videopath = "./"

parser = argparse.ArgumentParser(description="visualize the mpi")
parser.add_argument("videoname")
parser.add_argument("--isnet", default="auto", help="choose among 'mpi', 'net', 'auto', default is auto"
                                                    "select the video type")
parser.add_argument("--planenum", default=32, type=int, help="the number of planes")
args = parser.parse_args()
videoname = args.videoname
planenum = args.planenum
plane_disps = torch.reciprocal(make_depths(planenum)).reshape(1, 1, planenum, 1, 1).cuda()
plane_distance = float(plane_disps[0, 0, 2, 0, 0] - plane_disps[0, 0, 1, 0, 0])
isnet = True if args.isnet.lower() == "net" or (args.isnet.lower() == "auto" and '_net' in videoname) \
    else False

if __name__ == "__main__":
    repos = MPRGBA()
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
                    [[cosx, -sinx * siny, -sinx * cosy, currentdx],
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
