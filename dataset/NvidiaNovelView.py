import torch
from torch.nn import Sequential
import torch.nn.functional as torchf
import numpy as np
from io import BytesIO
from torchvision.transforms import ToTensor, Resize, Compose
import imageio
from PIL import Image as PImage
from torch.utils.data import Dataset
import cv2
import os
from glob import glob
from . import NvidiaNovelView_root, is_DEBUG
from .Augmenter import DataAugmenter, DataAdaptor
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NvidiaNovelView:
    def __init__(self, resolution=(540, 960), max_baseline=4, **kwargs):
        """
        resolution: resize output to the resolution
        """
        self.root = os.path.abspath(NvidiaNovelView_root)
        self.name = "Nvidia Novel View Synthesis Dataset"
        self.totensor = ToTensor()

        item_list = sorted(glob(os.path.join(self.root, "*")))
        self.scene_list = [self.getbasename(_p) for _p in item_list]
        self.adaptor = DataAdaptor(resolution)
        self.max_baseline = max_baseline
        print(f"{self.name}\n"
              f"total {len(self)} videos/scenes\n"
              f"resolution={resolution}, max_baseline={max_baseline}")

    @staticmethod
    def getbasename(path):
        return os.path.basename(path.strip('\n').replace('\\', '/'))

    def getinputpath(self, base_name, timei):
        assert 0 < timei <= 12
        path = os.path.join(self.root, base_name, "input_images", f"cam{timei:02d}.jpg")
        assert os.path.exists(path), f"{self.name}::cannot find {path}"
        return path

    def getdepthgtpath(self, base_name, timei):
        assert 0 < timei <= 12
        path = os.path.join(self.root, base_name, "depth_GT", f"cam{timei:02d}.npy")
        assert os.path.exists(path), f"{self.name}::cannot find {path}"
        return path

    def getposegtpath(self, base_name, cami):
        assert 0 < cami <= 12
        path = os.path.join(self.root, base_name, "calibration", f"cam{cami:02d}")
        assert os.path.exists(path), f"{self.name}::cannot find {path}"
        return path
    
    def getviewgtpath(self, base_name, cami, timei):
        assert 0 < cami <= 12 and 0 < timei <= 12
        path = os.path.join(self.root, base_name, "multiview_GT", f"{timei:08d}", f"cam{cami:02d}.jpg")
        assert os.path.exists(path), f"{self.name}::cannot find {path}"
        return path

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, item):
        basename = self.scene_list[item]
        print(f"{item}/{len(self)}::{basename}...")
        return self.getitem_bybase(basename)

    def getitem_bybase(self, base):
        ret = {
            "scene_name": base,
            "in_imgs": [],
            "in_poses": [],
            "gt_poses": [],
            "gt_depth": [],
            "gt_imgs": []
        }
        extrins, intrins, centers = [], [], []
        # read all the pose info
        for camidx in range(1, 13):
            with open(os.path.join(self.getposegtpath(base, camidx), "extrinsic.txt")) as f:
                lines = f.readlines()
                c = list(map(float, lines[0].strip('\n').split(' ')))
                c = np.array(c).reshape(3, 1)
                r = lines[1].strip('\n') + ' ' + lines[2].strip('\n') + ' ' + lines[3].strip('\n')
                r = list(map(float, r.split(' ')))
                r = np.array(r).reshape(3, 3)
                extrin = np.hstack([r, -r @ c])
                extrins.append(extrin)
                centers.append(c)
            with open(os.path.join(self.getposegtpath(base, camidx), "intrinsic.txt")) as f:
                lines = f.readlines()
                m = lines[0].strip('\n') + ' ' + lines[1].strip('\n') + ' ' + lines[2].strip('\n')
                m = list(map(float, m.split(' ')))
                intrin = np.array(m).reshape(3, 3)
                intrins.append(intrin)

        for frameidx in range(1, 13):
            # input images
            img = cv2.imread(self.getinputpath(base, frameidx))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hei, wid, _ = img.shape
            self.adaptor.random_generate((hei, wid))
            img = self.adaptor.apply_img(img)
            ret["in_imgs"].append(self.totensor(img))

            # depth
            depth = np.load(self.getdepthgtpath(base, frameidx))
            depth = self.adaptor.apply_img(depth, interpolation='nearest')
            ret["gt_depth"].append(torch.tensor(depth).type(torch.float32))

            # novel views
            intrin = torch.tensor(self.adaptor.apply_intrin(intrins[frameidx - 1])).type(torch.float32)
            extrin = torch.tensor(extrins[frameidx - 1]).type(torch.float32)
            ret["in_poses"].append((extrin, intrin))
            ret["gt_imgs"].append([])
            ret["gt_poses"].append([])
            for viewidx in range(1, 13):
                if viewidx == frameidx:
                    continue
                distance = np.linalg.norm(centers[viewidx - 1] - centers[frameidx - 1])
                if distance > self.max_baseline:
                    continue
                gtimg = cv2.imread(self.getviewgtpath(base, viewidx, frameidx))
                gtimg = cv2.cvtColor(gtimg, cv2.COLOR_BGR2RGB)
                gtimg = self.adaptor.apply_img(gtimg)
                ret["gt_imgs"][-1].append(self.totensor(gtimg))
                intrin = torch.tensor(self.adaptor.apply_intrin(intrins[viewidx - 1])).type(torch.float32)
                extrin = torch.tensor(extrins[viewidx - 1]).type(torch.float32)
                ret["gt_poses"][-1].append((extrin, intrin))

            if len(ret["gt_imgs"][-1]) == 0:
                print(f"{self.name}:: Warning! Scene={base} "
                      f"time={frameidx} has no GT view with max_baseline={self.max_baseline}")

        return ret
