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
from . import StereoVideo_root, is_DEBUG, OutputSize, StereoVideo_test_pct
from .colmap_wrapper import *
from .cv2disparity import compute_disparity_uncertainty
from .Augmenter import DataAugmenter
import sys
sys.path.append('..')
from models.mpi_utils import *


class StereoVideo_Base:
    def __init__(self, is_train, is_test=False):
        self.root = os.path.abspath(StereoVideo_root)
        self.trainstr = "not_specified"
        self.name = "StereoVideo3_Base"
        self.totensor = ToTensor()
        self._cur_file_base = ""

        video_list = sorted(glob(os.path.join(self.root, "videos", "*.mp4")))
        self.filebase_list = [self.getbasename(_p) for _p in video_list]

        test_split_file = self.gettestidx_path()
        if not os.path.exists(test_split_file):  # random generate test split if not exist
            test_split_idx = np.random.choice(len(self.filebase_list),
                                              int(len(self.filebase_list) * StereoVideo_test_pct) + 1, replace=False)
            test_base = [self.filebase_list[idx] for idx in test_split_idx]
            with open(test_split_file, 'w') as f:
                for base in test_base:
                    f.write(base + '\n')
        else:
            with open(self.gettestidx_path()) as f:
                lines = f.readlines()
            test_base = [self.getbasename(l_) for l_ in lines]

        if is_train:
            self.trainstr = "train"
            self.filebase_list = list(set(self.filebase_list) - set(test_base))
        else:
            self.trainstr = "test"
            self.filebase_list = test_base

        if is_test:
            self.filebase_list = [n_ for n_ in self.filebase_list if "StereoBlur" in n_]
        self.filebase_list.sort()

    @staticmethod
    def getbasename(path):
        return os.path.basename(path.strip('\n').replace('\\', '/')).split('.')[0]

    def getfullvideopath(self, base_name):
        return os.path.join(self.root, "videos", f"{base_name}.mp4")

    def getdisparitypath(self, base_name, isleft, frameidx):
        if isleft:
            return os.path.join(self.root, "disparities", base_name, "left", f"{frameidx:06d}.npy")
        else:
            return os.path.join(self.root, "disparities", base_name, "right", f"{frameidx:06d}.npy")

    def gettestidx_path(self):
        return os.path.join(self.root, "test_split.txt")


class StereoVideo_Img(Dataset, StereoVideo_Base):
    def __init__(self, is_train, mode='crop'):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
        model=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        super().__init__(is_train=is_train)
        self.name = f"StereoVideo_Img_{self.trainstr}"

        print(f"{self.name}: find {len(self.filebase_list)} video files in {self.trainstr} set")

        self.augmenter = DataAugmenter(OutputSize, mode=mode)

    def __len__(self):
        return len(self.filebase_list)

    def __getitem__(self, item):
        # try 3 times
        datas = None
        for i in range(3):
            try:
                datas = self.getitem(item)
            except Exception as e:
                print(e)
                datas = None
            if datas is not None:
                return datas

        # if still not working, randomly pick another idx untill success
        print(f"{self.name}.warning: cannot load {self.filebase_list[item]} after 3 tries")
        return datas

    def getitem(self, idx):
        return self.getitem_bybase(self.filebase_list[idx])

    def getitem_bybase(self, filebase):
        self._cur_file_base = filebase
        videofile = self.getfullvideopath(filebase)
        cap = cv2.VideoCapture(videofile)
        if not cap.isOpened():
            print(f"{self.name}: cannot open video file {filebase}.mp4")
            return None

        framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # random variables
        if self.trainstr == "train":
            idx = np.random.randint(0, framenum)
            retleft = (np.random.randint(2) == 0)
        else:
            idx = 1
            retleft = True
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = cap.read()
        cap.release()
        if not (ret and len(img) > 0):
            print(f"{self.name}: cannot read frame {idx} in {filebase}, which is said to have {framenum} frames")
            return None

        # read and split the left and right view
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hei, wid, _ = img.shape

        wid //= 2
        imgl, imgr = img[:, :wid], img[:, wid:]

        # augmentation
        self.augmenter.random_generate((hei, wid))
        disp_path = self.getdisparitypath(filebase, retleft, idx)

        if not os.path.exists(disp_path):
            print(f"Error::{self.name}:: choose to use npy but doesn't find {disp_path}")
            return None
        disp = np.load(disp_path, allow_pickle=True)
        mask = disp > np.finfo(np.half).min
        disp = np.where(mask, disp, 0).astype(np.float32)
        uncertainty = mask.astype(np.float32)

        imgl = self.augmenter.apply_img(imgl)
        imgr = self.augmenter.apply_img(imgr)
        disp = self.augmenter.apply_disparity(disp)
        uncertainty = self.augmenter.apply_img(uncertainty)
        isleft = torch.tensor([1.])

        if retleft:
            refimg = self.totensor(imgl)
            tarimg = self.totensor(imgr)
        else:
            refimg = self.totensor(imgr)
            tarimg = self.totensor(imgl)
            isleft = -isleft
        return refimg, tarimg, torch.tensor(disp), torch.tensor(uncertainty), isleft


class StereoVideo_Seq(Dataset, StereoVideo_Base):
    def __init__(self, is_train, mode='crop', seq_len=4, max_skip=7, test=False):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
        model=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        super().__init__(is_train=is_train, is_test=test)
        self.name = f"StereoVideo_Seq_{self.trainstr}"

        print(f"{self.name}: find {len(self.filebase_list)} video files in {self.trainstr} set")

        self.augmenter = DataAugmenter(OutputSize, mode=mode)
        self.sequence_length = seq_len
        self.maxskip_framenum = max(2, max_skip)  # 2 means no skip

    def __len__(self):
        return len(self.filebase_list)

    def __getitem__(self, item):
        # try 3 times
        datas = None
        for i in range(3):
            try:
                datas = self.getitem(item)
            except Exception as e:
                print(e)
                datas = None
            if datas is not None:
                return datas

        # if still not working, randomly pick another idx untill success
        print(f"{self.name}: cannot load {self.filebase_list[item]} after 3 tries")
        while datas is None:
            print(f"{self.name}: try fetch another data", flush=True)
            try:
                item = np.random.randint(len(self))
                datas = self.getitem(item)
            except Exception as e:
                print(e)
                datas = None
        return datas

    def getitem(self, idx):
        return self.getitem_bybase(self.filebase_list[idx])

    def getitem_bybase(self, filebase):
        self._cur_file_base = filebase
        videofile = self.getfullvideopath(filebase)
        cap = cv2.VideoCapture(videofile)
        if not cap.isOpened():
            print(f"{self.name}: cannot open video file {filebase}.mp4")
            return None

        framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if framenum < self.sequence_length + 1:
            return None

        # random variables
        # if self.trainstr == "train":
        max_skip = min(self.maxskip_framenum, 1 + (framenum - 1) // (self.sequence_length - 1))
        skipnum = np.random.randint(1, max_skip)
        framenum_wid = (self.sequence_length - 1) * skipnum + 1
        startid = np.random.randint(0, framenum - framenum_wid + 1)
        retleft = (np.random.randint(2) == 0)
        # else:
        #     skipnum = 1
        #     framenum_wid = self.sequence_length
        #     startid = 1
        #     retleft = False
        idxs = np.arange(startid, startid + framenum_wid, skipnum)
        hei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        self.augmenter.random_generate((hei, wid))

        imgls, imgrs, disps, uncertainty_maps = [], [], [], []

        shift = np.finfo(np.float32).min if retleft else np.finfo(np.float32).max
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, img = cap.read()
            if not (ret and len(img) > 0):
                print(f"{self.name}: cannot read frame {idx} in {filebase}, which is said to have {framenum} frames")
                return None

            # read and split the left and right view
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hei, wid, _ = img.shape

            wid //= 2
            imgl, imgr = img[:, :wid], img[:, wid:]

            # augmentation
            disp_path = self.getdisparitypath(filebase, retleft, idx)
            if not os.path.exists(disp_path):
                print(f"Error::{self.name}:: choose to use npy but doesn't find {disp_path}")
                return None
            disp = np.load(disp_path, allow_pickle=True)
            mask = disp > np.finfo(np.half).min
            if retleft:
                disp = np.where(mask, disp, np.finfo(np.float32).min).astype(np.float32)
                shift = max(self.augmenter.apply_disparity_scale(disp.max()), shift)
            else:
                disp = np.where(mask, disp, np.finfo(np.float32).max).astype(np.float32)
                shift = min(self.augmenter.apply_disparity_scale(disp.min()), shift)
            disp = np.where(mask, disp, 0).astype(np.float32)
            uncertainty = mask.astype(np.float32)

            imgl = self.augmenter.apply_img(imgl)
            imgr = self.augmenter.apply_img(imgr)
            disp = self.augmenter.apply_disparity(disp, interpolation='nearest')
            uncertainty = self.augmenter.apply_img(uncertainty, interpolation='nearest')

            # if uncertainty part is too little, return None
            if uncertainty.sum() < 0.2 * uncertainty.size:
                print(f"{self.name}:: uncertainty map too sparse, this data not available")
                return None

            imgls.append(self.totensor(imgl))
            imgrs.append(self.totensor(imgr))
            disps.append(torch.tensor(disp))
            uncertainty_maps.append(torch.tensor(uncertainty))

        cap.release()

        isleft = torch.tensor([1.])
        if retleft:
            refimg = torch.stack(imgls, dim=0)
            tarimg = torch.stack(imgrs, dim=0)
        else:
            refimg = torch.stack(imgrs, dim=0)
            tarimg = torch.stack(imgls, dim=0)
            isleft = -isleft
        disps = torch.stack(disps, dim=0)
        uncertainty_maps = torch.stack(uncertainty_maps, dim=0)

        return refimg, tarimg, disps, uncertainty_maps, torch.cat([isleft, torch.tensor([shift])])

