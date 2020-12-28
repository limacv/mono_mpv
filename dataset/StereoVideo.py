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
from . import StereoBlur_root, is_DEBUG, OutputSize
from . import StereoBlur_use_saved_disparity as use_Disp_npy
from .colmap_wrapper import *
from .cv2disparity import compute_disparity_uncertainty
from .Augmenter import DataAugmenter
import sys
sys.path.append('..')


def post_process_dispnpy(disp: np.ndarray):
    return cv2.resize(disp, (1280, 720))


class StereoVideo_Base:
    def __init__(self):
        self.root = os.path.abspath(StereoBlur_root)
        self.trainstr = "not_specified"
        self.name = f"StereoBlur_Base"
        self.totensor = ToTensor()
        self._cur_file_base = ""

    @staticmethod
    def getbasename(path):
        return os.path.basename(path.strip('\n').replace('\\', '/')).split('.')[0]

    def getfullvideopath(self, base_name):
        return os.path.join(self.root, self.trainstr, f"{base_name}.mp4")

    def getdisparitypath(self, base_name, isleft, frameidx):
        # todo: now that train & test set has save content, so disparity is the same. Please modify after I've done
        #   all the tests
        if isleft:
            return os.path.join(self.root, "train_disp", base_name, "left", f"{frameidx:05d}.npy")
        else:
            return os.path.join(self.root, "train_disp", base_name, "right", f"{frameidx:05d}.npy")


class StereoVideo_Img(Dataset, StereoVideo_Base):
    def __init__(self, is_train, mode='crop'):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
        model=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        super().__init__()
        if is_train:
            self.trainstr = "train"
        else:
            self.trainstr = "test"
        video_list = sorted(glob(os.path.join(self.root, self.trainstr, "*.mp4")))
        self.filebase_list = [self.getbasename(_p) for _p in video_list]

        self.name = f"StereoBlur_Img_{self.trainstr}"
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
        print(f"{self.name}: cannot load {self.filebase_list[item]} after 3 tries")
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
        if imgl.shape != imgr.shape:
            print(f"{self.name}: {filebase} frame {idx} left right view has different size, resizing to left")
            imgr = cv2.resize(imgr, (imgl.shape[1], imgl.shape[0]))

        # augmentation
        self.augmenter.random_generate((hei, wid))
        disp_path = self.getdisparitypath(filebase, retleft, idx)

        if os.path.exists(disp_path) and use_Disp_npy:
            disp = post_process_dispnpy(np.load(disp_path))
            uncertainty = np.ones_like(disp)
        elif use_Disp_npy:
            print(f"Warning {self.name}:: choose to use npy but doesn't find {disp_path}")
            return None
        else:
            disp, uncertainty = compute_disparity_uncertainty(imgl, imgr, retleft)

        imgl = self.augmenter.apply_img(imgl)
        imgr = self.augmenter.apply_img(imgr)
        disp = self.augmenter.apply_disparity(disp)
        uncertainty = self.augmenter.apply_img(uncertainty)

        # if uncertainty part is too little, return None
        if uncertainty.sum() < 0.2 * uncertainty.size:
            print(f"{self.name}:: uncertainty map too sparse, this data not available")
            return None

        if retleft:
            refimg = self.totensor(imgl)
            tarimg = self.totensor(imgr)
            isleft = torch.tensor([1.])
        else:
            refimg = self.totensor(imgr)
            tarimg = self.totensor(imgl)
            isleft = torch.tensor([-1.])
        return refimg, tarimg, torch.tensor(disp), torch.tensor(uncertainty), isleft


class StereoBlur_Seq(Dataset, StereoVideo_Base):
    def __init__(self, is_train, mode='crop', seq_len=4, max_skip=10):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
        model=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        super().__init__()
        if is_train:
            self.trainstr = "train"
        else:
            self.trainstr = "test"
        video_list = sorted(glob(os.path.join(self.root, self.trainstr, "*.mp4")))
        self.filebase_list = [self.getbasename(_p) for _p in video_list]

        self.name = f"StereoBlur_Seq_{self.trainstr}"
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
        if self.trainstr == "train":
            skipnum = np.random.randint(1, self.maxskip_framenum)
            framenum_wid = (self.sequence_length - 1) * skipnum + 1
            startid = np.random.randint(0, framenum - framenum_wid)
            retleft = (np.random.randint(2) == 0)
        else:
            skipnum = 1
            framenum_wid = self.sequence_length
            startid = 1
            retleft = True
        idxs = np.arange(startid, startid + framenum_wid, skipnum)
        hei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        self.augmenter.random_generate((hei, wid))

        imgls, imgrs, disps, uncertainty_maps = [], [], [], []
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
            if imgl.shape != imgr.shape:
                print(f"{self.name}: {filebase} frame {idx} left right view has different size, resizing to left")
                return None

            # augmentation
            disp_path = self.getdisparitypath(filebase, retleft, idx)
            if os.path.exists(disp_path) and use_Disp_npy:
                disp = np.load(disp_path, allow_pickle=True)
                disp = post_process_dispnpy(disp.astype(np.float32))
                uncertainty = np.ones_like(disp)
            elif use_Disp_npy:
                print(f"Warning {self.name}:: choose to use npy but doesn't find {disp_path}")
                return None
            else:
                disp, uncertainty = compute_disparity_uncertainty(imgl, imgr, retleft)

            imgl = self.augmenter.apply_img(imgl)
            imgr = self.augmenter.apply_img(imgr)
            disp = self.augmenter.apply_disparity(disp)
            uncertainty = self.augmenter.apply_img(uncertainty)

            # if uncertainty part is too little, return None
            if uncertainty.sum() < 0.2 * uncertainty.size:
                print(f"{self.name}:: uncertainty map too sparse, this data not available")
                return None

            imgls.append(self.totensor(imgl))
            imgrs.append(self.totensor(imgr))
            disps.append(torch.tensor(disp))
            uncertainty_maps.append(torch.tensor(uncertainty))
        cap.release()

        if retleft:
            refimg = torch.stack(imgls, dim=0)
            tarimg = torch.stack(imgrs, dim=0)
            isleft = torch.tensor([1.])
        else:
            refimg = torch.stack(imgrs, dim=0)
            tarimg = torch.stack(imgls, dim=0)
            isleft = torch.tensor([-1.])
        disps = torch.stack(disps, dim=0)
        uncertainty_maps = torch.stack(uncertainty_maps, dim=0)

        return refimg, tarimg, disps, uncertainty_maps, isleft

