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
from . import WSVD_root
from . import OutputSize as outSize
from .Augmenter import DataAugmenter
from .cv2disparity import compute_disparity_uncertainty


class WSVD_Img(Dataset):
    def __init__(self, is_train):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
        model=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        if is_train:
            self.trainstr = "train"
        else:
            self.trainstr = "test"
        self.root = os.path.abspath(WSVD_root)
        video_list = sorted(glob(os.path.join(self.root, self.trainstr, "*.mp4")))
        self.filebase_list = [self.getbasename(_p) for _p in video_list]

        self.name = f"WSVD_{self.trainstr}"
        print(f"{self.name}: find {len(self.filebase_list)} video files in {self.trainstr} set")

        self.totensor = ToTensor()
        self._cur_file_base = ""

    @staticmethod
    def getbasename(path):
        return os.path.basename(path.strip('\n').replace('\\', '/')).split('.')[0]

    def getfullvideopath(self, base_name):
        return os.path.join(self.root, self.trainstr, f"{base_name}.mp4")

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
        if self.trainstr == "train":
            idx = np.random.randint(0, framenum)
        else:
            idx = 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = cap.read()
        cap.release()
        if not (ret and len(img) > 0):
            print(f"{self.name}: cannot read frame {idx} in {filebase}, which is said to have {framenum} frames")
            return None

        # split the left and right view
        hei, wid, _ = img.shape
        if wid > hei * 4:  # might be something wrong with ratio
            img = cv2.resize(img, None, None, 0.5, 1)
            hei, wid, _ = img.shape

        wid //= 2
        imgl, imgr = img[:, :wid], img[:, wid:]
        if imgl.shape != imgr.shape:
            print(f"{self.name}: {filebase} frame {idx} left right view has different size, resizing to left")
            imgr = cv2.resize(imgr, (imgl.shape[1], imgl.shape[0]))

        return compute_disparity_uncertainty(imgl, imgr, resize=True)
