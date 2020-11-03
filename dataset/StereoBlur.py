from pytube import YouTube
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
from . import RealEstate10K_root, is_DEBUG
from .colmap_wrapper import *


class StereoBlur(Dataset):
    """
    The dataset has 7711 test video and 71556 train video from youtube real estate video
    """
    def __init__(self, is_train=True, mode='resize', ptnum=2000):
        """
        model=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        if is_train:
            self.root = os.path.join(RealEstate10K_root, "train")
            self.trainstr = "train"
        else:
            self.root = os.path.join(RealEstate10K_root, "test")
            self.trainstr = "test"

        self.name = f"RealEstate10K_{self.trainstr}"
        self.file_list = glob(f"{self.root}/*.txt")
        print(f"RealEstate10K: find {len(self.file_list)} video files in {self.trainstr} set")

        self.tmp_root = os.path.join(os.path.dirname(self.root), f"{self.trainstr}tmp")
        if not os.path.exists(self.tmp_root):
            os.mkdir(self.tmp_root)

        self.mode = mode
        if mode == 'none':
            self.preprocess = ToTensor()
        elif mode == 'resize':
            self.preprocess = Compose([Resize([512, 512]),
                                       ToTensor()])
        elif mode == 'pad':
            self.preprocess = ToTensor()
            raise NotImplementedError
        elif mode == 'crop':
            self.preprocess = ToTensor()
            raise NotImplementedError

        self.ptnum = ptnum

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        pass

    def getitem(self, item):
        """
        Get item specified in item-th .txt file
        Will return None is something's wrong, pay special attention to this cast
        Parse Txt file -> Download video -> Fetch&save frame -> SfM
        Attantion for the None return, means something is wrong
        """
        pass


if __name__ == "__main__":
    example_file = "D:\\MSI_NB\\source\\data\\RealEstate10K\\train\\aaa1ef2a365d7781.txt"
    output_root = "D:\\MSI_NB\\source\\data\\RealEstate10K\\temp\\"
