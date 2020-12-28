import torch
import numpy as np
import cv2


class DataAugmenter:
    """
    __init__() -> generate() -> apply_*()
    """

    def __init__(self, outshape, mode="crop", ratio_tol=1.1, resize_tol=1.):
        """
        outshape: (height, weight)
        mode="none", "crop", "resize", *"pad"   * = not implemented
            when use crop:  in_shape -> crop_shape -> out_shape
        """
        self.mode = mode.lower()
        self.outhei, self.outwid = outshape
        self.ratio_tol = ratio_tol if ratio_tol > 1 else 1. / ratio_tol  # ratio = wid / hei
        self.resize_tol = resize_tol if resize_tol > 1 else 1. / resize_tol  # max resize factor after croping

        # maintain current status
        self.cur_inhei, self.cur_inwid = 0, 0
        self.cur_crop_ratio = 1.
        self.cur_crop_left, self.cur_crop_top = 0, 0
        self.cur_crop_wid, self.cur_crop_hei = 0, 0
        self.cur_resize = 1.

    def random_generate(self, in_shape):
        """
        randomly generate new augmentation
        call this every time before augmentation
        """
        self.cur_inhei, self.cur_inwid = in_shape
        if self.mode in ["none", "resize"]:
            return
        elif self.mode == "crop":
            self.cur_crop_ratio = np.random.uniform(1. / self.ratio_tol, self.ratio_tol) * self.outwid / self.outhei

            # decide crop window
            if self.cur_crop_ratio > self.cur_inwid / self.cur_inhei:
                min_rsz = self.outwid / self.cur_inwid
                wid_major = True
            else:
                min_rsz = self.outhei / self.cur_inhei
                wid_major = False
            if min_rsz > self.resize_tol:
                print("DataAugmenter: warning! minimum resize factor exceed the tollerant, input is too small")
                print(f"    If you see this warning often, please adjust resize_tollerant to {min_rsz}")
                # self.resize_tol = min_rsz
            self.cur_resize = np.random.uniform(min_rsz, self.resize_tol)
            if wid_major:  # min for security reason
                self.cur_crop_wid = min(int(self.outwid / self.cur_resize), self.cur_inwid)
                self.cur_crop_hei = min(int(self.cur_crop_wid / self.cur_crop_ratio), self.cur_inhei)
            else:
                self.cur_crop_hei = min(int(self.outhei / self.cur_resize), self.cur_inhei)
                self.cur_crop_wid = min(int(self.cur_crop_hei * self.cur_crop_ratio), self.cur_inwid)
            self.cur_crop_top = np.random.randint(self.cur_inhei - self.cur_crop_hei + 1)
            self.cur_crop_left = np.random.randint(self.cur_inwid - self.cur_crop_wid + 1)
        else:
            raise NotImplementedError(f"DataAugmenter::{self.mode} not implemented")
        return

    def apply_pts(self, ptxy: np.array, ptz: np.array):
        """
        ptxy of size (numpt, 2), with axis=-1 being (x_im, y_im)
        ptz size (numpt, )
        return normalized points position
        """
        if self.mode == "none":
            pass
        elif self.mode == "crop":
            ptxy -= np.array([self.cur_crop_left, self.cur_crop_top]).astype(ptxy.dtype).reshape(1, 2)
            remain = (ptxy[:, 0] > 0) \
                     & (ptxy[:, 0] < self.cur_crop_wid - 1) \
                     & (ptxy[:, 1] > 0) \
                     & (ptxy[:, 1] < self.cur_crop_hei - 1)
            ptxy = ptxy[remain]
            ptz = ptz[remain]
            ptxy = ptxy * np.array([2 / self.cur_crop_wid, 2 / self.cur_crop_hei]).astype(ptxy.dtype) - 1.
        elif self.mode == "resize":
            ptxy = ptxy * np.array([2 / (self.cur_inwid - 1), 2 / (self.cur_inhei - 1)]).astype(ptxy.dtype) - 1.
        else:
            raise NotImplementedError(f"DataAugmenter::{self.mode} not implemented")
        if len(ptxy) < 100:
            print(f"DataAugmenter: warning! after filter the points, only {len(ptxy)} pts left")
            print(f"    If you see this warning often, please add fitering to the dataset")
        return ptxy, ptz

    def apply_intrin(self, intrin: np.array):
        if self.mode == "none":
            return intrin
        elif self.mode == "crop":
            fx, fy = self.outwid / self.cur_crop_wid, self.outhei / self.cur_crop_hei
            intrin_calib = np.array([fx, 0, - fx * self.cur_crop_left,
                                     0, fy, - fy * self.cur_crop_top,
                                     0, 0, 1], dtype=np.float32).reshape(3, 3)
        elif self.mode == "resize":
            intrin_calib = np.array([self.outwid / self.cur_inwid, 0, 0,
                                     0, self.outhei / self.cur_inhei, 0,
                                     0, 0, 1], dtype=np.float32).reshape(3, 3)
        else:
            raise NotImplementedError(f"DataAugmenter::{self.mode} not implemented")
        return intrin_calib @ intrin

    def crop(self, img: np.array):
        return img[self.cur_crop_top:
                           self.cur_crop_top + self.cur_crop_hei, self.cur_crop_left:
                                                                  self.cur_crop_left + self.cur_crop_wid]

    def apply_img(self, img: np.array):
        if self.mode == "none":
            return img
        elif self.mode == "crop":
            img_crop = self.crop(img)
            imgout = cv2.resize(img_crop, (self.outwid, self.outhei), interpolation=cv2.INTER_AREA)
            return imgout
        elif self.mode == "resize":
            imgout = cv2.resize(img, (self.outwid, self.outhei), interpolation=cv2.INTER_AREA)
            return imgout
        else:
            raise NotImplementedError(f"DataAugmenter::{self.mode} not implemented")

    def apply_disparity(self, disp: np.array):
        if self.mode == "none":
            return disp
        elif self.mode == "crop":
            disp_crop = self.crop(disp)
            dispout = cv2.resize(disp_crop, (self.outwid, self.outhei), interpolation=cv2.INTER_AREA)
            # scale the disp value
            dispout *= (self.outwid / self.cur_crop_wid)
            return dispout
        elif self.mode == "resize":
            dispout = cv2.resize(disp, (self.outwid, self.outhei), interpolation=cv2.INTER_AREA)
            return dispout
        else:
            raise NotImplementedError(f"DataAugmenter::{self.mode} not implemented")
