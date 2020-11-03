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


RealEstate10K_skip_framenum = 3
# resize_to_w, resize_to_h = 512 + 128 * 3, 512
resize_to_w, resize_to_h = 512 + 128 * 3, 512


class RealEstate10K(Dataset):
    """
    The dataset has 7711 test video and 71556 train video from youtube real estate video
    """
    def __init__(self, is_train=True, subset_byfile=False, mode='resize', ptnum=2000):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
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

        if subset_byfile:
            self.valid_file_name = os.path.join(RealEstate10K_root, f"{self.trainstr}_valid.txt")
            with open(self.valid_file_name, 'r') as valid_file:
                lines = valid_file.readlines()
                self.file_list = [os.path.join(self.root, os.path.basename(line.strip('\n').replace('\\', '/')))
                                  for line in lines]
        else:
            self.file_list = glob(f"{self.root}/*.txt")

        self.name = f"RealEstate10K_{self.trainstr}"
        print(f"RealEstate10K: find {len(self.file_list)} video files in {self.trainstr} set")

        self.tmp_root = os.path.join(os.path.dirname(self.root), f"{self.trainstr}tmp")
        if not os.path.exists(self.tmp_root):
            os.mkdir(self.tmp_root)

        self.mode = mode
        if mode == 'none':
            self.preprocess = ToTensor()
        elif mode == 'resize':
            self.preprocess = Compose([Resize([resize_to_h, resize_to_w]),
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
        # try 5 times
        datas = None
        for i in range(5):
            try:
                datas = self.getitem(item)
            except Exception:
                datas = None
            if datas is not None:
                return datas

        # if still not working, randomly pick another idx untill success
        while datas is None:
            item = np.random.randint(len(self))
            datas = self.getitem(item)
        return datas

    def getitem(self, item):
        return self.getitem_bypath(self.file_list[item])

    def getitem_bypath(self, path):
        """
        Get item specified in item-th .txt file
        Will return None is something's wrong, pay special attention to this cast
        Parse Txt file -> Download video -> Fetch&save frame -> SfM
        Attantion for the None return, means something is wrong
        """
        file_name = path  # self.file_list[item]
        assert path in self.file_list
        file_base = os.path.basename(file_name).split('.')[0]
        dir_base = os.path.join(self.tmp_root, file_base)
        image_path = os.path.join(dir_base, "images")

        if not os.path.exists(dir_base):
            os.mkdir(dir_base)
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        timestamps, extrins = [], []
        with open(file_name, 'r') as file:
            link = file.readline()
            while True:
                line = file.readline().split(' ')
                if len(line) != 19:
                    break
                timestamps.append(int(line[0]) / 1000)  # in millisecond
                intrin = list(map(float, line[1:5]))
                extrin = list(map(float, line[7:19]))
                extrins.append(np.array(extrin, dtype=np.float32).reshape((3, 4)))

        timestamps = timestamps[::RealEstate10K_skip_framenum]
        extrins = extrins[::RealEstate10K_skip_framenum]
        self._curtimestamps = timestamps.copy()
        # ================================================
        # if we don't have the images, we need to fetch the video and save images
        # ================================================
        video_trim_path = os.path.join(dir_base, "video_Trim.mp4")
        if not (os.path.exists(image_path) and len(os.listdir(image_path)) == len(timestamps)) \
                and not os.path.exists(video_trim_path):
            print(f"  RealEstate10K: download video from {os.path.basename(dir_base)}", flush=True)
            try:
                youtube = YouTube(link)
                stream = youtube.streams.get_highest_resolution()
                stream.download(dir_base, "video", skip_existing=True)
            except KeyError as e:
                print(f"RealEstate10K: error when fetching link {link[:-1]} specified in {file_base}.txt"
                      f"with error {e}")
                return None
            except Exception as _:
                try:
                    youtube = YouTube(link)
                    stream = youtube.streams.first()
                    stream.download(dir_base, "video", skip_existing=True)
                except Exception as e:
                    print(f"RealEstate10K: error when fetching link {link} specified in {file_base}.txt"
                          f"with error {e} in second try")
                finally:
                    return None

            video_file = glob(f"{dir_base}/video*")[0]
            video = cv2.VideoCapture(video_file)
            if not video.isOpened():
                print(f"RealEstate10K: cannot open video in {file_base}.txt")
                return None
            for i, timestamp in enumerate(timestamps):
                video.set(cv2.CAP_PROP_POS_MSEC, timestamp)
                ret, frame = video.read()
                if not ret or frame.size == 0:
                    print(f"RealEstate10K: error when reading frame {i} in {file_base}.txt")
                    return None
                cv2.imwrite(os.path.join(image_path, f"{int(timestamp):010d}.jpg"), frame)

            video.release()
            os.remove(video_file)

        # ================================================
        # if don't have sprase model, compute from colmap
        # ================================================
        col_model_root = os.path.join(dir_base, "sparse", "0")
        if not os.path.exists(col_model_root)\
                or not os.path.exists(os.path.join(col_model_root, "points3D.bin")):
            print(f"  RealEstate10K: run rolmap on {os.path.basename(dir_base)}", flush=True)
            # provide pose and intrinsic to colmap
            image_list = list(map(os.path.basename, glob(f"{image_path}/*.jpg")))
            if len(image_list) != len(extrins):
                print(f"RealEstate10K: image path corrpted in {file_base}.txt")
                return None
            frame = cv2.imread(image_path + '/' + image_list[0])
            _hei, _wid, _ = frame.shape
            camera_id = 1
            colcamera = Camera(camera_id, "PINHOLE", _wid, _hei,
                               (intrin[0] * _wid, intrin[1] * _hei, _wid * intrin[2], _hei * intrin[3]))
            colimages = [Image(i+1, rotmat2qvec(extr[:, :3]), extr[:, 3], camera_id, image_list[i], (), ())
                         for i, extr in enumerate(extrins)]

            # run colmap with provided model
            try:
                run_colmap(dir_base, ["feature", "match", "triangulator"],
                           camera=colcamera, images=colimages, remove_database=True)
            except BaseException as e:
                print(f"RealEstate10K: error when doing colmap in {file_base}.txt with error {e}")
                return None

        # ================================================
        # to save memory usage, we covert images to video
        # ================================================
        image_list = sorted(glob(f"{image_path}/*.jpg"))
        if not os.path.exists(video_trim_path) and len(image_list) > 0:
            videoout = cv2.VideoWriter()
            for image_name in image_list:
                img = cv2.imread(image_name)
                if not videoout.isOpened():
                    videoout.open(video_trim_path, 828601953, 30,
                                  (img.shape[1], img.shape[0]), True)
                    if not videoout.isOpened():
                        print(f"RealEstate10K: seems not support video encoder")
                        return None
                videoout.write(img)
                os.remove(image_name)
            videoout.release()

        # ================================================
        # now we actually read data and return
        # ================================================
        self._curvideo_trim_path = video_trim_path
        video = cv2.VideoCapture(video_trim_path)
        if not video.isOpened():
            print(f"RealEstate10K: cannot open video file {video_trim_path}")
            return None
        framenum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.trainstr == "train":
            if framenum < 3:
                print(f"RealEstate10K: {file_base}.txt has less than 3 image, skip")
                return None
            stride = min(np.random.randint(3 // RealEstate10K_skip_framenum + 1,
                                           20 // RealEstate10K_skip_framenum), framenum - 1)
            startidx = np.random.randint(framenum - stride)
            endidx = startidx + stride
            refidx, taridx = (startidx, endidx) if np.random.randint(2) else (endidx, startidx)
        else:
            refidx, taridx = 0, 5

        if is_DEBUG:
            print(f"read from {image_path} on {refidx} and {taridx}")

        video.set(cv2.CAP_PROP_POS_FRAMES, refidx)
        ret0, refimg = video.read()
        video.set(cv2.CAP_PROP_POS_FRAMES, taridx)
        ret1, tarimg = video.read()
        if not (ret0 and len(refimg) > 0 and ret1 and len(tarimg)):
            print(f"RealEstate10K: {file_base}.txt cannot read frame idx {refidx}, {taridx}")
            return None
        video.release()
        # refname, tarname = image_list[refidx], image_list[taridx]

        heiori, widori, _ = refimg.shape
        refimg = PImage.fromarray(cv2.cvtColor(refimg, cv2.COLOR_BGR2RGB))
        tarimg = PImage.fromarray(cv2.cvtColor(tarimg, cv2.COLOR_BGR2RGB))
        refimg = self.preprocess(refimg)
        tarimg = self.preprocess(tarimg)
        # refimg = PImage.open(refname)
        # widori, heiori = refimg.size
        #
        # refimg = self.preprocess(refimg)
        # tarimg = self.preprocess(PImage.open(tarname))

        colcameras, colimages, colpoints3D = read_model(col_model_root, ".bin")
        if len(colimages) != framenum:
            print(f"RealEstate10K: {file_base} colmap model doesn't match images, deleting everything")
            os.remove(os.path.join(col_model_root, "cameras.bin"))
            os.remove(os.path.join(col_model_root, "images.bin"))
            os.remove(os.path.join(col_model_root, "points3D.bin"))
            os.remove(video_trim_path)
            return None
        refview, tarview = colimages[refidx + 1], colimages[taridx + 1]
        # if refview.name != os.path.basename(refname) or tarview.name != os.path.basename(tarname):
        #     print(f"RealEstate10K: error when choose ref view")
        #     return None
        point3ds = [colpoints3D[ptid].xyz for ptid in refview.point3D_ids if ptid >= 0]
        point3ds = np.array(point3ds, dtype=np.float32)
        if len(point3ds) <= 100:
            return None
        point2ds = refview.xys[refview.point3D_ids >= 0].astype(np.float32)

        refextrin = np.hstack([refview.qvec2rotmat(), refview.tvec.reshape(3, 1)]).astype(np.float32)
        tarextrin = np.hstack([tarview.qvec2rotmat(), tarview.tvec.reshape(3, 1)]).astype(np.float32)
        intrin = colcameras[1].params
        intrin = np.array([[intrin[0], 0, intrin[2]],
                           [0, intrin[1], intrin[3]],
                           [0, 0, 1]], dtype=np.float32)

        if self.mode == 'resize':
            intrin_calib = np.array([resize_to_w / widori, 0, 0,
                                     0, resize_to_h / heiori, 0,
                                     0, 0, 1], dtype=np.float32).reshape(3, 3)
            intrin = intrin_calib @ intrin

        # compute depth of point3ds
        # print(f"refextrin.shape={refextrin.shape}")
        # print(f"later.shape={np.vstack([point3ds.T, np.ones((1, len(point3ds)), dtype=point3ds.dtype)]).shape}")
        point3ds_camnorm = refextrin @ np.vstack([point3ds.T, np.ones((1, len(point3ds)), dtype=point3ds.dtype)])
        point2ds_depth = point3ds_camnorm[2]

        good_ptid = point2ds_depth > 0.01
        point2ds = point2ds[good_ptid]
        point2ds_depth = point2ds_depth[good_ptid]
        # normalize pt2ds to -1 to 1
        point2ds = point2ds * np.array([2 / widori, 2 / heiori], dtype=np.float32) - 1.

        # random sample point so that output fixed number of points
        ptnum = point2ds.shape[0]
        ptsample = np.random.choice(ptnum, self.ptnum, replace=(ptnum < self.ptnum))
        point2ds = point2ds[ptsample]
        point2ds_depth = point2ds_depth[ptsample]

        return refimg, tarimg, \
               torch.tensor(refextrin), torch.tensor(tarextrin), \
               torch.tensor(intrin), torch.tensor(point2ds), torch.tensor(point2ds_depth)


if __name__ == "__main__":
    example_file = "D:\\MSI_NB\\source\\data\\RealEstate10K\\train\\aaa1ef2a365d7781.txt"
    output_root = "D:\\MSI_NB\\source\\data\\RealEstate10K\\temp\\"
    base_name = os.path.basename(example_file).split('.')[0]
    with open(example_file, 'r') as file:
        timestamps = []
        intrins = []
        extrins = []
        link = file.readline()
        while True:
            line = file.readline().split(' ')
            if len(line) != 19:
                break
            timestamp = int(line[0])
            timestamps.append(timestamp / 1000)
            intrin = list(map(float, line[1:5]))
            extrin = list(map(float, line[7:19]))
            intrins.append(np.array(intrin, dtype=np.float32))
            extrins.append(np.array(extrin, dtype=np.float32).reshape((3, 4)))
    ytb = YouTube(link)
    stream = ytb.streams.get_highest_resolution()
    stream.download(output_root, base_name)

    video = cv2.VideoCapture(f"{output_root}/{base_name}.mp4")
    for timestamp in timestamps:
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        ret, frame = video.read()
        cv2.imshow('few', frame)
        cv2.waitKey(0)
    pass
