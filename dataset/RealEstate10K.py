from pytube import YouTube
import numpy as np
from io import BytesIO
from torchvision.transforms import ToTensor
import imageio
from torch.utils.data import Dataset
import cv2
import os
from glob import glob
from .colmap_wrapper import *


class RealEstate10K(Dataset):
    """
    The dataset has 7711 test video and 71556 train video from youtube real estate video
    """
    def __init__(self, path):
        self.root = os.path.abspath(path)
        if "test" in path:
            self.is_train = "test"
        elif "train" in path:
            self.is_train = "train"
        else:
            raise RuntimeError(f"RealEstate10K: unrecoginized path")

        self.file_list = glob(f"{path}/*.txt")
        print(f"RealEstate10K: find {len(self.file_list)} video files in {self.is_train} set")

        self.tmp_root = os.path.join(os.path.dirname(self.root), f"{self.is_train}tmp")
        if not os.path.exists(self.tmp_root):
            os.mkdir(self.tmp_root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        # try 5 times
        datas = None
        for i in range(5):
            datas = self.getitem(item)
            if datas is not None:
                return datas

        # if still not working, randomly pick another idx untill success
        while datas is None:
            item = np.random.randint(len(self))
            datas = self.getitem(item)
        return datas

    def getitem(self, item):
        """
        Get item specified in item-th .txt file
        Will return None is something's wrong, pay special attention to this cast
        Parse Txt file -> Download video -> Fetch&save frame -> SfM
        Attantion for the None return, means something is wrong
        """
        file_name = self.file_list[item]
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

        # reduce memory by skip every 6 frames
        timestamps = timestamps[::6]
        extrins = extrins[::6]
        # ================================================
        # if we don't have the images, we need to fetch the video and save images
        # ================================================
        if not os.path.exists(image_path) or len(os.listdir(image_path)) == 0:
            try:
                youtube = YouTube(link)
                stream = youtube.streams.first()
                stream.download(dir_base, "video", skip_existing=True)
            except BaseException:
                print(f"RealEstate10K: error when fetching link {link} specified in {file_base}.txt")
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
                cv2.imwrite(os.path.join(image_path, f"{int(timestamp)}.jpg"), frame)

            video.release()
            os.remove(video_file)

        # ================================================
        # if don't have sprase model, compute from colmap
        # ================================================
        col_model_root = os.path.join(dir_base, "sparse", "0")
        if not os.path.exists(col_model_root)\
                or not os.path.exists(os.path.join(col_model_root, "points3D.bin")):
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
            except BaseException:
                print(f"RealEstate10K: error when doing colmap in {file_base}.txt")
                return None

        # ================================================
        # now we actually read data and return
        # ================================================
        image_list = sorted(glob(f"{image_path}/*.jpg"))
        refidx, taridx = np.random.choice(len(image_list), 2, replace=False)  # randomly choose two frames
        refname, tarname = image_list[refidx], image_list[taridx]
        refimg = imageio.imread(refname)
        tarimg = imageio.imread(tarname)

        colcameras, colimages, colpoints3D = read_model(col_model_root, ".bin")
        refview, tarview = colimages[refidx + 1], colimages[taridx + 1]
        if refview.name != os.path.basename(refname) or tarview.name != os.path.basename(tarname):
            print(f"RealEstate10K: error when choose ref view")
            return None
        point3ds = [colpoints3D[ptid].xyz for ptid in refview.point3D_ids if ptid >= 0]
        point3ds = np.array(point3ds)
        point2ds = refview.xys[refview.point3D_ids >= 0]

        refpose = np.hstack([refview.qvec2rotmat(), refview.tvec.reshape(3, 1)])
        tarpose = np.hstack([tarview.qvec2rotmat(), tarview.tvec.reshape(3, 1)])
        intrin = colcameras[1].params
        intrin = np.array([[intrin[0], 0, intrin[2]],
                           [0, intrin[1], intrin[3]],
                           [0, 0, 1]], dtype=np.float32)
        # compute depth of point3ds
        point3ds_camnorm = refpose @ np.vstack([point3ds.T, np.ones((1, len(point3ds)), dtype=point3ds.dtype)])
        point2ds_depth = point3ds_camnorm[2]

        return refimg, tarimg, refpose, tarpose, intrin, point2ds, point2ds_depth


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
