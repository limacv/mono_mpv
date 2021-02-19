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
from . import MannequinChallenge_root, is_DEBUG, MannequinChallenge_skip_framenum
from . import OutputSize, OutputSize_test, MaxPointNumberForce
from .colmap_wrapper import *
from .Augmenter import DataAugmenter
from .youtubedl_wrapper import run_youtubedl


class MannequinChallenge_Base:
    def __init__(self, is_train, black_list, ptnum):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
        mpimodel=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        if is_train:
            self.root = os.path.join(MannequinChallenge_root, "train")
            self.trainstr = "train"
        else:
            self.root = os.path.join(MannequinChallenge_root, "test")
            self.trainstr = "test"
        
        self.valid_file_name = os.path.join(MannequinChallenge_root, f"{self.trainstr}_valid.txt")
        with open(self.valid_file_name, 'r') as valid_file:
            lines = valid_file.readlines()
            self.filebase_list = [self.txtpath2basename(line) for line in lines]

        if black_list:
            with open(os.path.join(MannequinChallenge_root, "black_list.txt")) as bl_file:
                lines = bl_file.readlines()
                _black_list = {self.txtpath2basename(line) for line in lines}
            print(f"MannequinChallenge: originally {len(self.filebase_list)}")
            self.filebase_list = list(set(self.filebase_list) - _black_list)
            print(f"MannequinChallenge: find {len(_black_list)} in black_list, "
                  f"after removal, got {len(self.filebase_list)} datas")

        self.name = f"MannequinChallenge_Base_{self.trainstr}"
        print(f"MannequinChallenge: find {len(self.filebase_list)} video files in {self.trainstr} set")

        self.tmp_root = os.path.join(os.path.dirname(self.root), f"{self.trainstr}tmp")
        if not os.path.exists(self.tmp_root):
            os.mkdir(self.tmp_root)

        self.maxptnum = ptnum if ptnum > 0 else MaxPointNumberForce
        self.totensor = ToTensor()
        self._cur_file_base = ""
        self._curvideo_trim_path = ""

    def video_trim_path(self, base_name):
        """
        base name is the code
        """
        return os.path.join(self.tmp_root, base_name, "video_Trim.mp4")

    @staticmethod
    def txtpath2basename(path):
        return os.path.splitext(os.path.basename(path.strip('\n').replace('\\', '/')))[0]

    def txt_path(self, base_name):
        return os.path.join(self.root, f"{base_name}.txt")

    def image_path(self, base_name):
        return os.path.join(self.tmp_root, base_name, "images")

    def tmp_base_path(self, base_name):
        return os.path.join(self.tmp_root, base_name)

    def colmap_model_path(self, base_name):
        return os.path.join(self.tmp_root, base_name, "sparse", "0")

    def pre_fetch_bybase(self, file_base):
        """
        Parse Txt file -> Download video -> Fetch&save frame -> SfM
        return: True: successfully fetch, False: failed
        """
        image_path = self.image_path(file_base)
        tmp_base_path = self.tmp_base_path(file_base)

        if not os.path.exists(tmp_base_path):
            os.mkdir(tmp_base_path)
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        timestamps, extrins = [], []
        with open(self.txt_path(file_base), 'r') as file:
            link = file.readline()
            while True:
                line = file.readline().split(' ')
                if len(line) != 19:
                    break
                timestamps.append(int(line[0]) / 1000)  # in millisecond
                intrin = list(map(float, line[1:5]))
                extrin = list(map(float, line[7:19]))
                extrins.append(np.array(extrin, dtype=np.float32).reshape((3, 4)))

        timestamps = timestamps[::MannequinChallenge_skip_framenum]
        extrins = extrins[::MannequinChallenge_skip_framenum]
        self._curtimestamps = timestamps.copy()
        # ================================================
        # if we don't have the images, we need to fetch the video and save images
        # ================================================
        video_trim_path = self.video_trim_path(file_base)
        if not (os.path.exists(image_path) and len(os.listdir(image_path)) == len(timestamps)) \
                and not os.path.exists(video_trim_path):
            print(f"  MannequinChallenge: download video from {file_base}", flush=True)

            try:
                run_youtubedl(link, tmp_base_path)
            except Exception as _:
                print(f"MannequinChallenge: error when fetching link {link} specified in {file_base}.txt"
                      f"with error:\n {_}")
                return False

            video_file = glob(f"{tmp_base_path}/video*")[0]
            video = cv2.VideoCapture(video_file)
            if not video.isOpened():
                print(f"MannequinChallenge: cannot open video in {file_base}.txt")
                return False
            for i, timestamp in enumerate(timestamps):
                video.set(cv2.CAP_PROP_POS_MSEC, timestamp)
                ret, frame = video.read()
                if not ret or frame.size == 0:
                    print(f"MannequinChallenge: error when reading frame {i} in {file_base}.txt")
                    return False
                cv2.imwrite(os.path.join(image_path, f"{int(timestamp):010d}.jpg"), frame)

            video.release()
            os.remove(video_file)

        # ================================================
        # if don't have sprase mpimodel, compute from colmap
        # ================================================
        col_model_root = self.colmap_model_path(file_base)
        if not os.path.exists(col_model_root) \
                or not os.path.exists(os.path.join(col_model_root, "points3D.bin")):
            print(f"  MannequinChallenge: run rolmap on {file_base}", flush=True)
            # provide pose and intrinsic to colmap
            image_list = list(map(os.path.basename, glob(f"{image_path}/*.jpg")))
            if len(image_list) != len(extrins):
                print(f"MannequinChallenge: image path corrpted in {file_base}.txt")
                return False
            frame = cv2.imread(image_path + '/' + image_list[0])
            _hei, _wid, _ = frame.shape
            camera_id = 1
            colcamera = Camera(camera_id, "PINHOLE", _wid, _hei,
                               (intrin[0] * _wid, intrin[1] * _hei, _wid * intrin[2], _hei * intrin[3]))
            colimages = [Image(i + 1, rotmat2qvec(extr[:, :3]), extr[:, 3], camera_id, image_list[i], (), ())
                         for i, extr in enumerate(extrins)]

            # run colmap with provided mpimodel
            try:
                run_colmap(tmp_base_path, ["feature", "match", "triangulator", "bundle_adjust"],
                           camera=colcamera, images=colimages, remove_database=True)
            except BaseException as e:
                print(f"MannequinChallenge: error when doing colmap in {file_base}.txt with error {e}")
                return False

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
                        print(f"MannequinChallenge: seems not support video encoder")
                        return False
                videoout.write(img)
                os.remove(image_name)
            videoout.release()
        return True

    def post_check(self, file_base, verbose=False):
        """
        check the quality of the point cloud
        make sure to call pre_fetcih_bybase before call this function,
        return false if not pass
        return true if pass
        """
        col_model_root = self.colmap_model_path(file_base)

        colcameras, colimages, colpoints3D = read_model(col_model_root, ".bin")

        framenum = len(colimages)
        # condition1: number of frames shouldn't be too little
        # -----------------------------------------------------
        if framenum < 6:
            if verbose:
                print(f"MannequinChallenge: too little frame ({framenum} frames)")
            return False

        ptnum = len(colpoints3D)
        pt3ds = [
            pt.xyz
            for i, pt in colpoints3D.items()
        ]
        pt3ds = np.array(pt3ds)
        # condition2: number of points shouldn't be too little
        # ---------------------------------------------------
        if ptnum < 1500:
            if verbose:
                print(f"MannequinChallenge: too little 3d points ({ptnum} points)")
            return False

        campt3d = [
            - cam.qvec2rotmat().T @ cam.tvec.reshape(3, 1)
            for i, cam in colimages.items()
        ]
        campt3d = np.array(campt3d).squeeze(-1)
        travel_distance = np.linalg.norm(campt3d[1:] - campt3d[:-1], axis=-1).sum()
        # condition3: camera position should be large enough
        # ---------------------------------------------------
        if travel_distance < 1.3:
            if verbose:
                print(f"MannequinChallenge: too little travel distance ({travel_distance})")
                return False

        # condition4: a special case when points are in a common plane
        # ---------------------------------------------------
        abc = np.linalg.inv(pt3ds.T @ pt3ds) @ pt3ds.T  @ np.ones_like(pt3ds[:, 0:1])
        abc /= np.linalg.norm(abc)
        dist2plane = pt3ds @ abc
        zs_std = np.std(dist2plane)
        if zs_std < 0.7:
            if verbose:
                print(f"MannequinChallenge: z direction std: ({zs_std})")
                return False

        return True


class MannequinChallenge_Img(Dataset, MannequinChallenge_Base):
    def __init__(self, is_train=True, black_list=True, mode='resize', ptnum=-1):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
        model=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        super().__init__(is_train=is_train,
                         black_list=black_list,
                         ptnum=ptnum)
        self.name = f"MannequinChallenge_Img_{self.trainstr}"
        Outsz = OutputSize if is_train else OutputSize_test
        self.augmenter = DataAugmenter(Outsz, mode=mode)

    def __len__(self):
        return len(self.filebase_list)

    def __getitem__(self, item):
        # try 3 times
        datas = None
        for i in range(3):
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
        return self.getitem_bybase(self.filebase_list[item])

    def getitem_bybase(self, file_base):
        """
        get triplit from path (img&pose_in_reference_view, img&pose_in_target_view, sparse_depth_in_ref_view)
        Get item specified in .txt file
        Will return None is something's wrong, pay special attention to this cast
        Attantion for the None return, means something is wrong
        """
        # ================================================
        # now we actually read data and return
        # ================================================
        if not self.pre_fetch_bybase(file_base):
            return None

        video_trim_path = self.video_trim_path(file_base)
        col_model_root = self.colmap_model_path(file_base)
        self._cur_file_base = file_base
        self._curvideo_trim_path = video_trim_path
        video = cv2.VideoCapture(video_trim_path)
        if not video.isOpened():
            print(f"MannequinChallenge: cannot open video file {video_trim_path}")
            return None
        framenum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.trainstr == "train":
            if framenum < 3:
                print(f"MannequinChallenge: {file_base}.txt has less than 3 image, skip")
                return None
            stride = min(np.random.randint(1, 4), framenum - 1)
            startidx = np.random.randint(framenum - stride)
            endidx = startidx + stride
            refidx, taridx = (startidx, endidx) if np.random.randint(2) else (endidx, startidx)
        else:
            refidx, taridx = 0, 5

        video.set(cv2.CAP_PROP_POS_FRAMES, refidx)
        ret0, refimg = video.read()
        video.set(cv2.CAP_PROP_POS_FRAMES, taridx)
        ret1, tarimg = video.read()
        if not (ret0 and len(refimg) > 0 and ret1 and len(tarimg)):
            print(f"MannequinChallenge: {file_base}.txt cannot read frame idx {refidx}, {taridx}")
            return None
        video.release()
        # refname, tarname = image_list[refidx], image_list[taridx]

        heiori, widori, _ = refimg.shape
        self.augmenter.random_generate((heiori, widori))
        refimg = cv2.cvtColor(refimg, cv2.COLOR_BGR2RGB)
        tarimg = cv2.cvtColor(tarimg, cv2.COLOR_BGR2RGB)
        refimg = self.totensor(self.augmenter.apply_img(refimg))
        tarimg = self.totensor(self.augmenter.apply_img(tarimg))
        # refimg = PImage.open(refname)
        # widori, heiori = refimg.size
        #
        # refimg = self.preprocess(refimg)
        # tarimg = self.preprocess(PImage.open(tarname))

        colcameras, colimages, colpoints3D = read_model(col_model_root, ".bin")
        if len(colimages) != framenum:
            print(f"MannequinChallenge: {file_base} colmap model doesn't match images, deleting everything")
            os.remove(os.path.join(col_model_root, "cameras.bin"))
            os.remove(os.path.join(col_model_root, "images.bin"))
            os.remove(os.path.join(col_model_root, "points3D.bin"))
            os.remove(video_trim_path)
            return None
        refview, tarview = colimages[refidx + 1], colimages[taridx + 1]
        # if refview.name != os.path.basename(refname) or tarview.name != os.path.basename(tarname):
        #     print(f"MannequinChallenge: error when choose ref view")
        #     return None
        point3ds = [colpoints3D[ptid].xyz for ptid in refview.point3D_ids if ptid >= 0]
        point3ds = np.array(point3ds, dtype=np.float32)
        if len(point3ds) <= 70:
            return None
        point2ds = refview.xys[refview.point3D_ids >= 0].astype(np.float32)

        refextrin = np.hstack([refview.qvec2rotmat(), refview.tvec.reshape(3, 1)]).astype(np.float32)
        tarextrin = np.hstack([tarview.qvec2rotmat(), tarview.tvec.reshape(3, 1)]).astype(np.float32)
        intrin = colcameras[1].params
        intrin = np.array([[intrin[0], 0, intrin[2]],
                           [0, intrin[1], intrin[3]],
                           [0, 0, 1]], dtype=np.float32)

        intrin = self.augmenter.apply_intrin(intrin)

        # compute depth of point3ds
        # print(f"refextrin.shape={refextrin.shape}")
        # print(f"later.shape={np.vstack([point3ds.T, np.ones((1, len(point3ds)), dtype=point3ds.dtype)]).shape}")
        point3ds_camnorm = refextrin @ np.vstack([point3ds.T, np.ones((1, len(point3ds)), dtype=point3ds.dtype)])
        point2ds_depth = point3ds_camnorm[2]
        # debug for corectness of intrin please delete afterwards
        # point3ds_imgnorm = intrin @ point3ds_camnorm
        # point3ds_imgnorm /= point3ds_imgnorm[2]

        ptindex = np.argsort(point2ds_depth)
        ptindex = ptindex[int(0.01 * len(ptindex)):int(0.99 * len(ptindex))]
        point2ds = point2ds[ptindex]
        point2ds_depth = point2ds_depth[ptindex]

        good_ptid = point2ds_depth > 0.000001
        point2ds = point2ds[good_ptid]
        point2ds_depth = point2ds_depth[good_ptid]

        point2ds, point2ds_depth = self.augmenter.apply_pts(point2ds, point2ds_depth)

        # random sample point so that output fixed number of points
        ptnum = point2ds.shape[0]
        if ptnum > self.maxptnum:
            ptsample = np.random.choice(ptnum, self.maxptnum, replace=False)
            point2ds = point2ds[ptsample]
            point2ds_depth = point2ds_depth[ptsample]

        return refimg, tarimg, \
               torch.tensor(refextrin), torch.tensor(tarextrin), \
               torch.tensor(intrin), torch.tensor(point2ds), torch.tensor(point2ds_depth)


class MannequinChallenge_Seq(Dataset, MannequinChallenge_Base):
    def __init__(self, is_train=True, black_list=True, mode='crop', ptnum=-1, seq_len=4, max_skip=1):
        """
        subset_byfile: if yes, then the dataset is get from the xxx_valid.txt file
        mpimodel=  'none': do noting
                'resize': resize to 512x512,
                'pad': pad to multiple of 128, usually used in evaluation,
                'crop': crop to 512x512 or multiple of 128
        """
        super().__init__(is_train=is_train,
                         black_list=black_list,
                         ptnum=ptnum)
        self.name = f"MannequinChallenge_Video_{self.trainstr}"
        self.sequence_length = seq_len
        self.tar_margin = 0
        Outsz = OutputSize if is_train else OutputSize_test
        self.augmenter = DataAugmenter(Outsz, mode=mode)
        self.maxskip_framenum = max(2, max_skip)  # 1 means no skip

    def __len__(self):
        return len(self.filebase_list)

    def __getitem__(self, item):
        # try 3 times
        datas = None
        for i in range(1):
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
        return self.getitem_bybase(self.filebase_list[item])

    def getitem_bybase(self, file_base):
        """
        get sequence from file
        Get item specified in .txt file
        Will return None is something's wrong, pay special attention to this cast
        Attantion for the None return, means something is wrong
        """
        # ================================================
        # select index and read images
        # ================================================
        file_base = os.path.basename(file_base).split('.')[0]
        video_trim_path = self.video_trim_path(file_base)
        col_model_root = self.colmap_model_path(file_base)
        self._cur_file_base = file_base
        self._curvideo_trim_path = video_trim_path

        video = cv2.VideoCapture(video_trim_path)
        if not video.isOpened():
            print(f"{self.name}: cannot open video file {video_trim_path}")
            return None
        framenum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if framenum < self.sequence_length + 1:
            return None

        if self.trainstr == "train":
            skipnum = np.random.randint(1, self.maxskip_framenum)
            framenum_wid = (self.sequence_length - 1) * skipnum + 1
            startid = np.random.randint(0, framenum - framenum_wid)
            refidxs = np.arange(startid, startid + framenum_wid, skipnum)
            taridxs = list(range(max(startid - self.tar_margin, 0),
                                 min(refidxs[-1] + self.tar_margin + 1, framenum)))
            taridx = taridxs[np.random.randint(len(taridxs))]
        else:
            refidxs, taridx = np.arange(1, 1 + self.sequence_length), self.sequence_length // 2 + 1

        refimgs = []
        for idx in refidxs:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret0, refimg = video.read()
            if not (ret0 and len(refimg) > 0):
                print(f"MannequinChallenge: {file_base}.txt cannot read frame idx {taridx}")
                return None

            refimg = refimg[:, :, ::-1]
            refimgs.append(refimg)

        heiori, widori, _ = refimgs[0].shape
        self.augmenter.random_generate((heiori, widori))
        refimgs = [self.totensor(self.augmenter.apply_img(im)) for im in refimgs]

        video.set(cv2.CAP_PROP_POS_FRAMES, taridx)
        ret1, tarimg = video.read()
        if not (ret1 and len(tarimg)):
            print(f"MannequinChallenge: {file_base}.txt cannot read frame idx {taridx}")
            return None
        video.release()

        tarimg = tarimg[:, :, ::-1]
        tarimg = self.totensor(self.augmenter.apply_img(tarimg))
        refimgs = torch.stack(refimgs, dim=0)

        # ================================================
        # read models
        # ================================================
        colcameras, colimages, colpoints3D = read_model(col_model_root, ".bin")
        if len(colimages) != framenum:
            print(f"MannequinChallenge: {file_base} colmap mpimodel doesn't match images, deleting everything")
            os.remove(os.path.join(col_model_root, "cameras.bin"))
            os.remove(os.path.join(col_model_root, "images.bin"))
            os.remove(os.path.join(col_model_root, "points3D.bin"))
            os.remove(video_trim_path)
            return None
        # 3d points in each reference view

        refextrins, pointxys, pointzs = [], [], []
        for refidx in refidxs:
            refview = colimages[refidx + 1]
            point3ds = [colpoints3D[ptid].xyz for ptid in refview.point3D_ids if ptid >= 0]
            point3ds = np.array(point3ds, dtype=np.float32)
            if len(point3ds) <= 70:
                return None
            point2ds = refview.xys[refview.point3D_ids >= 0].astype(np.float32)
            refextrin = np.hstack([refview.qvec2rotmat(), refview.tvec.reshape(3, 1)]).astype(np.float32)
            refextrins.append(refextrin)

            point3ds_camnorm = refextrin @ np.vstack([point3ds.T, np.ones((1, len(point3ds)), dtype=point3ds.dtype)])
            point2ds_depth = point3ds_camnorm[2]

            ptindex = np.argsort(point2ds_depth)
            ptindex = ptindex[int(0.01 * len(ptindex)):int(0.99 * len(ptindex))]
            point2ds = point2ds[ptindex]
            point2ds_depth = point2ds_depth[ptindex]

            good_ptid = point2ds_depth > 0.000001
            point2ds = point2ds[good_ptid]
            point2ds_depth = point2ds_depth[good_ptid]

            point2ds, point2ds_depth = self.augmenter.apply_pts(point2ds, point2ds_depth)
            if len(point2ds) < 70:
                return None
            # random sample point so that output fixed number of points
            # ptnum = point2ds.shape[0]
            # ptsample = np.random.choice(ptnum, self.maxptnum, replace=(ptnum < self.maxptnum))

            pointxys.append(point2ds)
            pointzs.append(point2ds_depth)

        # decide the number of points
        ptnum = min([len(pts) for pts in pointzs] + [self.maxptnum])
        pointxys1, pointzs1 = [], []
        for point2ds, point2ds_depth in zip(pointxys, pointzs):
            ptsample = np.random.choice(len(point2ds), ptnum, replace=False)
            pointxys1.append(point2ds[ptsample])
            pointzs1.append(point2ds_depth[ptsample])

        refextrins = np.stack(refextrins, axis=0)
        pointxys = np.stack(pointxys1, axis=0)
        pointzs = np.stack(pointzs1, axis=0)

        # target view parameters and intrinsic
        tarview = colimages[taridx + 1]
        tarextrin = np.hstack([tarview.qvec2rotmat(), tarview.tvec.reshape(3, 1)]).astype(np.float32)

        intrin = colcameras[1].params
        intrin = np.array([[intrin[0], 0, intrin[2]],
                           [0, intrin[1], intrin[3]],
                           [0, 0, 1]], dtype=np.float32)
        intrin = self.augmenter.apply_intrin(intrin)

        return refimgs, tarimg, \
               torch.tensor(refextrins), torch.tensor(tarextrin), \
               torch.tensor(intrin), torch.tensor(pointxys), torch.tensor(pointzs)

