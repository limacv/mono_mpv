from utils import *
from models.ModelWithLoss import *
from models.loss_utils import *
from models.mpi_utils import *
from models.flow_utils import *
from testposes import *

import torch.backends.cudnn


# Adjust configurations here ########################################################################################
path = "./log/checkpointsave/Ultly2ok_r0_6.pth"

out_prefix = "/home/lmaag/xgpu-scratch/mali_data/DAVIS_VID/mpv/"
video_list = glob("/home/lmaag/xgpu-scratch/mali_data/DAVIS_VID/video/*.mp4")
pipeline = smart_select_pipeline(path,
                                 force_pipelinename="fullv4")
ret_cfg = "dilate"

for video_path in video_list:
    pipeline.clear()
    save_disparity = True
    save_net = True

    saveprefix = "ZV4" + os.path.basename(path).split('.')[0] \
                 + os.path.basename(video_path).split('.')[0]
    dispvideo_path = os.path.join(out_prefix, saveprefix + "disparity")
    mpvout_path = os.path.join(out_prefix, saveprefix + ".mp4")
    if save_net:
        ret_cfg += "ret_net"

    # ## ### #### ##### ###### ####### ######## ####### ###### ##### #### ### ## #

    cap = cv2.VideoCapture(video_path)
    dispout = cv2.VideoWriter()
    mpvout = MPVWriter(mpvout_path)
    netout = NetWriter(mpvout_path)
    disparity_list = []

    reply_flag = True
    frameidx = 0
    with torch.no_grad():
        while True:
            print(f"\r{frameidx}", end='')
            ret, img = cap.read()
            # if frameidx % 10 != 0:
            #     frameidx += 1
            #     continue

            if not ret:
                break
            hei, wid, _ = img.shape
            if wid > hei * 2:
                img = img[:, :wid // 2]
                wid //= 2

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            hei, wid, _ = img.shape
            img_tensor = ToTensor()(img).cuda().unsqueeze(0)
            mpi = pipeline.infer_forward(img_tensor, ret_cfg=ret_cfg)

            if mpi is None:
                continue
            if isinstance(mpi, tuple):
                mpi, net = mpi
                disparity_list.append(pipeline.net2disparity(net[:, :4]).cpu())

            if save_net:
                netout.write(net)

            depthes = make_depths(mpi.shape[1]).cuda()
            disparity = estimate_disparity_torch(mpi, depthes)
            visdisp = draw_dense_disp(disparity, depthes[-1])[:, :, ::-1]

            if save_disparity:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if not dispout.isOpened():
                    dispout.open(dispvideo_path, fourcc, 30., (wid, hei), True)
                dispout.write(visdisp)
            frameidx += 1

        print("\n")
        if dispout is not None:
            dispout.release()
