from utils import *
from models.ModelWithLoss import *
from models.loss_utils import *
from models.mpi_utils import *
from models.flow_utils import *
import torch.backends.cudnn


# Adjust configurations here ########################################################################################
path = "./log/checkpoint/V5Dual_fgbgsame_032123_r0.pth"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-07-16-53-18.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-02-16-06-57.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-06-15-23-27.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-02-15-49-26.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\pg6_Trim.mp4"
video_path = "D:\\MSI_NB\\source\\data\\RealEstate10K\\testtmp\\ccc439d4b28c87b2\\video_Trim.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\RealEstate10K\\testtmp\\ccc439d4b28c87b2\\video_Trim_r.mp4"
# video_path = "Z:\\dataset\\StereoBlur_processed\\30fps\\HD720-02-15-49-26.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\MannequinChallenge\\testtmp\\00c4a2d23c90fbc9\\video_Trim.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\MannequinChallenge\\traintmp\\0a312f741fdf5d89\\video_Trim.mp4"

# modelloss = smart_select_pipeline(path, "MPINetv2", "disp_img")
modelloss = smart_select_pipeline(path)

ret_cfg = " "

save_newview = False
save_disparity = True
save_mpv = True
save_net = False
regular_video = False
# \Adjust configuration here ########################################################################################

out_prefix = "Z:\\tmp\\Visual"
if not os.path.exists(out_prefix):
    out_prefix = "D:\\MSI_NB\\source\\data\\Visual"
if "StereoBlur" in video_path:
    saveprefix = os.path.basename(path).split('.')[0] \
                 + os.path.basename(video_path).split('.')[0] + ret_cfg
elif "MannequinChallenge" in video_path or "RealEstate10K" in video_path:
    saveprefix = os.path.basename(path).split('.')[0] \
                 + os.path.basename(os.path.dirname(video_path)).split('.')[0] + ret_cfg
else:  # regular video
    regular_video = True
    saveprefix = os.path.basename(path).split('.')[0] \
                 + os.path.basename(video_path).split('.')[0]
dispvideo_path = os.path.join(out_prefix, saveprefix + "_disparity.mp4")
newviewsvideo_path = os.path.join(out_prefix, saveprefix + "_newview.mp4")
mpiout_path = os.path.join(out_prefix, saveprefix)
mpvout_path = os.path.join(out_prefix, saveprefix + ".mp4")
if save_net:
    ret_cfg += "ret_net"

# ## ### #### ##### ###### ####### ######## ####### ###### ##### #### ### ## #

cap = cv2.VideoCapture(video_path)
dispout = cv2.VideoWriter()
newview_out = cv2.VideoWriter() if save_newview else None
mpvout = MPVWriter(mpvout_path)
netout = NetWriter(mpvout_path)

frameidx = 0
with torch.no_grad():
    while True:
        print(f"\r{frameidx}", end='')
        ret, img = cap.read()
        # if frameidx % 10 != 0:
        #     frameidx += 1
        #     continue

        if not ret or frameidx > 50:
            break
        hei, wid, _ = img.shape
        if wid > hei * 2:
            img = img[:, :wid // 2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, None, None, 0.5, 0.5)
        hei, wid, _ = img.shape
        img_tensor = ToTensor()(img).cuda()

        mpi = modelloss.infer_forward(img_tensor, ret_cfg=ret_cfg)

        if mpi is None:
            continue
        if isinstance(mpi, tuple):
            mpi, net = mpi
        if save_net:
            netout.write(net)
        if save_mpv:
            mpvout.write(mpi[0])
        depthes = make_depths(mpi.shape[1]).cuda()
        disparity = estimate_disparity_torch(mpi, depthes)
        visdisp = draw_dense_disp(disparity, depthes[-1])[:, :, ::-1]

        if save_newview:
            target_pose = torch.tensor(
                [[1.0, 0.0, 0.0, -0.1],
                 [0.0, 1.0, 0.0, 0],
                 [0.0, 0.0, 1.0, 0]]
            ).type_as(mpi).unsqueeze(0)
            source_pose = torch.tensor(
                [[1.0, 0.0, 0.0, 0],
                 [0.0, 1.0, 0.0, 0],
                 [0.0, 0.0, 1.0, 0]]
            ).type_as(mpi).unsqueeze(0)
            intrin = torch.tensor(
                [[wid / 2, 0.0, wid / 2],
                 [0.0, hei / 2, hei / 2],
                 [0.0, 0.0, 1.0]]
            ).type_as(mpi).unsqueeze(0)
            view = render_newview(mpi, source_pose, target_pose, intrin, intrin, depthes)
            visview = (view * 255).type(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
            visview = cv2.cvtColor(visview, cv2.COLOR_RGB2BGR)

        if save_disparity:
            if not dispout.isOpened():
                dispout.open(dispvideo_path, 828601953, 30., (wid, hei), True)
                if newview_out is not None:
                    newview_out.open(newviewsvideo_path, 828601953, 30., (wid, hei), True)
            dispout.write(visdisp)
        if newview_out is not None:
            newview_out.write(visview)

        frameidx += 1

    print("\n")
    if dispout is not None:
        dispout.release()
    if newview_out is not None:
        newview_out.release()
