from torchvision.transforms import ToTensor

from trainer import *
from models.ModelWithLoss import *
from models.loss_utils import *
from models.mpi_utils import *
import torch.backends.cudnn


# Adjust configurations here ############################################
state_dict_path = "./log/checkpoint/mpfnet_learnonlympf_031801_r0.pth"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-07-16-53-18.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-02-16-06-57.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-02-14-07-38.mp4"
video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-05-16-38-15.mp4"
model = select_module("MPFNet").cuda()
modelloss = select_modelloss("disp_mpf")(model, {"loss_weights": {}})
infer_single_frame = False
save_infer_mpi = True and infer_single_frame
infer_entire_video = not infer_single_frame
save_newview = False
save_mpv = True
save_mpf = True
# \Adjust configuration here ############################################

out_prefix = "D:\\MSI_NB\\source\\data\\Visual"
saveprefix = os.path.basename(state_dict_path).split('.')[0] + os.path.basename(video_path).split('.')[0]
dispvideo_path = os.path.join(out_prefix, saveprefix + "_disparity.mp4")
newviewsvideo_path = os.path.join(out_prefix, saveprefix + "_newview.mp4")
mpiout_path = os.path.join(out_prefix, saveprefix)
mpvout_path = os.path.join(out_prefix, saveprefix + ".mp4")
state_dict = torch.load(state_dict_path, map_location='cuda:0')
# torch.save({"state_dict": state_dict["state_dict"]}, state_dict_path)
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict)

# ## ### #### ##### ###### ####### ######## ####### ###### ##### #### ### ## #

cap = cv2.VideoCapture(video_path)
dispout = cv2.VideoWriter()
newview_out = cv2.VideoWriter() if save_newview else None
mpvout = MPVWriter(mpvout_path)
mpfout = MPFWriter(mpvout_path)
# if infer_single_frame:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 5)

frameidx = 0
with torch.no_grad():
    while True:
        print(f"\r{frameidx}", end='')
        ret, img = cap.read()
        if not ret or frameidx > 20:
            break
        hei, wid, _ = img.shape
        if wid > hei * 2:
            img = img[:, :wid//2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, None, None, 0.5, 0.5)
        hei, wid, _ = img.shape
        img_tensor = ToTensor()(img).cuda()
        mpi = modelloss.infer_forward(img_tensor)

        if isinstance(mpi, tuple):
            mpi, mpf = mpi

        if save_mpf:
            mpfout.write(mpi[0], mpf[0])
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

        if infer_single_frame and frameidx == 5:
            if save_infer_mpi:
                save_mpi(mpi, mpiout_path)
            cv2.imwrite(dispvideo_path + ".jpg", visdisp)
            if save_newview:
                cv2.imwrite(newviewsvideo_path + ".jpg", visview)
            break

        if not dispout.isOpened():
            dispout.open(dispvideo_path, 828601953, 30., (wid, hei), True)
            if newview_out is not None:
                newview_out.open(newviewsvideo_path, 828601953, 30., (wid, hei), True)
        dispout.write(visdisp)
        if newview_out is not None:
            newview_out.write(visview)

        frameidx += 1

    print("\n")
    dispout.release()
    if newview_out is not None:
        newview_out.release()
