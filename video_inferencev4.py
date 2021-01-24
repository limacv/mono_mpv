from utils import *
from models.ModelWithLoss import *
from models.loss_utils import *
from models.mpi_utils import *
from models.flow_utils import *
import torch.backends.cudnn


def str2bool(s_):
    try:
        s_ = int(s_)
    except ValueError:
        try:
            s_ = float(s_)
        except ValueError:
            if s_ == 'True':
                s_ = True
            elif s_ == 'False':
                s_ = False
    return s_


path = "./log/checkpoint/v53LR_netflownet_191850_r0.pth"
# Adjust configurations here ############################################
state_dict_path = {
    '': "./log/checkpoint/v53LR_netflownet_191850_r0.pth",
    # "MPF.": "./log/checkpoint/mpf_bugfix_ord1smth_052107_r0.pth"
}
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-07-16-53-18.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-02-16-06-57.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-02-14-07-38.mp4"
video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-02-15-49-26.mp4"
# video_path = "Z:\\dataset\\StereoBlur_processed\\30fps\\HD720-02-15-49-26.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\MannequinChallenge\\testtmp\\00c4a2d23c90fbc9\\video_Trim.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\MannequinChallenge\\traintmp\\0a312f741fdf5d89\\video_Trim.mp4"

# read config from state_dict
cfg_str = torch.load(state_dict_path[''])["cfg"]
loss_weights = [i for i in cfg_str.split('\n') if i.startswith("Loss: ")][0][6:].split(', ')
loss_weights = {i.split(':')[0]: str2bool(i.split(':')[1]) for i in loss_weights}
modelname = [i for i in cfg_str.split('\n') if i.startswith("Model: ")][0].split(', ')[0][7:]

model = select_module(modelname).cuda()
pipeline = PipelineV4(model, {"loss_weights": loss_weights,
                              "winsz": 7})
ret_cfg = "dbg"

save_infer_mpi = True
save_disparity = True
save_mpv = True
# \Adjust configuration here ############################################

out_prefix = "Z:\\tmp\\Visual"
if not os.path.exists(out_prefix):
    out_prefix = "D:\\MSI_NB\\source\\data\\Visual"
if "StereoBlur" in video_path:
    saveprefix = "ZV4" + os.path.basename(state_dict_path['']).split('.')[0] \
                 + os.path.basename(video_path).split('.')[0] + ret_cfg
else:
    saveprefix = "ZV4" + os.path.basename(state_dict_path['']).split('.')[0] \
                 + os.path.basename(os.path.dirname(video_path)).split('.')[0] + ret_cfg
dispvideo_path = os.path.join(out_prefix, saveprefix + "_disparity.mp4")
mpvout_path = os.path.join(out_prefix, saveprefix + ".mp4")
smart_load_checkpoint('', {"check_point": state_dict_path}, model)

# ## ### #### ##### ###### ####### ######## ####### ###### ##### #### ### ## #

cap = cv2.VideoCapture(video_path)
dispout = cv2.VideoWriter()
mpvout = MPVWriter(mpvout_path)

frameidx = 0
with torch.no_grad():
    while True:
        print(f"\r{frameidx}", end='')
        ret, img = cap.read()
        # if frameidx % 10 != 0:
        #     frameidx += 1
        #     continue

        if not ret or frameidx > 57:
            break
        hei, wid, _ = img.shape
        if wid > hei * 2:
            img = img[:, :wid // 2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, None, None, 0.5, 0.5)
        hei, wid, _ = img.shape
        img_tensor = ToTensor()(img).cuda().unsqueeze(0)
        mpi = pipeline.infer_forward(img_tensor, ret_cfg=ret_cfg)

        if mpi is None:
            continue
        if save_mpv:
            mpvout.write(mpi[0])
        depthes = make_depths(mpi.shape[1]).cuda()
        disparity = estimate_disparity_torch(mpi, depthes)
        visdisp = draw_dense_disp(disparity, depthes[-1])[:, :, ::-1]

        if save_disparity:
            if not dispout.isOpened():
                dispout.open(dispvideo_path, 828601953, 30., (wid, hei), True)
            dispout.write(visdisp)
        frameidx += 1

    print("\n")
    if dispout is not None:
        dispout.release()
