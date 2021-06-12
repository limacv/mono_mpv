from utils import *
from models.ModelWithLoss import *
from models.loss_utils import *
from models.mpi_utils import *
from models.flow_utils import *
from testposes import *

import torch.backends.cudnn

# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-07-16-53-18.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-02-16-06-57.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-04-15-33-25.mp4"
video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-02-15-49-26.mp4"
# video_path = "Z:\\dataset\\StereoVideoFinalv3\\videos\\StereoBlur_HD720-02-15-49-26_0.mp4"
# video_path = "Z:\\dataset\\WebVideo\\cook\\_4fH_GX3rEM_2.mp4"
# video_path = "Z:\\dataset\\WebVideo\\penguin_Trim1.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-05-16-39-56.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\pg6_Trim.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\flickr2.mp4"

outputres = (640, 360)  # (960, 540)
# \Adjust configuration here ########################################################################################

out_prefix = "Z:\\tmp\\VisualNeat"
if not os.path.exists(out_prefix):
    out_prefix = "D:\\MSI_NB\\source\\data\\Visual"

saveprefix = "GT" + os.path.basename(video_path).split('.')[0]
viewvideo_path = os.path.join(out_prefix, saveprefix + "_newview.mp4")

# ## ### #### ##### ###### ####### ######## ####### ###### ##### #### ### ## #

cap = cv2.VideoCapture(video_path)
dispout = cv2.VideoWriter()
viewout = MyVideoWriter(viewvideo_path)
disparity_list = []

reply_flag = True
framestart = 0
frameidx = 0
with torch.no_grad():
    while True:
        print(f"\r{frameidx}", end='')
        ret, img = cap.read()
        if frameidx < framestart:
            frameidx += 1
            continue

        if not ret or frameidx > 60:
            break

        # process
        heiori, widori, _ = img.shape
        img = img[:, widori // 2:]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, outputres)

        hei, wid, _ = img.shape
        if (wid / hei) != (outputres[0] / outputres[1]) and reply_flag:
            reply_flag = False
            reply = input(f"original ratio {wid / hei:.2f} ({wid}x{hei}),"
                          f"the outputres' ratio {outputres[0] / outputres[1]:.2f} {outputres}, \n"
                          f"Do you want to continue? y/yes, \nor continue with original resolution? c")
            if 'c' in reply:
                outputres = (wid, hei)
            elif 'y' not in reply:
                exit()

        img_tensor = ToTensor()(img).unsqueeze(0)

        viewout.write(img_tensor)
        frameidx += 1
