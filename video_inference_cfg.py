from utils import *
from models.ModelWithLoss import *
from models.loss_utils import *
from models.mpi_utils import *
from models.flow_utils import *
from testposes import *

import torch.backends.cudnn
import os
# Adjust configurations here ########################################################################################
# path = "./log/checkpointsave/ablation01_svtemp_r0_4.pth"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-07-16-53-18.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_processed\\test\\HD720-02-16-06-57.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-04-15-33-25.mp4"
# video_path = "D:\\MSI_NB\\source\\data\\StereoBlur_test\\test\\HD720-02-15-49-26.mp4"
# video_path = "Z:\\dataset\\StereoVideoFinalv3\\videos\\StereoBlur_HD720-02-15-49-26_0.mp4"
# video_path = "Z:\\tmp\\VisualVideo\\crop\\zsoapbox.mp4"
# video_path = "Z:\\dataset\\StereoVideoFinalv3\\videos\\StereoBlur_HD720-02-15-34-24_1.mp4"
# video_path = "Z:\\dataset\\WebVideo\\cook\\_4fH_GX3rEM_2.mp4"
# video_path = "Z:\\dataset\\WebVideo\\penguin_Trim1.mp4"
video_path = "Z:\\dataset\\DAVIS-2017-Unsupervised-trainval-480p\\DAVIS\\JPEGImages\\vis\\car-turn.mp4"
outputres = (848, 480)  # (960, 540)

# outputres = (640, 360)  # (960, 540)
framestart = 0
frameend = 66
# newview_pose = target_posedx(70 / 640)
newview_pose = target_pose_soapbox

save_disparity = True
save_newview = True
save_mpv = True
save_net = True
reply_flag = False

out_prefix = "Z:\\tmp\\VisualLDI2MPI"
if not os.path.exists(out_prefix):
    out_prefix = "D:\\MSI_NB\\source\\data\\VisualLDI2MPI"


if __name__ == "__main__":
    import video_inferencev1
    import video_inferencev2
    import video_inferencev3
    import video_inferencev4
    import video_inferencelbtc
