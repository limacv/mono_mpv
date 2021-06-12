
from utils import *
from models.ModelWithLoss import *
from models.loss_utils import *
from models.mpi_utils import *
from models.flow_utils import *
from testposes import *
from video_inference_cfg import video_path, framestart, frameend

isleft = True

lrstr = "left" if isleft else "right"
disparity_path = video_path.replace("\\videos\\", "\\disparities\\").split('.')[0] + f"\\{lrstr}"

disparity_list = []
disp_max = 0

for frameidx in range(framestart, frameend):
    disp_file = os.path.join(disparity_path, f"{frameidx:06d}.npy")
    if not os.path.exists(disp_file):
        print(f"cannot find {disp_file}!")
        exit()

    disp = np.load(disp_file).astype(np.float)
    disp[disp == -np.finfo(np.float16).max] = np.inf
    disp = -disp if isleft else disp
    disp = cv2.resize(disp, (640, 360), interpolation=cv2.INTER_NEAREST)
    disp_max = max(disp_max, disp[disp != np.inf].max())
    disparity_list.append(disp)

img = np.stack(disparity_list, axis=0)[:, 180, :] / (disp_max)
matplot_img(img)
save_img(img, "ft_disptemp.png")

