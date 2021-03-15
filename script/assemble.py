import os
from glob import glob
import sys
sys.path.append("../")
from evaluator import MetricCounter

root = "Z:\\tmp\\VisualEval"

item = "StereoVideo_3D-Photo_mydepth"

metrics = ['ssim', 'psnr', 'mse', 'lpips',
           'visflowepe', 'visflowmax', 'visflowmid',
           'flowcor_ssim', 'flowcor_psnr', 'flowcor_mse', 'flowcor_lpips',
           'disocc_ssim', 'disocc_psnr',
           'disp_abs_rel', 'disp_log10', 'disp_rmse', 'disp_sigma1', 'disp_sigma2', 'disp_sigma3',
           'temp_disp_o1', 'temp_disp_o2', 'temp_nvs_o1', 'temp_nvs_epe']

if __name__ == "__main__":
    # collect scores
    item_scores = []
    counter = MetricCounter()

    path = os.path.join(root, item)
    videonames = glob(os.path.join(path, "perscene_StereoBlur_*.txt"))  # existing videoname

    # wanted
    with open("Z:\\dataset\\StereoVideoFinalv3\\test_split2.txt", 'r') as f:
        lines = f.readlines()
    basenames_wanted = [line.strip('\n') for line in lines]
    videonames_wanted = [os.path.join(path, f"perscene_{bn}.txt") for bn in basenames_wanted]

    for videoname in videonames_wanted:
        name = os.path.basename(videoname)[9:-4]
        if not os.path.exists(videoname):
            videoname = os.path.join(os.path.dirname(videoname), name, os.path.basename(videoname))
        assert os.path.exists(videoname), f"{videoname} not exist"
        with open(videoname, 'r') as f:
            content = f.readlines()
        for metric in metrics:
            lines = [line for line in content if f" {metric} " in line]
            if len(lines) != 1:
                continue
            m, v, c = lines[0].split('|')
            v, c = float(v), float(c)
            counter.collect(metric, v * c, c)

    with open(os.path.join(path, "all_results.txt"), 'r') as f:
        lines = f.readlines()
        headerid = [idx for idx, line in enumerate(lines) if "+========" in line]
        header = lines[headerid[0]:headerid[1] + 1]
        with open(os.path.join(path, "split2.txt"), 'w') as fo:
            fo.writelines(header)
            fo.writelines(counter.make_table())

    # use first as reference, check video that is good in first
    for line in header:
        print(line, end='')
    print(counter.make_table())
