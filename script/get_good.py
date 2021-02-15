import os
from glob import glob

root = "/home/lmaag/xgpu-scratch/mali_data/Visual/"

items = [
    "StereoVideo_V52setcnn_adapts_122429_r0",
    "StereoVideo_mpinet_ori",
    "StereoVideo_raSV_scratch_adapts_122129_r0",
]

if __name__ == "__main__":
    # collect scores
    item_scores = []
    for item in items:
        path = os.path.join(root, item)
        videonames = glob(os.path.join(path, "perscene_StereoBlur_*.txt"))
        scores = {}  # name -> dict {metric -> value}
        for videoname in videonames:
            table = {}
            name = os.path.basename(videoname)[9:-4]
            with open(videoname, 'r') as f:
                content = f.readlines()
            startidx = [i for i in range(len(content)) if "ssim" in content[i]][0]
            table["ssim"] = float(content[startidx].split('|')[1])
            table["psnr"] = float(content[startidx+1].split('|')[1])
            table["mse"] = float(content[startidx+2].split('|')[1])
            table["disp_mae"] = float(content[startidx+3].split('|')[1])
            table["disp_msle"] = float(content[startidx+4].split('|')[1])
            table["disp_goodpct"] = float(content[startidx+5].split('|')[1])
            scores[name] = table

        item_scores.append(scores)

    # use first as reference, check video that is good in first
    goodlist = []
    nvsisgoodlist = []
    depisgoodlist = []
    for videoname in item_scores[0].keys():
        scores = [s[videoname] for s in item_scores]
        nvs = ["ssim", "psnr", "mse"]
        dep = ["disp_mae", "disp_msle", "disp_goodpct"]
        nvsisgood = True
        if scores[0]["ssim"] < scores[2]["ssim"] or scores[0]["ssim"] < scores[2]["ssim"]:
            nvsisgood = False
        if scores[0]["psnr"] < scores[2]["psnr"] or scores[0]["psnr"] < scores[2]["psnr"]:
            nvsisgood = False
        if scores[0]["mse"] > scores[2]["mse"] or scores[0]["mse"] > scores[2]["mse"]:
            nvsisgood = False
        depisgood = True
        if scores[0]["disp_mae"] > scores[2]["disp_mae"] or scores[0]["disp_mae"] > scores[2]["disp_mae"]:
            nvsisgood = False
        if scores[0]["disp_msle"] > scores[2]["disp_msle"] or scores[0]["disp_msle"] > scores[2]["disp_msle"]:
            nvsisgood = False
        if scores[0]["disp_goodpct"] < scores[2]["disp_goodpct"] or scores[0]["disp_goodpct"] < scores[2]["disp_goodpct"]:
            nvsisgood = False

        if nvsisgood and depisgood:
            goodlist.append(videoname)
        elif not nvsisgood:
            depisgoodlist.append(videoname)
        elif not depisgood:
            nvsisgoodlist.append(videoname)

    print("good")
    for n in goodlist:
        print(n)
    print("\nonly nvs metric is good:")
    for n in nvsisgoodlist:
        print(n)
    print("\nonly dep metric is good:")
    for n in depisgoodlist:
        print(n)
