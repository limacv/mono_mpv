import os
from glob import glob

root = "Z:\\tmp\\VisualEval"

items = [
    "StereoVideo_ablation00_svbase_r0_5_MPINetv2_disp_img√",
    "StereoVideo_Ultly2ok_r0_4_auto_fullv4_hard_sigmoid",
]

if __name__ == "__main__":
    # collect scores
    item_scores = []
    for item in items:
        path = os.path.join(root, item)
        videonames = glob(os.path.join(path, "StereoBlur_HD*", "perscene_StereoBlur_*.txt"))
        scores = {}  # name -> dict {metric -> value}
        for videoname in videonames:
            table = {}
            name = os.path.basename(videoname)[9:-4]
            with open(videoname, 'r') as f:
                content = f.readlines()
            table["ssim"] = float([line for line in content if " ssim " in line][0].split('|')[1])
            table["psnr"] = float([line for line in content if " psnr " in line][0].split('|')[1])
            scores[name] = table

        item_scores.append(scores)

    # use first as reference, check video that is good in first
    goodlist = []
    nvsisgoodlist = []
    depisgoodlist = []
    for videoname in item_scores[0].keys():
        scores = [s[videoname] for s in item_scores]
        nvs = ["ssim", "psnr"]
        nvsisgood = True
        if scores[0]["ssim"] < scores[1]["ssim"]:
            nvsisgood = False
        if scores[0]["psnr"] < scores[1]["psnr"]:
            nvsisgood = False

        if nvsisgood:
            goodlist.append(videoname)

    print("good")
    for n in goodlist:
        print(n)
    print("\nonly nvs metric is good:")
    for n in nvsisgoodlist:
        print(n)
