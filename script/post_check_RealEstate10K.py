"""
This file will update test_valid.txt and train_valid.txt (by computing ***_valid.new.txt, will need manually replace)
"""

import os
import sys
import numpy as np
import time
import datetime
import shutil

sys.path.append("..")
from dataset.RealEstate10K import RealEstate10K_Img, RealEstate10K_root


def main(istrain=True, check_black_list=False):
    trainset = RealEstate10K_Img(istrain, black_list=False)
    trainset_list = list(set(trainset.file_list))
    new_trainset_list = []
    new_black_list = set()

    # with open(os.path.join(RealEstate10K_root, "black_list.txt")) as f:
    #     trainset_list = [trainset.txtpath2basename(l) for l in f.readlines()]
    print(f"this process will generate .val.txt")
    print(f"Checking {trainset.name} (len {len(trainset_list)})...", flush=True)
    for i in range(len(trainset_list)):
        name = trainset_list[i]
        try:
            ret = trainset.pre_fetch_bybase(name)
            if not ret:
                raise RuntimeError(f"pre fetch error in line {i}: {name}")

            ret = trainset.post_check(name, verbose=True)
            if not ret:
                raise RuntimeError(f"post check error in line {i}: {name}")

            ret = trainset.getitem_bybase(name)
        except Exception as e:
            print(e)
            ret = None

        # if ret is not None:
        #     timestamp = trainset._curtimestamps
        #     timestamp = [len(str(int(_t))) for _t in timestamp]
        #     timestamp = np.array(timestamp)
        #     timestamp = np.sum(np.abs(timestamp - timestamp[0]))
        #     if timestamp > 0:  # means have error
        #         os.remove(trainset._curvideo_trim_path)
        #         print(f"line {i}: {name} is not correct, removing {trainset._curvideo_trim_path}", flush=True)
        #         ret = None

        if ret is None:
            print(f">error in line{i}: {name}, remove from list", flush=True)
            new_black_list.add(name)
        else:
            new_trainset_list.append(name)
        print(f"Current Accpt Rate: {len(new_trainset_list)}/{i+1} ({(100 * len(new_trainset_list) / (i+1)):2f}%)")

    if len(new_trainset_list) == len(trainset_list):
        print("===================================================\n"
              "=========Congradulation!, it's all correct=========\n"
              "===================================================\n")
        return
    print(f"finished checking! start writing file!!")
    file_path_tmp = os.path.join(RealEstate10K_root, f"{trainset.trainstr}_valid.new.txt")
    with open(file_path_tmp, 'w') as f:
        for line in new_trainset_list:
            f.writelines(line + "\n")
    print(f"successfully write {file_path_tmp}")

    if check_black_list:
        print(f"start writing blacklist!!")
        black_list_oldfile = os.path.join(RealEstate10K_root, "black_list.txt")
        if os.path.exists(black_list_oldfile):
            with open(black_list_oldfile, 'r') as f:
                lines = f.readlines()
            old_black_list = {trainset.txtpath2basename(line) for line in lines}
            new_black_list = new_black_list.union(old_black_list)
        black_list_newfile = os.path.join(RealEstate10K_root, "black_list.new.txt")
        with open(black_list_newfile, 'w') as f:
            for line in new_black_list:
                f.writelines(line + "\n")
        print(f"successfully write {black_list_newfile}")

    print(f"please manually replace the _valid.txt to _valid.new.txt")


main(True, True)
main(False, False)
