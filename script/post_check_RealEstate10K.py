import os
import sys
import numpy as np
import time
import datetime
import shutil

sys.path.append("..")
from dataset.RealEstate10K import RealEstate10K_Img, RealEstate10K_root


def main(istrain=True):
    trainset = RealEstate10K_Img(istrain, subset_byfile=True)
    trainset_list = list(set(trainset.file_list))
    new_trainset_list = []

    print(f"this process will generate .val.txt")
    print(f"Checking {trainset.name} (len {len(trainset_list)})...", flush=True)
    for i in range(len(trainset_list)):
        name = trainset_list[i]
        try:
            ret = trainset.getitem_bypath(name)
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
            print(f"error in line{i}: {name}, remove from list", flush=True)
        else:
            new_trainset_list.append(name)

    if len(new_trainset_list) == len(trainset_list):
        print("===================================================\n"
              "=========Congradulation!, it's all correct=========\n"
              "===================================================\n")
        return
    print(f"finished checking! start writing ***_valid.txt file!!")
    file_path_tmp = os.path.join(RealEstate10K_root, f"{trainset.trainstr}_valid.val.txt")
    with open(file_path_tmp, 'w') as f:
        for line in new_trainset_list:
            f.writelines(line + "\n")
    print(f"successfully write {file_path_tmp}")


main(True)
main(False)
