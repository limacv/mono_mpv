import os
import sys
import numpy as np
import time
import datetime
import shutil
from glob import glob

sys.path.append("..")
from dataset.RealEstate10K import RealEstate10K_Img, RealEstate10K_root


def main(istrain=True):
    trainset = RealEstate10K_Img(istrain)
    file_list_from_tmp = glob(os.path.join(RealEstate10K_root, f"{trainset.trainstr}tmp/*"))
    file_list_from_tmp = {os.path.join(RealEstate10K_root, trainset.trainstr, f"{os.path.basename(_f)}.txt")
                          for _f in file_list_from_tmp}
    new_trainset_list = []

    print(f"detect {len(file_list_from_tmp)} files from {trainset.trainstr}set", flush=True)
    for name in file_list_from_tmp:
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
        if ret is not None:
            print(f"{name} is ok, add to file", flush=True)
            new_trainset_list.append(name)

    print(f"finished checking! start writing ***_valid.tmp.txt file!!")
    file_path_tmp = os.path.join(RealEstate10K_root, f"{trainset.trainstr}_valid.tmp.txt")
    with open(file_path_tmp, 'w') as f:
        for line in new_trainset_list:
            f.writelines(line + "\n")
    print(f"successfully write {file_path_tmp}")


main(True)
main(False)
