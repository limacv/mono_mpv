import os
import sys
import numpy as np
import time
import datetime
import shutil

sys.path.append("..")
from dataset.RealEstate10K import RealEstate10K, RealEstate10K_root

np.random.seed(0)
process_train = True

train_str = "train" if process_train else "test"
success_file = os.path.join(RealEstate10K_root, f"{train_str}_valid.txt")
black_list_name = os.path.join(RealEstate10K_root, "black_list.txt")

print(f"synchonize with {success_file}")
with open(success_file, 'r') as file:
    lines = file.readlines()
with open(black_list_name, 'r') as file:
    black_lines = file.readlines()
lines = {line.strip('\n') for line in lines}
black_lines = {os.path.basename(line.strip('\n')) for line in black_lines}

trainset = RealEstate10K(process_train)

print(f"totally {len(trainset)} {train_str} video")

train_seq = np.random.permutation(len(trainset))
train_failed_num = 0
starttime = time.time()

for train_id in train_seq:
    print(f"{len(lines)}/{len(train_seq)} ({train_failed_num} failed): {(time.time() - starttime):.1f}s"
          f" at {datetime.datetime.now()}")
    starttime = time.time()
    try:
        ret = trainset.getitem(train_id)
    except Exception as e:
        ret = None
        print(e)
    name = trainset.file_list[train_id]
    if os.path.basename(name) in black_lines:
        ret = None

    if ret is not None:
        if name in lines:
            print(f"{os.path.basename(name)} already exists")
        else:
            print(f"{os.path.basename(name)} add to success list")
            with open(success_file, 'a') as file:
                file.writelines(name + '\n')
            lines.add(name)
    else:
        train_failed_num += 1
        if name in lines:
            print(f"{os.path.basename(name)} failed but in the success list, toggle rewrte whole success file")
            lines.remove(name)
            with open(success_file, 'w') as file:
                file.writelines([line + '\n' for line in lines])
            print("success file rewritten!")
        else:
            file_base = os.path.basename(name).split('.')[0]
            dir_base = os.path.join(trainset.tmp_root, file_base)
            print(f"{os.path.basename(name)} failed, deleting {dir_base}")
            shutil.rmtree(dir_base)
    print("-----------------------------------------")

