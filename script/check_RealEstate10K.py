import os
import sys
import numpy as np
import time
import datetime
import shutil

sys.path.append("..")
from dataset.RealEstate10K import RealEstate10K, RealEstate10K_root

trainset = RealEstate10K(True, subset_byfile=True)
evalset = RealEstate10K(False, subset_byfile=True)

print(f"Checking trainset (len {len(trainset)})...", flush=True)
for i in range(len(trainset)):
    try:
        ret = trainset.getitem(i)
    except Exception as e:
        print(e)
        ret = None
    if ret is None:
        name = trainset.file_list[i]
        print(f"error in line{i}: {name}", flush=True)

print(f"Checking evalset (len {len(evalset)})...", flush=True)
for i in range(len(evalset)):
    try:
        ret = evalset.getitem(i)
    except Exception as e:
        print(e)
        ret = None
    if ret is None:
        name = evalset.file_list[i]
        print(f"error in line{i}: {name}", flush=True)
        break

