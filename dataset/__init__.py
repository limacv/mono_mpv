import os
import sys

RealEstate10K_root = "/scratch/PI/psander/mali_data/RealEstate10K/"
RealEstate10K_skip_framenum = 3
MannequinChallenge_root = "/scratch/PI/psander/mali_data/MannequinChallenge/"
MannequinChallenge_skip_framenum = 1

WSVD_root = "/home/lmaag/xgpu-scratch/mali_data/WSVD_processed/"
# StereoBlur_root = "/home/lmaag/xgpu-scratch/mali_data/StereoBlur_processed"
StereoBlur_root = "/home/lmaag/xgpu-scratch/mali_data/StereoBlur_test"
StereoBlur_use_saved_disparity = True

# OutputSize = (512j, 512 + 128 * 2)  # (h, w)
OutputSize = (384, 512)  # (h, w)
colmap_path = "colmap"

if 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "MSI":
    RealEstate10K_root = "D:\\MSI_NB\\source\\data\\RealEstate10K\\"
    colmap_path = "D:\\MSI_NB\\source\\util\\COLMAP-3.6-exe\\COLMAP.bat"
    OutputSize = (200, 300)
    WSVD_root = "D:\\MSI_NB\\source\\data\\WSVD_processed"
    # StereoBlur_root = "D:\\MSI_NB\\source\\data\\StereoBlur_processed"
    StereoBlur_root = "D:\\MSI_NB\\source\\data\\StereoBlur_test"
    StereoBlur_use_saved_disparity = True
    MannequinChallenge_root = "D:\\MSI_NB\\source\\data\\MannequinChallenge"

elif 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "DESKTOP-FCSORVT":
    RealEstate10K_root = "D:\\dataset\\RealEstate10K\\"
    colmap_path = "C:\\Users\\86712\\source\\COLMAP-3.6-windows\\COLMAP.bat"
    MannequinChallenge_root = "D:\\dataset\\MannequinChallenge\\"

elif "LOGNAME" in os.environ.keys() and os.environ["LOGNAME"] == 'jrchan':
    RealEstate10K_root = "./haven't specified in dataset.__init__.py"
    MannequinChallenge_root = "./haven't specified in dataset.__init__.py"
    # StereoBlur_root = "/home/jrchan/MALi/dataset/StereoBlur_processed"
    StereoBlur_root = "/home/jrchan/MALi/dataset/StereoBlur_test"
    StereoBlur_use_saved_disparity = True

is_DEBUG = False  # (getattr(sys, 'gettrace', None) is not None)
