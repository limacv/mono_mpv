import os
import sys

RealEstate10K_root = "/home/lmaag/xgpu-scratch/mali_data/RealEstate10K/"
RealEstate10K_skip_framenum = 3
MannequinChallenge_root = "/home/lmaag/xgpu-scratch/mali_data/MannequinChallenge/"
MannequinChallenge_skip_framenum = 1

WSVD_root = "/home/lmaag/xgpu-scratch/mali_data/WSVD_processed/"
# StereoBlur_root = "/home/lmaag/xgpu-scratch/mali_data/StereoBlur_processed"
StereoBlur_root = "/home/lmaag/xgpu-scratch/mali_data/StereoBlur_test/"
StereoBlur_use_saved_disparity = True
StereoVideo_root = "/home/lmaag/xgpu-scratch/mali_data/StereoVideoGood/"

# OutputSize = (512j, 512 + 128 * 2)  # (h, w)
OutputSize = (384, 512)  # (h, w)
colmap_path = "/home/lmaag/xgpu-scratch/mali_data/Programs/colmap_exec/colmap"
youtubedl_path = "youtube-dl"

is_DEBUG = False  # (getattr(sys, 'gettrace', None) is not None)

if 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "MSI":
    colmap_path = "D:\\MSI_NB\\source\\util\\COLMAP-3.6-exe\\COLMAP.bat"
    youtubedl_path = "C:\\Users\\MSI_NB\\AppData\\Roaming\\Python\\Python37\\Scripts\\youtube-dl.exe"
    OutputSize = (200, 300)

    RealEstate10K_root = "Z:\\dataset\\RealEstate10K_test\\"
    if not os.path.exists(RealEstate10K_root):
        RealEstate10K_root = "D:\\MSI_NB\\source\\data\\RealEstate10K\\"
    WSVD_root = "D:\\MSI_NB\\source\\data\\WSVD_processed"
    # StereoBlur_root = "D:\\MSI_NB\\source\\data\\StereoBlur_processed"
    StereoBlur_root = "D:\\MSI_NB\\source\\data\\StereoBlur_test"
    StereoBlur_use_saved_disparity = True
    StereoVideo_root = "Z:\\dataset\\StereoVideoGood\\"
    if not os.path.exists(StereoVideo_root):
        StereoVideo_root = "D:\\MSI_NB\\source\\data\\StereoVideoGood"

    MannequinChallenge_root = "D:\\MSI_NB\\source\\data\\MannequinChallenge"
    is_DEBUG = True

elif 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "DESKTOP-FCSORVT":
    colmap_path = "C:\\Users\\86712\\source\\COLMAP-3.6-windows\\COLMAP.bat"
    youtubedl_path = "youtube-dl.exe"

    RealEstate10K_root = "D:\\dataset\\RealEstate10K\\"
    MannequinChallenge_root = "D:\\dataset\\MannequinChallenge\\"

elif "LOGNAME" in os.environ.keys() and os.environ["LOGNAME"] == 'jrchan':
    RealEstate10K_root = "./haven't specified in dataset.__init__.py"
    MannequinChallenge_root = "/home/jrchan/MALi/dataset/MannequinChallenge"
    # StereoBlur_root = "/home/jrchan/MALi/dataset/StereoBlur_processed"
    StereoBlur_root = "/home/jrchan/MALi/dataset/StereoBlur_test"
    StereoBlur_use_saved_disparity = True

