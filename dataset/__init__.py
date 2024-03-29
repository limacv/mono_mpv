import os
import sys

COMPUTER_NAME = "xgpu"
RealEstate10K_root = "/home/lmaag/xgpu-scratch/mali_data/RealEstate10K/"
RealEstate10K_skip_framenum = 3

MannequinChallenge_root = "/home/lmaag/xgpu-scratch/mali_data/MannequinChallenge/"
MannequinChallenge_skip_framenum = 1

# StereoBlur_root = "/home/lmaag/xgpu-scratch/mali_data/StereoBlur_processed"
StereoBlur_root = "/home/lmaag/xgpu-scratch/mali_data/StereoBlur_test/"
StereoBlur_use_saved_disparity = True

StereoVideo_version = "v3"
StereoVideo_root = f"/home/lmaag/xgpu-scratch/mali_data/StereoVideoFinal{StereoVideo_version}/"
StereoVideo_test_pct = 0.1

NvidiaNovelView_root = "/home/lmaag/xgpu-scratch/mali_data/NvidiaNovelViewSynthesis/datas/"

# OutputSize = (512j, 512 + 128 * 2)  # (h, w)
OutputSize = (448, 448)  # (h, w)
OutputSize_test = (360, 640)
LBTC_OutputSize = (192, 192)
MaxPointNumberForce = 2000
colmap_path = "/home/lmaag/xgpu-scratch/mali_data/Programs/colmap_exec/colmap"
youtubedl_path = "youtube-dl"

is_DEBUG = False  # (getattr(sys, 'gettrace', None) is not None)

if 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "MSI":
    COMPUTER_NAME = "msi"
    colmap_path = "D:\\MSI_NB\\source\\util\\COLMAP-3.6-exe\\COLMAP.bat"
    youtubedl_path = "C:\\Users\\MSI_NB\\AppData\\Roaming\\Python\\Python37\\Scripts\\youtube-dl.exe"
    OutputSize = (184, 296)
    RealEstate10K_root = "Z:\\dataset\\RealEstate10K_test\\"
    MannequinChallenge_root = "Z:\\dataset\\MannequinChallenge"

    StereoBlur_root = "D:\\MSI_NB\\source\\data\\StereoBlur_test"
    StereoBlur_use_saved_disparity = True
    StereoVideo_root = f"Z:\\dataset\\StereoVideoFinalv3\\"

    NvidiaNovelView_root = "Z:\\dataset\\NvidiaNovelViewSynthesis\\datas\\"

    if not os.path.exists(RealEstate10K_root):  # not pluged
        RealEstate10K_root = "D:\\MSI_NB\\source\\data\\RealEstate10K\\"
        MannequinChallenge_root = "D:\\MSI_NB\\source\\data\\MannequinChallenge"
        StereoVideo_root = f"D:\\MSI_NB\\source\\data\\StereoVideoFinalv3"

    is_DEBUG = True

elif "LOGNAME" in os.environ.keys() and os.environ["LOGNAME"] == 'jrchan':
    COMPUTER_NAME = "jrchan"
    RealEstate10K_root = "/home/jrchan/MALi/dataset/RealEstate10K_test"
    MannequinChallenge_root = "/home/jrchan/MALi/dataset/MannequinChallenge"
    StereoBlur_root = "/home/jrchan/MALi/dataset/StereoBlur_test"
    StereoBlur_use_saved_disparity = True
    StereoVideo_root = f"/home/jrchan/MALi/dataset/StereoVideoFinal{StereoVideo_version}"
