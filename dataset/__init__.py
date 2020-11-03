import os
import sys

# the settings for debug
RealEstate10K_root = "/scratch/PI/psander/mali_data/RealEstate10K/"
colmap_path = "colmap"

if 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "MSI":
    RealEstate10K_root = "D:\\MSI_NB\\source\\data\\RealEstate10K\\"
    colmap_path = "D:\\MSI_NB\\source\\maybeUseful\\COLMAP-3.6-exe\\COLMAP.bat"

elif 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "DESKTOP-FCSORVT":
    RealEstate10K_root = "D:\\dataset\\RealEstate10K\\"
    colmap_path = "C:\\Users\\86712\\source\\COLMAP-3.6-windows\\COLMAP.bat"

is_DEBUG = False  # (getattr(sys, 'gettrace', None) is not None)
