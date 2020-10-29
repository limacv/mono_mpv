import os
import sys

# the settings for debug
RealEstate10K_root = "/scratch/PI/psander/mali_data/RealEstate10K/"
colmap_path = "D:\\MSI_NB\\source\\maybeUseful\\COLMAP-3.6-exe\\COLMAP.bat"

if 'COMPUTERNAME' in os.environ.keys() and os.environ['COMPUTERNAME'] == "MSI":
    RealEstate10K_root = "D:\\MSI_NB\\source\\data\\RealEstate10K\\"

is_DEBUG = (getattr(sys, 'gettrace', None) is not None)
