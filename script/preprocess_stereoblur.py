import os
import shutil
import tarfile
import subprocess
from glob import glob

ffmpeg_exe = "C:\\Users\\86712\\source\\ffmpeg\\bin\\ffmpeg.exe"
StereoBlur_root = "D:\\dataset\\StereoBlur\\"
StereoBlur_out_root = "D:\\dataset\\StereoBlur_processed\\"

file_list = glob(os.path.join(StereoBlur_root, "HD*.tar.gz"))
for file_name in file_list:
    file_base = os.path.basename(file_name).split('.')[0]
    file = tarfile.open(file_name, 'r:gz')
    file.extractall(StereoBlur_root)

    extrachpath = os.path.join(StereoBlur_root, file_base)
    image_left_path = os.path.join(extrachpath, "image_left")
    image_right_path = os.path.join(extrachpath, "image_right")

    converter_args = [
        ffmpeg_exe,
        '-i', os.path.join(image_left_path, "%04d.png"),
        '-i', os.path.join(image_right_path, "%04d.png"),
        '-filter_complex', "hstack",
        '-c:v', 'libx264',
        os.path.join(StereoBlur_out_root, f"{file_base}.mp4")
    ]

    converter_output = (subprocess.check_output(converter_args))
    print(f"removing extracted folder {file_base}")
    shutil.rmtree(extrachpath)
