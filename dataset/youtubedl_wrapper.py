"""
wrapper to download video use youtube-dl
"""

import os
import subprocess
import sys
from . import youtubedl_path, is_DEBUG


def run_youtubedl(link: str, savepath: str):
    """
    may throw error
    """
    def printdbg(_st):
        if is_DEBUG:
            print(_st)

    youtube_args = [
        youtubedl_path,
        '-q',
        '--abort-on-error',
        link,
        '-o', os.path.join(savepath, "video.mp4")
    ]
    feat_output = (subprocess.check_output(youtube_args, universal_newlines=True))

    printdbg(feat_output)
    printdbg('Video Download Success!')
