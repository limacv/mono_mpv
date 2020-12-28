"""
This script will prefetch datas, post check datas and update *_valid and black_list
"""
import os
import sys
import time
import datetime
from glob import glob
import argparse
import multiprocessing as mp

sys.path.append("..")
from dataset.RealEstate10K import RealEstate10K_Img, RealEstate10K_root

parser = argparse.ArgumentParser()
parser.add_argument('--work_id', dest='work_id', type=int, help="index of num_worker")
parser.add_argument('--num_worker', dest='num_worker', type=int, default=6, help="total number of Nodes used")
parser.add_argument('--inner_worker', dest='inner_worker', type=int, default=2, help="number of multi-process used")
parser.add_argument('--train_str', dest='train_str', default='train', help="choose train or test")
parser.add_argument('--try_black_list', dest='try_black_list', default=False, type=bool,
                    help="whether to try to fetch black_list")
parser.add_argument('--try_post_black_list', dest='try_post_black_list', default=False, type=bool,
                    help="whether to try to fetch post_black_list")
parser.add_argument('--try_success', dest='try_success', default=False, type=bool,
                    help="whether to try to fetch successs")
args = parser.parse_args()


success_file = os.path.join(RealEstate10K_root, f"{args.train_str}_valid_{args.work_id}X{args.num_worker}.txt")
if not os.path.exists(success_file):
    open(success_file, 'a').close()
black_list_name = os.path.join(RealEstate10K_root, f"black_list_{args.work_id}X{args.num_worker}.txt")
if not os.path.exists(black_list_name):
    open(black_list_name, 'a').close()
postcheck_black_list_name = os.path.join(RealEstate10K_root, f"post_check_black_list_{args.work_id}X{args.num_worker}.txt")
if not os.path.exists(postcheck_black_list_name):
    open(postcheck_black_list_name, 'a').close()


def callback(ret):
    global failed_num
    status, basename = ret
    if status == "Success":
        if basename in black_lines:
            print(f"    {basename} in black_list, but it's now success, please manually remove since this"
                  f"bearly happened")

        if basename in success_basename:
            print(f"    {basename} already exists", flush=True)
        else:
            print(f"    {basename} add to success list", flush=True)
            with open(success_file, 'a') as file:
                file.writelines(basename + '\n')
            success_basename.add(basename)
    else:
        failed_num += 1
        if basename in success_basename:
            print(f"    {basename} failed but in the success list, it's add to black list now", flush=True)
        else:
            print(f"    {basename} failed, add to black list", flush=True)

        black_list = black_list_name if status == "Failed" else postcheck_black_list_name
        with open(black_list, 'a') as file:
            file.writelines(basename + '\n')
        black_lines.add(basename)

    print(f"{len(success_basename)}/{len(file_bases)} ({failed_num} failed): at {datetime.datetime.now()}", flush=True)


def propress_one(basename):
    # check validaity
    starttime = time.time()
    trainset = RealEstate10K_Img(args.train_str == 'train', black_list=False)
    try:
        ret = trainset.getitem_bybase(basename)
    except Exception as e:
        ret = None
        print(e, flush=True)

    if ret is not None:
        ret = trainset.post_check(basename)
        if ret:
            ret = "Success"
        else:
            print(f"    {basename} post check failed", flush=True)
            ret = "Post_Failed"

    else:
        ret = "Failed"

    print(f"    {basename} finished with status {ret} using {(time.time() - starttime):.1f}s")
    # now we know that this is a successful one if ret is not None
    return ret, basename


if __name__ == "__main__":

    failed_num = 0
    print(f"synchonize with {success_file}", flush=True)
    file_bases = glob(os.path.join(RealEstate10K_root, args.train_str, "*.txt"))
    file_bases = sorted([os.path.basename(fb).split('.')[0] for fb in file_bases])
    file_bases = file_bases[args.work_id::args.num_worker]
    file_bases = set(file_bases)

    with open(success_file, 'r') as file:
        success_basename = file.readlines()
    with open(black_list_name, 'r') as file:
        black_lines = file.readlines()
    with open(postcheck_black_list_name, 'r') as file:
        post_black_lines = file.readlines()
    success_basename = {os.path.basename(line.strip('\n')).split('.')[0] for line in success_basename}
    black_lines = {os.path.basename(line.strip('\n')) for line in black_lines}
    post_black_lines = {os.path.basename(line.strip('\n')) for line in post_black_lines}

    print(f"worker{args.work_id}: totally {len(file_bases)} {args.train_str} video", flush=True)
    if not args.try_black_list:
        file_bases = file_bases - black_lines
        print(f"    after remove the black_lines, got totally {len(file_bases)} videos")
    if not args.try_post_black_list:
        file_bases = file_bases - post_black_lines
        print(f"    after remove the post_black_lines, got totally {len(file_bases)} videos")
    if not args.try_success:
        file_bases = file_bases - success_basename
        print(f"    after remove the already successful one, got totally {len(file_bases)} videos")

    pool = mp.Pool(processes=args.inner_worker)
    for base in file_bases:
        pool.apply_async(propress_one, (base, ), callback=callback)

    pool.close()
    pool.join()

    print("___________finished____________\n\n", flush=True)
