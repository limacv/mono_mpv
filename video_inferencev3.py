from video_inference_cfg import *


path = "./log/checkpointsave/ablation00_svbase_r0_5.pth"
pipeline = smart_select_pipeline(path,
                                  force_pipelinename="fullv3")
ret_cfg = ""

if "StereoBlur" in video_path:
    saveprefix = "ZV3" + os.path.basename(path).split('.')[0] \
                 + os.path.basename(video_path).split('.')[0] + ret_cfg
elif "MannequinChallenge" in video_path:
    saveprefix = "ZV3" + os.path.basename(path).split('.')[0] \
                 + os.path.basename(os.path.dirname(video_path)).split('.')[0] + ret_cfg
else:
    saveprefix = "ZV3" + os.path.basename(path).split('.')[0] \
                 + os.path.basename(video_path).split('.')[0]
dispvideo_path = os.path.join(out_prefix, saveprefix + "_disparity.mp4")
newviewvideo_path = os.path.join(out_prefix, saveprefix + "_newview.mp4")
mpvout_path = os.path.join(out_prefix, saveprefix + ".mp4")

# ## ### #### ##### ###### ####### ######## ####### ###### ##### #### ### ## #

cap = cv2.VideoCapture(video_path)
mpvout = MPVWriter(mpvout_path)
netout = NetWriter(mpvout_path)
viewout = MyVideoWriter(newviewvideo_path)
dispout = MyVideoWriter(dispvideo_path)
disparity_list = []

frameidx = 0
with torch.no_grad():
    while True:
        print(f"\r{frameidx}", end='')
        ret, img = cap.read()
        if frameidx < framestart:
            frameidx += 1
            continue

        if not ret or frameidx > frameend:
            break
        hei, wid, _ = img.shape
        if wid > hei * 2:
            img = img[:, :wid // 2]
            wid //= 2

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, None, None, 0.5, 0.5)
        if (wid / hei) != (outputres[0] / outputres[1]) and reply_flag:
            reply_flag = False
            reply = input(f"original ratio {wid / hei:.2f} ({wid}x{hei}),"
                          f"the outputres' ratio {outputres[0] / outputres[1]:.2f} {outputres}, \n"
                          f"Do you want to continue? y/yes, \nor continue with original resolution? c")
            if 'c' in reply:
                outputres = (wid, hei)
            elif 'y' not in reply:
                exit()

        img = cv2.resize(img, outputres)
        hei, wid, _ = img.shape
        img_tensor = ToTensor()(img).cuda().unsqueeze(0)
        mpi = pipeline.infer_forward(img_tensor, ret_cfg=ret_cfg)

        if mpi is None:
            continue
        if isinstance(mpi, tuple):
            mpi, net = mpi

        depthes = make_depths(mpi.shape[1]).cuda()
        disparity = estimate_disparity_torch(mpi, depthes)
        disparity_list.append(disparity.cpu())

        if save_mpv:
            mpvout.write(mpi[0])

        if save_newview:
            img = renderto(mpi, newview_pose)
            viewout.write(img)

        if save_disparity:
            disparity = disparity_list[-1]
            visdisp = draw_dense_disp(disparity, 1)[:, :, ::-1]
            dispout.write(visdisp)
        frameidx += 1

    print("\n")
    del mpvout
    del netout
    del viewout
    del dispout
