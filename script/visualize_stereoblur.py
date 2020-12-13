from dataset.StereoBlur import *
from glob import glob
from models.flow_utils import *
from models.mpi_utils import *

count = 0
videos = glob(os.path.join(StereoBlur_root, "test", "*.mp4"))

flow_estim = RAFTNet(False)
flow_estim.eval()
flow_estim.cuda()
state_dict = torch.load(RAFT_path["sintel"], map_location='cpu')
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
flow_estim.load_state_dict(state_dict)

for videofile in videos:
    cap = cv2.VideoCapture(videofile)
    framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter()
    out1 = cv2.VideoWriter()

    imglast = None
    flow_last = None
    for idx in range(framenum):
        ret, img = cap.read()
        if not (ret and len(img) > 0):
            print(f"{videofile}: cannot read frame {idx}, which is said to have {framenum} frames")
        if idx > 120:
            break
        # count += 1
        # if count > 10:
        #     count = 0
        # else:
        #     continue
        # read and split the left and right view
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hei, wid, _ = img.shape

        wid //= 2
        imgl, imgr = img[:, :wid], img[:, wid:]

        disp, uncertainty = compute_disparity_uncertainty(imgl, imgr)
        disp = (disp * (255 / 100)).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_HOT)
        uncertainty = (uncertainty * 255).astype(np.uint8)
        uncertainty = cv2.applyColorMap(uncertainty, cv2.COLORMAP_JET)
        # cv2.imshow("disp", disp)
        # cv2.imshow("uncertainty", uncertainty)
        # imgl = cv2.resize(imgl, None, None, 0.5, 0.5)
        # hei, wid, cnl = imgl.shape
        # img = ToTensor()(imgl).unsqueeze(0).cuda()
        # if imglast is not None:
        #     with torch.no_grad():
        #         if flow_last is not None:
        #             flow_last = downflow8(flow_last)
        #             flow_last = forward_scatter(flow_last, flow_last)
        #         flow = flow_estim(imglast, img, flow_last)
        #         flow_last = flow
        #     flowvis = flow_to_png_middlebury(flow[0].cpu().numpy())
        # warp_last = warp_flow(imglast, flow)
        # diff = (warp_last - img).norm(dim=1)
        # cv2.imshow("flow", (warp_last * 255).type(torch.uint8)[0].permute(1, 2, 0).cpu().numpy())
        # cv2.imshow("flow", (diff * 255).type(torch.uint8)[0].cpu().numpy())
        print(f'\r{idx}', end='')
        if out.isOpened():
            out.write(disp)
            out1.write(uncertainty)
        else:
            if os.path.exists(videofile.replace('\\test\\', '\\visdisp\\')):
                break
            else:
                out.open(videofile.replace('\\test\\', '\\visdisp\\'),  828601953, 20., (wid, hei), True)
                out1.open(videofile.replace('\\test\\', '\\visdisp\\unce_'),  828601953, 20., (wid, hei), True)
        imglast = img

    out.release()
