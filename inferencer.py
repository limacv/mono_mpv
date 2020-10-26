import torch.nn as nn
import torch
from models.IRR_PWC import IRRPWCNet
from models.PWCNet import ORIPWCDCNet
from models.AR_PWCLite import ARPWCLite
from dataset.sintel_seq import SintelSeq
from models.ModelWithLoss import ModelandUnsuLoss
import matplotlib.pyplot as plt
import numpy as np
from main import select_module

model_name = "RAFTNet"
dataset_path = "D:\\MSI_NB\\source\\dataset\\Sintel\\" + "training\\final\\market_6"
check_point = "./log/" + model_name + "/0_sintel.ckpt"
visualize_frame = 3


def main():
    dataset = SintelSeq(dataset_path)

    model = select_module(model_name)
    model.load_state_dict(torch.load(check_point)["state_dict"])
    modelandloss = ModelandUnsuLoss(model, {}, {})

    frame0 = dataset.get_tensorframe(visualize_frame).unsqueeze(0).cuda()
    frame1 = dataset.get_tensorframe(visualize_frame+1).unsqueeze(0).cuda()
    flow, occ = modelandloss.infer_forward(frame0, frame1, occ_out=True)
    flow = flow[0].permute(1, 2, 0).cpu().numpy()
    occ = (occ[0] > 0.5).type(torch.uint8) * 255
    occ = torch.cat([occ, occ, occ], 0).permute(1, 2, 0).cpu().numpy()
    flowrgb = flow_to_png_middlebury(flow)
    rgb = np.vstack([flowrgb, occ])

    flowgt = dataset.get_gtflow(visualize_frame)
    gtrgb = flow_to_png_middlebury(flowgt)
    occgt = dataset.get_gtocc(visualize_frame)
    occgt = np.stack([occgt] * 3, -1).astype(np.uint8) * (gtrgb.max() - gtrgb.min()) + gtrgb.min()
    gtrgb = np.vstack([gtrgb, occgt])
    epe_map = np.linalg.norm(flowgt - flow, axis=-1)
    epe_map = np.minimum(epe_map, 5)
    
    plt.figure()
    plt.imshow(gtrgb)
    plt.figure()
    plt.imshow(rgb)
    plt.figure()
    plt.imshow(epe_map, cmap='jet')
    plt.show()


if __name__ == "__main__":
    main()
