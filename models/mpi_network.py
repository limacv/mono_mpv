import torch
import torch.nn as nn
import torch.nn.functional as torchf
from ._modules import *


class MPINet(nn.Module):
    def __init__(self, mpi_layers):
        super().__init__()
        self.mpi_layers = mpi_layers
        self.down = nn.MaxPool2d(2)
        self.up = up
        self.down1 = conv(3, 32, 7)
        self.down1b = conv(32, 32, 7)
        self.down2 = conv(32, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 256, 3)
        self.down5 = conv(256, 512, 3)
        self.down5b = conv(512, 512, 3)
        self.down6 = conv(512, 512, 3)
        self.down6b = conv(512, 512, 3)
        self.down7 = conv(512, 512, 3)
        self.down7b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up7 = conv(1024, 512, 3)
        self.up7b = conv(512, 512, 3)
        self.up6 = conv(1024, 512, 3)
        self.up6b = conv(512, 512, 3)
        self.up5 = conv(1024, 512, 3)
        self.up5b = conv(512, 512, 3)
        self.up4 = conv(768, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(384, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(192, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.post1 = conv(96, 64, 3)
        self.post2 = conv(64, 64, 3)
        self.up1 = conv(64, 64, 3)
        self.up1b = conv(64, 64, 3)
        self.output = nn.Conv2d(64, self.mpi_layers - 1 + 3, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    def forward(self, img):
        down1 = self.down1b(self.down1(img))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        down5 = self.down5b(self.down5(self.down(down4)))
        down6 = self.down6b(self.down6(self.down(down5)))
        down7 = self.down7b(self.down7(self.down(down6)))
        x = self.up(self.mid2(self.mid1(self.down(down7))))
        x = self.up(self.up7b(self.up7(torch.cat([x, down7], dim=1))))
        x = self.up(self.up6b(self.up6(torch.cat([x, down6], dim=1))))
        x = self.up(self.up5b(self.up5(torch.cat([x, down5], dim=1))))
        x = self.up(self.up4b(self.up4(torch.cat([x, down4], dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat([x, down3], dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat([x, down2], dim=1))))
        x = self.post2(self.post1(torch.cat([x, down1], dim=1)))
        x = self.output(self.up1b(self.up1(x)))
        x = self.output_bias(x, self.mpi_layers)
        return (torch.tanh(x) + 1.) / 2.0

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


def infer_mpi(model: MPINet, img: torch.Tensor):
    """
    img: tensor of shape [B, H, W, 3] or [H, W, 3]
    return: tensor of shape [B, mpi_layer, 4, H, W]
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    batchsz, chnl, height, width = img.shape
    output = model(img)  # output.shape == B, cnl, H, W
    alpha = output[:, :-3, :, :]  # alpha.shape == B, 31, H, W
    alpha = torch.cat([torch.ones([batchsz, 1, height, width]).type_as(alpha), alpha], dim=1)
    blend = torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                       torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    blend = blend.unsqueeze(2)
    layer_rgb = blend * img.unsqueeze(1) + (-blend + 1.) * output[:, -3:, :, :].unsqueeze(1)
    layers = torch.cat([layer_rgb, alpha.unsqueeze(2)], dim=2)
    return layers


if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import estimate_disparity_torch
    model = MPINet(32)
    model.initial_weights()
    model.load_state_dict(torch.load("../weights/mpinet_ori.pth"))
    model.cuda()

    img = cv2.imread("../weights/input.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hei, wid, _ = img.shape

    img = (img / 255).astype(np.float32)
    img_tensor = torch.tensor(img).cuda().permute(2, 0, 1)

    out = infer_mpi(model, img_tensor)

    disparity = estimate_disparity_torch(out)
    plt.imshow(disparity.squeeze(0).detach().cpu().numpy())
    plt.show()
