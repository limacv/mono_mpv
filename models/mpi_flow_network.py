"""
Models that can output flow / or multi plane flows
"""

import torch
import torch.nn as nn
import torch.nn.functional as torchf
from ._modules import *


class MPINet2In(nn.Module):
    """
    2 frames input, output flow, not good
    """
    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(6, 32, 7)
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

        self.fup2 = conv(192, 128, 3)
        self.fup2b = conv(128, 64, 3)
        self.fpost1 = conv(64, 64, 3)
        self.fpost2 = conv(64, 64, 3)
        self.fup1 = conv(64, 64, 3)
        self.fup1b = conv(64, 2 * self.num_layers, 3, isReLU=False)

        self.output = nn.Conv2d(64, self.num_layers - 1 + 3, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, imgs):
        down1 = self.down1b(self.down1(imgs))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        down5 = self.down5b(self.down5(self.down(down4)))
        down6 = self.down6b(self.down6(self.down(down5)))
        down7 = self.down7b(self.down7(self.down(down6)))
        x = self.up(self.mid2(self.mid1(self.down(down7))))
        x = self.up(self.up7b(self.up7(torch.cat(self.shapeto(x, down7), dim=1))))
        x = self.up(self.up6b(self.up6(torch.cat(self.shapeto(x, down6), dim=1))))
        x = self.up(self.up5b(self.up5(torch.cat(self.shapeto(x, down5), dim=1))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x1 = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x1 = self.post2(self.post1(torch.cat(self.shapeto(x1, down1), dim=1)))
        x1 = self.output(self.up1b(self.up1(x1)))
        x1 = self.output_bias(x1, self.num_layers)

        x2 = self.up(self.fup2b(self.fup2(torch.cat(self.shapeto(x, down2), dim=1))))
        x2 = self.fpost2(self.fpost1(x2))
        x2 = self.fup1b(self.fup1(x2))
        return x1, x2

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class MPFNet(nn.Module):
    """
    Multi plane flow input, output multiplane flow
    in: B x LayerNum x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(self.num_layers + 2, 64, 5)
        self.down1b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down5 = conv(256, 512, 3)
        self.down7 = conv(512, 512, 3)
        self.up7 = conv(512, 512, 3)
        self.up7b = conv(512, 512, 3)
        self.up5 = conv(1024, 512, 3)
        self.up4 = conv(768, 256, 3)
        self.up3 = conv(384, 128, 3)
        self.post1 = conv(192, 64, 3)
        self.output = nn.Conv2d(64, self.num_layers * 2, 3, padding=1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, mpf):
        down1 = self.down1b(self.down1(mpf))
        down3 = self.down3(self.down(down1))
        down4 = self.down4(self.down(down3))
        down5 = self.down5(self.down(down4))
        down7 = self.down7(self.down(down5))
        x = self.up(self.up7b(self.up7(down7)))
        x = self.up(self.up5(torch.cat(self.shapeto(x, down5), dim=1)))
        x = self.up(self.up4(torch.cat(self.shapeto(x, down4), dim=1)))
        x = self.up(self.up3(torch.cat(self.shapeto(x, down3), dim=1)))
        x = self.post1(torch.cat(self.shapeto(x, down1), dim=1))
        x = self.output(x)
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class AMPFNetAIn2D(nn.Module):
    """
    Appearance Multiplane flow with alpha input
    alpha input
    Multi plane flow input, output multiplane flow
    """

    def __init__(self, mpi_layers):
        self.num_layers = mpi_layers
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(self.num_layers + 2, 64, 5)
        self.down1b = conv(64, 64, 5)
        self.down2 = conv(64, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 384, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(256 + 128, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(128 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.up1 = conv(64 + 64, 64, 3)
        self.up1b = conv(64, 64, 3)
        self.output = nn.Conv2d(64, 2 * self.num_layers, 3, padding=1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, flow, alphas):
        batchsz, _, hei, wid = flow.shape
        if alphas.dim() == 5:
            alphas = alphas.squeeze(2)
        x = torch.cat([flow, alphas], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.output(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        return x.reshape(batchsz, self.num_layers, 2, hei, wid)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class ASPFNetAIn(nn.Module):
    """
    Appearance Multiplane flow with alpha input
    alpha input
    Multi plane flow input, output multiplane flow
    in: B x LayerNum x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self, mpi_layers):
        self.num_layers = mpi_layers
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(self.num_layers + 2, 32, 5)
        self.down1b = conv(32, 32, 5)
        self.down2 = conv(32, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 384, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(256 + 128, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(128 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.up1 = conv(64 + 32, 64, 3)
        self.up1b = conv(64, 32, 3)
        self.output = nn.Conv2d(32, 2, 3, padding=1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, flow, alphas):
        if alphas.dim() == 5:
            alphas = alphas.squeeze(2)
        x = torch.cat([flow, alphas], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.output(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class ASPFNetDIn(nn.Module):
    """
    Appearance Multiplane flow with disparity input
    alpha input
    Multi plane flow input, output single plane appearance flow
    in: B x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(2 + 1, 32, 5)
        self.down1b = conv(32, 32, 5)
        self.down2 = conv(32, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 384, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(256 + 128, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(128 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.up1 = conv(64 + 32, 32, 3)
        self.up1b = conv(32, 32, 3)
        self.output = nn.Conv2d(32, 2, 3, padding=1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, flow, disp):
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        x = torch.cat([flow, disp], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.output(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class ASPFNetWithMaskInOut(nn.Module):
    """
    Appearance Multiplane flow with disparity input
    alpha input
    Multi plane flow input, output single plane appearance flow
    in: B x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(2 + 1 + 1, 32, 5)
        self.down1b = conv(32, 64, 5)
        self.down2 = conv(64, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 384, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(256 + 128, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(128 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.up1 = conv(64 + 64, 32, 3)
        self.up1b = conv(32, 32, 3)
        self.output = nn.Conv2d(32, 2, 3, padding=1)
        self.outputma = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, flow, disp, mask=None):
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        if mask is None:
            mask = torch.zeros_like(disp)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        x = torch.cat([flow, disp, mask], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1)))
        return self.output(x), self.outputma(x)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
        nn.init.xavier_normal_(self.outputma[0].weight)
        nn.init.constant_(self.outputma[0].bias, 0)


class ASPFNetWithMaskOut(nn.Module):
    """
    Appearance Multiplane flow with disparity input
    alpha input
    Multi plane flow input, output single plane appearance flow
    in: B x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(2 + 1, 32, 5)
        self.down1b = conv(32, 64, 5)
        self.down2 = conv(64, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 384, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(256 + 128, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(128 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.up1 = conv(64 + 64, 32, 3)
        self.up1b = conv(32, 32, 3)
        self.output = nn.Conv2d(32, 2, 3, padding=1)
        self.outputma = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, flow, disp):
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        x = torch.cat([flow, disp], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1)))
        return self.output(x), self.outputma(x)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
        nn.init.xavier_normal_(self.outputma[0].weight)
        nn.init.constant_(self.outputma[0].bias, 0)


class AFNet_HR_netflowin(nn.Module):
    """
    Appearance Multiplane flow with disparity input
    alpha input
    Multi plane flow input, output single plane appearance flow
    in: B x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self, netcnl=6):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(netcnl + 2, 64, 7)
        self.down1b = conv(64, 64, 5)
        self.down2 = conv(64, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 256, 3)
        self.mid1 = conv(256, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 256, 256, 3)
        self.up4b = conv(256, 128, 3)
        self.up3 = conv(128 + 128, 128, 3)
        self.up3b = conv(128, 64, 3)
        self.up2 = conv(64 + 64, 64, 3)
        self.up2b = conv(64, 32, 3)
        self.up1 = conv(32 + 64, 32, 3)
        self.up1b = conv(32, 32, 3)
        self.output = nn.Conv2d(32, 2, 3, padding=1)
        self.outputma = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, net, flow):
        x = torch.cat([net, flow], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1)))
        return self.output(x), self.outputma(x)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
        nn.init.xavier_normal_(self.outputma[0].weight)
        nn.init.constant_(self.outputma[0].bias, 0)


class AFNet_HR_netflowinbig(nn.Module):
    """
    Appearance Multiplane flow with disparity input
    alpha input
    Multi plane flow input, output single plane appearance flow
    in: B x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(6 + 2, 64, 7)
        self.down1b = conv(64, 64, 5)
        self.down2 = conv(64, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 256, 3)
        self.down5 = conv(256, 256, 3)
        self.down5b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up5 = conv(384 + 384, 384, 3)
        self.up5b = conv(384, 256, 3)
        self.up4 = conv(256 + 256, 256, 3)
        self.up4b = conv(256, 128, 3)
        self.up3 = conv(128 + 128, 128, 3)
        self.up3b = conv(128, 64, 3)
        self.up2 = conv(64 + 64, 64, 3)
        self.up2b = conv(64, 32, 3)
        self.up1 = conv(32 + 64, 32, 3)
        self.up1b = conv(32, 32, 3)
        self.output = nn.Conv2d(32, 2, 3, padding=1)
        self.outputma = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, net, flow):
        x = torch.cat([net, flow], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        down5 = self.down5b(self.down5(self.down(down4)))
        x = self.up(self.mid2(self.mid1(self.down(down5))))
        x = self.up(self.up5b(self.up5(torch.cat(self.shapeto(x, down5), dim=1))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1)))
        return self.output(x), self.outputma(x)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
        nn.init.xavier_normal_(self.outputma[0].weight)
        nn.init.constant_(self.outputma[0].bias, 0)


class AFNet_HR_netflowimgin(nn.Module):
    """
    Appearance Multiplane flow with disparity input
    alpha input
    Multi plane flow input, output single plane appearance flow
    in: B x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(6 + 2 + 3, 64, 7)
        self.down1b = conv(64, 64, 5)
        self.down2 = conv(64, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 256, 3)
        self.mid1 = conv(256, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 256, 384, 3)
        self.up4b = conv(384, 256, 3)
        self.up3 = conv(256 + 128, 256, 3)
        self.up3b = conv(256, 128, 3)
        self.up2 = conv(128 + 64, 128, 3)
        self.up2b = conv(128, 64, 3)
        self.up1 = conv(64 + 64, 64, 3)
        self.up1b = conv(64, 64, 3)
        self.output = nn.Conv2d(64, 2, 3, padding=1)
        self.outputma = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, net, flow, img):
        x = torch.cat([net, flow, img], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1)))
        return self.output(x), self.outputma(x)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
        nn.init.xavier_normal_(self.outputma[0].weight)
        nn.init.constant_(self.outputma[0].bias, 0)


class AFNet_LR_netflowin(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(6 + 2, 64, 3)
        self.down1b = conv(64, 64, 3)
        self.down2 = conv(64, 64, 3)
        self.down2b = conv(64, 128, 3)
        self.down3 = conv(128, 128, 3)
        self.down3b = conv(128, 256, 3)
        self.mid1 = conv(256, 256, 3)
        self.mid2 = conv(256, 256, 3)
        self.up3 = conv(256 + 256, 256, 3)
        self.up3b = conv(256, 256, 3)
        self.up2 = conv(256 + 128, 128, 3)
        self.up2b = conv(128, 64, 3)
        self.up1 = conv(64 + 64, 64, 3)
        self.up1b = conv(64, 32, 3)
        self.output = nn.Conv2d(32, 2, 3, padding=1)
        self.outputma = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, net, flow):
        x = torch.cat([net, flow], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1)))
        return self.output(x), self.outputma(x)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
        nn.init.xavier_normal_(self.outputma[0].weight)
        nn.init.constant_(self.outputma[0].bias, 0)


class AFNet_LR_netflownetin(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.pre1 = conv(6, 64, 3)
        self.pre1b = conv(64, 64, 3)
        self.down1 = conv(64 + 128, 128, 3)
        self.down1b = conv(128, 128, 3)
        self.down2 = conv(128, 128, 3)
        self.down2b = conv(128, 256, 3)
        self.down3 = conv(256, 256, 3)
        self.down3b = conv(256, 256, 3)
        self.mid1 = conv(256, 256, 3)
        self.mid2 = conv(256, 256, 3)
        self.up3 = conv(256 + 256, 384, 3)
        self.up3b = conv(384, 256, 3)
        self.up2 = conv(256 + 256, 256, 3)
        self.up2b = conv(256, 128, 3)
        self.up1 = conv(128 + 128, 128, 3)
        self.up1b = conv(128, 64, 3)
        self.output = nn.Conv2d(64, 2, 3, padding=1)
        self.outputma = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, net, flow_net):
        net_feat = self.pre1b(self.pre1(net))
        feat_in = torch.cat([net_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(feat_in))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1)))
        return self.output(x), self.outputma(x)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
        nn.init.xavier_normal_(self.outputma[0].weight)
        nn.init.constant_(self.outputma[0].bias, 0)


class SceneFlowNet(nn.Module):
    """
    Flow.xy / disparity in and Flow.z out
    """
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(2 + 1, 32, 5)
        self.down1b = conv(32, 32, 5)
        self.down2 = conv(32, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 384, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(256 + 128, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(128 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.up1 = conv(64 + 32, 32, 3)
        self.up1b = conv(32, 32, 3)
        self.output = nn.Conv2d(32, 1, 3, padding=1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, flow, disp):
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        x = torch.cat([flow, disp], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.output(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class SceneFlowNet_alpha(nn.Module):
    """
    Flow.xy / disparity in and Flow.z out
    """
    def __init__(self, num_plane):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(2 + num_plane, 64, 5)
        self.down1b = conv(64, 64, 5)
        self.down2 = conv(64, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 384, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(256 + 128, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(128 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.up1 = conv(64 + 64, 64, 3)
        self.up1b = conv(64, 32, 3)
        self.output = nn.Conv2d(32, 1, 3, padding=1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, flow, alpha):
        if alpha.dim() == 3:
            alpha = alpha.unsqueeze(1)
        x = torch.cat([flow, alpha], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.output(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class SceneFlowNet_img(nn.Module):
    """
    Flow.xy / disparity / rgb in and Flow.z out
    """
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(2 + 1 + 3, 32, 5)
        self.down1b = conv(32, 32, 5)
        self.down2 = conv(32, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up4 = conv(384 + 384, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(256 + 128, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(128 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.up1 = conv(64 + 32, 32, 3)
        self.up1b = conv(32, 32, 3)
        self.output = nn.Conv2d(32, 1, 3, padding=1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, flow, disp, img):
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        x = torch.cat([flow, disp, img], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.output(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
