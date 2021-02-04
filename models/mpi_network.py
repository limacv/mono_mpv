"""
models that only output multiplane images (may take multiple input)
"""
import torch
import torch.nn as nn
import torch.nn.functional as torchf
import numpy as np
import torchvision
from ._modules import *


class MPINet(nn.Module):
    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
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
        self.output = nn.Conv2d(64, self.num_layers - 1 + 3, 3, padding=1)
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
        x = self.output_bias(x, self.num_layers)
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class MPINetv2(nn.Module):
    """
    MPINet that takes in arbitrary shape image
    """

    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
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
        self.output = nn.Conv2d(64, self.num_layers - 1 + 3, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img):
        down1 = self.down1b(self.down1(img))
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
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.post2(self.post1(torch.cat(self.shapeto(x, down1), dim=1)))
        x = self.output(self.up1b(self.up1(x)))
        x = self.output_bias(x, self.num_layers)
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class MPI_FlowGrad(nn.Module):
    """
    Takes Flow gradient in (|u / dx| + |v / dx|, |u / dy| + |v / dy|)
    """

    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(3 + 2, 32, 7)
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
        self.output = nn.Conv2d(64, self.num_layers - 1 + 3, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img):
        down1 = self.down1b(self.down1(img))
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
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.post2(self.post1(torch.cat(self.shapeto(x, down1), dim=1)))
        x = self.output(self.up1b(self.up1(x)))
        x = self.output_bias(x, self.num_layers)
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class ImgFlowEncoder(nn.Module):
    """
    models that encode in images, flowgrad,
    """

    def __init__(self, incnl=3 + 2):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(incnl, 32, 7)
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

    def forward(self, img):
        down1 = self.down1b(self.down1(img))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        down5 = self.down5b(self.down5(self.down(down4)))
        down6 = self.down6b(self.down6(self.down(down5)))
        down7 = self.down7b(self.down7(self.down(down6)))
        x = self.up(self.mid2(self.mid1(self.down(down7))))
        return [down1, down2, down3, down4, down5, down6, down7, x]


class MPIEncoder2D(nn.Module):
    """
    models that encode in images, flowgrad,
    """

    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(mpi_layers * 4, 64, 3)
        self.down1b = conv(64, 64, 3)
        self.down2 = conv(64, 64, 3)
        self.down2b = conv(64, 64, 3, stride=2)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3, stride=2)
        self.down4 = conv(128, 128, 3)
        self.down4b = conv(128, 128, 3, stride=2)
        self.down5 = conv(128, 128, 3)
        self.down5b = conv(128, 256, 3, stride=2)
        self.down6 = conv(256, 256, 3)
        self.down6b = conv(256, 256, 3, stride=2)

    def forward(self, mpi):
        if mpi.dim() == 5:
            batch_sz, layernum, cnl, hei, wid = mpi.shape
            mpi = mpi.reshape(batch_sz, -1, hei, wid)
        down1 = self.down1b(self.down1(mpi))
        down2 = self.down2b(self.down2(down1))
        down3 = self.down3b(self.down3(down2))
        down4 = self.down4b(self.down4(down3))
        down5 = self.down5b(self.down5(down4))
        down6 = self.down6b(self.down6(down5))
        return [down1, down2, down3, down4, down5, down6]


class MPIEncoder3D(nn.Module):
    """
    models that encode in images, flowgrad,
    """

    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv3d(4, 8, 3)
        self.down1b = conv3d(8, 16, 3, stride=2)
        self.down2 = conv3d(16, 16, 3)
        self.down2b = conv3d(16, 32, 3, stride=2)
        self.down3 = conv3d(32, 32, 3)
        self.down3b = conv3d(32, 64, 3, stride=2)
        self.down4 = conv3d(64, 64, 3)
        self.down4b = conv3d(64, 64, 3, stride=2)
        self.down5 = conv3d(256, 256, 3)
        self.down5b = conv3d(256, 256, 3, stride=2)
        self.down6 = conv3d(256, 256, 3)
        self.down6b = conv3d(256, 256, 3, stride=2)

    def forward(self, mpi: torch.Tensor):
        mpi = mpi.permute(0, 2, 1, 3, 4)
        down1 = self.down1b(self.down1(mpi))
        down2 = self.down2b(self.down2(down1))
        down3 = self.down3b(self.down3(down2))
        down4 = self.down4b(self.down4(down3))
        down5 = self.down5b(self.down5(down4))
        down6 = self.down6b(self.down6(down5))
        return [down1, down2, down3, down4, down5, down6]


class MPI_alpha(nn.Module):
    """
    Takes in: rgb, flow_gradient, warpped disparity map
    """

    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(3 + 2 + 1, 32, 7)
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
        self.output = nn.Conv2d(64, self.num_layers - 1, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img):
        down1 = self.down1b(self.down1(img))
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
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.post2(self.post1(torch.cat(self.shapeto(x, down1), dim=1)))
        x = self.output(self.up1b(self.up1(x)))
        x = self.output_bias(x, self.num_layers)
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class MPI_down8_mask(nn.Module):
    """
    Takes in: rgb, flow_gradient, warpped disparity map
    """

    def __init__(self, mpi_layers, outcnl):
        super().__init__()
        self.num_layers = mpi_layers
        self.outcnl = outcnl
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        resnet = torchvision.models.resnet101(pretrained=True, norm_layer=nn.SyncBatchNorm)
        self.resnet_backbone = nn.ModuleList([
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        ])  # cnl = 256
        self.disp_last_encoder = nn.Sequential(
            conv(self.outcnl, 32, 3),
            conv(32, 64, 3)
        )
        self.down1 = conv(512 + 128 * 2, 512, 3)
        self.down1b = conv(512, 512, 3)
        self.down2 = conv(512, 512, 3)
        self.down2b = conv(512, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(1024, 512, 3)
        self.up2b = conv(512, 512, 3)
        self.up1 = conv(1024, 512, 3)
        self.up1b = conv(512, 512, 3)

        self.mask_decoder = nn.Sequential(
            conv(512 + 256, 512, 3),
            conv(512, 64 * 9, 1, isReLU=False),
        )
        self.depth_decoder = nn.Sequential(
            conv(512 + 512 + 64, 256, 3),
            conv(256, 128, 3),
            conv(128, 64, 3),
            nn.Conv2d(64, outcnl, 3, padding=1)
        )
        self.output_layer = nn.Sigmoid()

        self.register_buffer("outputscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        self.register_buffer("xscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness, scale, depth, thickness, scale
        self.outputscale[:, 1] = 0.3  # thickness
        if self.outcnl == 6:
            self.xscale[:, 3] = 0.3  # mimic the background depth
            self.outputscale[:, 4] = 0.3

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.resnet_backbone:
            im_feat = layer(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = x[..., :hei // 8, :wid // 8]

        disp_feat = self.disp_last_encoder(disp_warp)
        feat = self.depth_decoder(torch.cat([x, im_feat, disp_feat], dim=1))  # 256
        feat = self.output_layer(feat * self.xscale) * self.outputscale

        upmask = self.mask_decoder(torch.cat([x, flow_net], dim=1)) * 0.25
        return feat, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.depth_decoder[-1].bias, -0.5)


class MPI_down8_mask_nobn(nn.Module):
    """
    Takes in: rgb, flow_gradient, warpped disparity map
    """
    def __init__(self, mpi_layers, outcnl=6):
        super().__init__()
        self.num_layers = mpi_layers
        self.outcnl = outcnl
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.backbone = nn.ModuleList([
            conv(3, 32, 7),
            conv(32, 32, 7),
            down,
            conv(32, 64, 5),
            conv(64, 64, 5),
            down,
            conv(64, 128, 3),
            conv(128, 128, 3),
            down,
            conv(128, 256, 3),
            conv(256, 256, 3)
        ])  # cnl = 256
        self.disp_last_encoder = nn.Sequential(
            conv(self.outcnl, 32, 3),
            conv(32, 64, 3)
        )
        self.down1 = conv(256 + 128 * 2, 512, 3)
        self.down1b = conv(512, 512, 3)
        self.down2 = conv(512, 512, 3)
        self.down2b = conv(512, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(1024, 512, 3)
        self.up2b = conv(512, 512, 3)
        self.up1 = conv(1024, 512, 3)
        self.up1b = conv(512, 256, 3)

        self.mask_decoder = nn.Sequential(
            conv(256 + 256, 512, 3),
            conv(512, 64 * 9, 1, isReLU=False),
        )
        self.depth_decoder = nn.Sequential(
            conv(512 + 64, 256, 3),
            conv(256, 128, 3),
            conv(128, 64, 3),
            nn.Conv2d(64, outcnl, 3, padding=1)
        )
        self.output_layer = nn.Sigmoid()

        self.register_buffer("outputscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        self.register_buffer("xscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness, scale, depth, thickness, scale
        self.outputscale[:, 1] = 0.3  # thickness
        if self.outcnl == 6:
            self.xscale[:, 3] = 0.3  # mimic the background depth
            self.outputscale[:, 4] = 0.3

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = x[..., :hei // 8, :wid // 8]

        disp_feat = self.disp_last_encoder(disp_warp)
        feat = self.depth_decoder(torch.cat([x, im_feat, disp_feat], dim=1))  # 256
        feat = self.output_layer(feat * self.xscale) * self.outputscale

        upmask = self.mask_decoder(torch.cat([x, flow_net], dim=1)) * 0.25
        return feat, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.depth_decoder[-1].bias, -0.5)


class MPI_down8_mask_big(nn.Module):
    """
    Takes in: rgb, flow_gradient, warpped disparity map
    """
    def __init__(self, mpi_layers, outcnl=6):
        super().__init__()
        self.num_layers = mpi_layers
        self.outcnl = outcnl
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.backbone = nn.ModuleList([
            conv(3, 32, 7),
            conv(32, 32, 7),
            down,
            conv(32, 64, 5),
            conv(64, 64, 5),
            down,
            conv(64, 128, 3),
            conv(128, 128, 3),
            down,
            conv(128, 256, 3),
            conv(256, 256, 3)
        ])  # cnl = 256
        self.disp_last_encoder = nn.Sequential(
            conv(self.outcnl, 32, 3),
            conv(32, 64, 3)
        )
        self.down1 = conv(256 + 128 * 2, 512, 3)
        self.down1b = conv(512, 512, 3)
        self.down2 = conv(512, 512, 3)
        self.down2b = conv(512, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(1024, 512, 3)
        self.up2b = conv(512, 512, 3)
        self.up1 = conv(1024, 512, 3)
        self.up1b = conv(512, 512, 3)

        self.mask_decoder = nn.Sequential(
            conv(512 + 512, 512, 3),
            conv(512, 64 * 9, 1, isReLU=False),
        )
        self.depth_decoder = nn.Sequential(
            conv(512 + 512 + 64, 512, 3),
            conv(512, 256, 3),
            conv(256, 128, 3),
            conv(128, 64, 3),
            nn.Conv2d(64, outcnl, 3, padding=1)
        )
        self.output_layer = nn.Sigmoid()

        self.register_buffer("outputscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        self.register_buffer("xscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness, scale, depth, thickness, scale
        self.outputscale[:, 1] = 0.3  # thickness
        if self.outcnl == 6:
            self.xscale[:, 3] = 0.3  # mimic the background depth
            self.outputscale[:, 4] = 0.3

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = x[..., :hei // 8, :wid // 8]

        disp_feat = self.disp_last_encoder(disp_warp)
        feat = self.depth_decoder(torch.cat([x, im_feat, flow_net, disp_feat], dim=1))  # 256
        feat = self.output_layer(feat * self.xscale) * self.outputscale

        upmask = self.mask_decoder(torch.cat([x, im_feat, flow_net], dim=1)) * 0.25
        return feat, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.depth_decoder[-1].bias, -0.5)


class MPI_down8_mask_lite(nn.Module):
    """
    Takes in: rgb, flow_gradient, warpped disparity map
    """
    def __init__(self, mpi_layers, outcnl=6):
        super().__init__()
        self.num_layers = mpi_layers
        self.outcnl = outcnl
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.backbone = nn.ModuleList([
            conv(3, 32, 7),
            conv(32, 32, 7),
            down,
            conv(32, 64, 5),
            conv(64, 64, 5),
            down,
            conv(64, 128, 3),
            conv(128, 128, 3),
            down,
            conv(128, 256, 3),
            conv(256, 256, 3)
        ])  # cnl = 256
        self.disp_last_encoder = nn.Sequential(
            conv(self.outcnl, 32, 3),
            conv(32, 64, 3)
        )
        self.down1 = conv(256 + 128 * 2, 384, 3)
        self.down1b = conv(384, 384, 3)
        self.down2 = conv(384, 384, 3)
        self.down2b = conv(384, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 384, 3)
        self.up2 = conv(384 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(384 + 384, 384, 3)
        self.up1b = conv(384, 256, 3)
        self.post = conv(256 + 512, 512, 3)

        self.mask_decoder = nn.Sequential(
            conv(512, 512, 3),
            conv(512, 64 * 9, 1, isReLU=False),
        )
        self.depth_decoder = nn.Sequential(
            conv(512 + 64, 256, 3),
            conv(256, 128, 3),
            conv(128, 64, 3),
            nn.Conv2d(64, outcnl, 3, padding=1)
        )
        self.output_layer = nn.Sigmoid()

        self.register_buffer("outputscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        self.register_buffer("xscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness, scale, depth, thickness, scale
        self.outputscale[:, 1] = 0.3  # thickness
        if self.outcnl == 6:
            self.xscale[:, 3] = 0.3  # mimic the background depth
            self.outputscale[:, 4] = 0.3

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = self.post(torch.cat(self.shapeto(x, feat_in), dim=1))

        disp_feat = self.disp_last_encoder(disp_warp)
        feat = self.depth_decoder(torch.cat([x, disp_feat], dim=1))  # 256
        feat = self.output_layer(feat * self.xscale) * self.outputscale

        upmask = self.mask_decoder(x) * 0.25
        return feat, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.depth_decoder[-1].bias, -0.5)


class MPI_V3(nn.Module):
    """
    Takes in: rgb, flow_gradient, warpped disparity map
    """
    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.outcnl = 6
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.backbone = nn.ModuleList([
            conv(3, 32, 7),
            conv(32, 32, 7),
            down,
            conv(32, 64, 5),
            conv(64, 64, 5),
            down,
            conv(64, 128, 3),
            conv(128, 128, 3),
            down,
            conv(128, 256, 3),
            conv(256, 256, 3)
        ])  # cnl = 256
        self.disp_last_encoder = nn.Sequential(
            conv(self.outcnl, 32, 3),
            conv(32, 64, 3)
        )
        self.down1 = conv(256 + 128 * 2, 384, 3)
        self.down1b = conv(384, 512, 3)
        self.down2 = conv(512, 512, 3)
        self.down2b = conv(512, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(512 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(512 + 384, 512, 3)
        self.up1b = conv(512, 256, 3)
        self.post = conv(256 + 512, 512, 3)

        self.mask_decoder = nn.Sequential(
            conv(512, 512, 3),
            conv(512, 64 * 9, 1, isReLU=False),  # 8 x 8 x 9
        )
        self.depth_decoder = nn.Sequential(
            conv(512 + 64, 256, 3),
            conv(256, 128, 3),
            conv(128, 64, 3),
            nn.Conv2d(64, 6, 3, padding=1)  # dts, dts
        )
        self.output_layer = nn.Sigmoid()
        self.xscale = 2

        self.register_buffer("xbias", torch.zeros(1, self.outcnl, 1, 1, dtype=torch.float32))
        self.register_buffer("outputscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        # output channel: depth 0, thickness 1, scale 2, depth 3, thickness 4, scale 5
        self.xbias[:, 2] = self.xbias[:, 5] = 2  # scale should be near to 1
        self.outputscale[:, 1] = self.outputscale[:, 4] = 0.5  # thickness should be small
        self.xbias[:, 0] = 0.5
        self.xbias[:, 3] = -0.5
        # self.xbias[:, 3] = 0.5
        # self.xbias[:, 0] = -0.5

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = self.post(torch.cat(self.shapeto(x, feat_in), dim=1))

        disp_feat = self.disp_last_encoder(disp_warp)
        feat = self.depth_decoder(torch.cat([x, disp_feat], dim=1))  # 256
        feat = self.output_layer(feat * 2 + self.xbias) * self.outputscale

        upmask = self.mask_decoder(x) * 0.25
        return feat, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder[-1].weight)
        nn.init.constant_(self.depth_decoder[-1].bias, 0)


class MPI_V4(nn.Module):
    """
    Takes in: rgb, flow_gradient, warpped disparity map
    """
    def __init__(self, mpi_layers, backbone='cnn'):
        super().__init__()
        self.num_layers = mpi_layers
        self.outcnl = 4
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        if backbone == 'cnn':
            self.backbone = nn.ModuleList([
                conv(3, 32, 7),
                conv(32, 32, 7),
                down,
                conv(32, 64, 5),
                conv(64, 64, 5),
                down,
                conv(64, 128, 3),
                conv(128, 128, 3),
                down,
                conv(128, 256, 3),
                conv(256, 256, 3, isReLU=False)
            ])  # cnl = 256
        elif backbone == 'cnnbig':
            self.backbone = nn.ModuleList([
                conv(3, 32, 7),
                conv(32, 32, 7),
                conv(32, 64, 5, stride=2),
                conv(64, 64, 3),
                conv(64, 64, 3),
                conv(64, 128, 3, stride=2),
                conv(128, 128, 3),
                conv(128, 128, 3),
                conv(128, 256, 3, stride=2),
                conv(256, 256, 3, isReLU=False)
            ])  # cnl = 256
        elif backbone == 'resnet':
            resnet = torchvision.models.resnet101(pretrained=True, norm_layer=nn.SyncBatchNorm)
            self.backbone = nn.ModuleList([
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
            ])  # cnl = 256
        else:
            raise RuntimeError(f"backbone name {backbone} not recognized")
        self.disp_last_encoder = nn.Sequential(
            conv(self.outcnl, 32, 3),
            conv(32, 64, 3)
        )
        self.down1 = conv(256 + 128 * 2, 384, 3)
        self.down1b = conv(384, 512, 3)
        self.down2 = conv(512, 512, 3)
        self.down2b = conv(512, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(512 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(512 + 384, 512, 3)
        self.up1b = conv(512, 256, 3)
        self.post = conv(256 + 512, 512, 3)

        self.mask_decoder = nn.Sequential(
            conv(512, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(512 + 64, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=1, groups=self.outcnl),
            nn.ReLU(True),
            nn.Conv2d(256, 4, 3, padding=1, groups=self.outcnl),
        )
        self.output_layer = nn.Sigmoid()
        self.xscale = 2

        self.register_buffer("xbias", torch.zeros(1, self.outcnl, 1, 1, dtype=torch.float32))
        self.register_buffer("outputscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        # output channel: depth 0, thickness 1, scale 2, bgdepth
        self.xbias[:, 0] = 0.5
        self.xbias[:, 3] = -0.5

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)
        im_feat = torch.tanh(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = self.post(torch.cat(self.shapeto(x, feat_in), dim=1))

        disp_feat = self.disp_last_encoder(disp_warp)
        feat = self.depth_decoder(torch.cat([x, disp_feat], dim=1))  # 256
        feat = self.output_layer(feat * self.xscale + self.xbias) * self.outputscale

        upmask = self.mask_decoder(x) * 0.25
        return feat, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder[-1].weight)
        nn.init.constant_(self.depth_decoder[-1].bias, 0)


class MPI_V5(nn.Module):
    """
    Predict sigma, d, t, d
    """
    def __init__(self, mpi_layers, backbone='cnn', recurrent=True):
        super().__init__()
        self.num_layers = mpi_layers
        self.outcnl = 4
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.recurrent = recurrent
        self.imfeat_cnl = 256
        if backbone == 'cnn':
            self.backbone = nn.ModuleList([
                conv(3, 32, 7),
                conv(32, 32, 7),
                down,
                conv(32, 64, 5),
                conv(64, 64, 5),
                down,
                conv(64, 128, 3),
                conv(128, 128, 3),
                down,
                conv(128, 256, 3),
                conv(256, 256, 3, isReLU=False)
            ])  # cnl = 256
        elif backbone == 'cnnbig':
            self.backbone = nn.ModuleList([
                conv(3, 32, 7),
                conv(32, 32, 7),
                conv(32, 64, 5, stride=2),
                conv(64, 64, 3),
                conv(64, 64, 3),
                conv(64, 128, 3, stride=2),
                conv(128, 128, 3),
                conv(128, 128, 3),
                conv(128, 256, 3, stride=2),
                conv(256, 256, 3, isReLU=False)
            ])  # cnl = 256
        elif backbone == 'resnet':
            resnet = torchvision.models.resnet101(pretrained=True, norm_layer=nn.SyncBatchNorm)
            self.backbone = nn.ModuleList([
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                conv(512, 256, 1)
            ])  # cnl = 256
        else:
            raise RuntimeError(f"backbone name {backbone} not recognized")
        if recurrent:
            self.disp_last_encoder = nn.Sequential(
                conv(self.outcnl, 32, 3),
                conv(32, 64, 3)
            )
            decoder_cnl = 512 + 64
        else:
            self.disp_last_encoder = None
            decoder_cnl = 512
        self.down1 = conv(self.imfeat_cnl + 128 * 2, 384, 3)
        self.down1b = conv(384, 512, 3)
        self.down2 = conv(512, 512, 3)
        self.down2b = conv(512, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(512 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(512 + 384, 512, 3)
        self.up1b = conv(512, 256, 3)
        self.post = conv(self.imfeat_cnl + 512, 512, 3)

        self.mask_decoder = nn.Sequential(
            conv(512, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(decoder_cnl, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=1, groups=self.outcnl),
            nn.ReLU(True),
            nn.Conv2d(256, 4, 3, padding=1, groups=self.outcnl),
        )
        self.output_layer = nn.Sigmoid()
        self.xscale = 1

        self.register_buffer("xbias", torch.zeros(1, self.outcnl, 1, 1, dtype=torch.float32))
        # output channel: sigma0 depth 1, thickness 2, bgdepth 3
        self.xbias[:, 1] = 0.5
        self.xbias[:, 2] = -0.5
        self.xbias[:, 3] = -0.5

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = self.post(torch.cat(self.shapeto(x, feat_in), dim=1))

        if self.recurrent:
            disp_feat = self.disp_last_encoder(disp_warp)
            feat = self.depth_decoder(torch.cat([x, disp_feat], dim=1))  # 256
        else:
            feat = self.depth_decoder(x)  # 256

        params = feat * self.xscale + self.xbias
        params = torch.cat([
            params[:, 0:1].abs(),
            self.output_layer(params[:, 1:])
                            ], dim=1)

        upmask = self.mask_decoder(x) * 0.25
        return params, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder[-1].weight)
        nn.init.constant_(self.depth_decoder[-1].bias, 0)


class MPI_V5Dual(nn.Module):
    """
    Predict sigma, d, t, d
    """
    def __init__(self, mpi_layers, backbone='cnn', recurrent=True):
        super().__init__()
        self.num_layers = mpi_layers
        self.outcnl = 6
        self.recurrent = recurrent
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
        if backbone == 'cnn':
            self.backbone = nn.ModuleList([
                conv(3, 32, 7),
                conv(32, 32, 7),
                down,
                conv(32, 64, 5),
                conv(64, 64, 5),
                down,
                conv(64, 128, 3),
                conv(128, 128, 3),
                down,
                conv(128, 256, 3),
                conv(256, 256, 3, isReLU=False)
            ])  # cnl = 256
        elif backbone == 'cnnbig':
            self.backbone = nn.ModuleList([
                conv(3, 32, 7),
                conv(32, 32, 7),
                conv(32, 64, 5, stride=2),
                conv(64, 64, 3),
                conv(64, 64, 3),
                conv(64, 128, 3, stride=2),
                conv(128, 128, 3),
                conv(128, 128, 3),
                conv(128, 256, 3, stride=2),
                conv(256, 256, 3, isReLU=False)
            ])  # cnl = 256
        elif backbone == 'resnet':
            resnet = torchvision.models.resnet101(pretrained=True, norm_layer=nn.SyncBatchNorm)
            self.backbone = nn.ModuleList([
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                conv(512, 256, 1)
            ])  # cnl = 256
        else:
            raise RuntimeError(f"backbone name {backbone} not recognized")
        if recurrent:
            self.disp_last_encoder = nn.Sequential(
                conv(self.outcnl, 32, 3),
                conv(32, 64, 3)
            )
            decoder_cnl = 512 + 64
        else:
            self.disp_last_encoder = None
            decoder_cnl = 512

        self.down1 = conv(self.imfeat_cnl + 128 * 2, 384, 3)
        self.down1b = conv(384, 512, 3)
        self.down2 = conv(512, 512, 3)
        self.down2b = conv(512, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(512 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(512 + 384, 512, 3)
        self.up1b = conv(512, 256, 3)
        self.post = conv(self.imfeat_cnl + 512, 512, 3)

        self.mask_decoder = nn.Sequential(
            conv(512, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(decoder_cnl, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 258, 3, padding=1, groups=2),
            nn.ReLU(True),
            nn.Conv2d(258, self.outcnl, 3, padding=1, groups=self.outcnl),
        )
        self.output_layer = nn.Sigmoid()
        self.xscale = 1

        self.register_buffer("xbias", torch.zeros(1, self.outcnl, 1, 1, dtype=torch.float32))
        # output channel: sigma 0, depth 1, thickness 2, bgsigma 3, bgdepth 4, bgthick 5
        self.xbias[:, 1] = 0.5
        self.xbias[:, 4] = -0.5
        self.xbias[:, 2] = -0.5
        self.xbias[:, 5] = -0.5

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = self.post(torch.cat(self.shapeto(x, feat_in), dim=1))

        if self.recurrent:
            disp_feat = self.disp_last_encoder(disp_warp)
            feat = self.depth_decoder(torch.cat([x, disp_feat], dim=1))  # 256
        else:
            feat = self.depth_decoder(x)  # 256
        params = feat * self.xscale + self.xbias
        params = torch.cat([
            params[:, 0:1].abs(),
            self.output_layer(params[:, 1:])
                            ], dim=1)
        upmask = self.mask_decoder(x) * 0.25
        return params, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder[-1].weight)
        nn.init.constant_(self.depth_decoder[-1].bias, 0)


class MPI_V5Nset(nn.Module):
    """
    Predict sigma, d, t, d
    """
    def __init__(self, mpi_layers, backbone='cnn', num_set=2):
        super().__init__()
        self.num_layers = mpi_layers
        self.num_set = num_set
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
        if backbone == 'cnn':
            self.backbone = nn.ModuleList([
                conv(3, 32, 7),
                conv(32, 32, 7),
                down,
                conv(32, 64, 5),
                conv(64, 64, 5),
                down,
                conv(64, 128, 3),
                conv(128, 128, 3),
                down,
                conv(128, 256, 3),
                conv(256, 256, 3, isReLU=False)
            ])  # cnl = 256
        elif backbone == 'cnnbig':
            self.backbone = nn.ModuleList([
                conv(3, 32, 7),
                conv(32, 32, 7),
                conv(32, 64, 5, stride=2),
                conv(64, 64, 3),
                conv(64, 64, 3),
                conv(64, 128, 3, stride=2),
                conv(128, 128, 3),
                conv(128, 128, 3),
                conv(128, 256, 3, stride=2),
                conv(256, 256, 3, isReLU=False)
            ])  # cnl = 256
        elif backbone == 'resnet':
            resnet = torchvision.models.resnet101(pretrained=True, norm_layer=nn.SyncBatchNorm)
            self.backbone = nn.ModuleList([
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                conv(512, 256, 1)
            ])  # cnl = 256
        else:
            raise RuntimeError(f"backbone name {backbone} not recognized")

        self.down1 = conv(self.imfeat_cnl + 128 * 2, 384, 3)
        self.down1b = conv(384, 512, 3)
        self.down2 = conv(512, 512, 3)
        self.down2b = conv(512, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(512 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(512 + 384, 512, 3)
        self.up1b = conv(512, 256, 3)
        self.post = conv(self.imfeat_cnl + 512, 512, 3)

        self.mask_decoder = nn.Sequential(
            conv(512, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )
        cnl1 = self.num_set * 256
        cnl2 = self.num_set * 3 * 64
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(512, cnl1, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(cnl1, cnl2, 3, padding=1, groups=self.num_set),
            nn.ReLU(True),
            nn.Conv2d(cnl2, self.num_set * 3, 3, padding=1, groups=self.num_set * 3),
        )
        self.xscale = 1

        self.register_buffer("xbias", torch.zeros(1, self.num_set * 3, 1, 1, dtype=torch.float32))
        # output channel: sigma, depth, thickness
        self.xbias[:, :self.num_set] = 0.5
        self.xbias[:, 2 * self.num_set:] = -0.5

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)

        feat_in = torch.cat([im_feat, flow_net], dim=1)
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = self.post(torch.cat(self.shapeto(x, feat_in), dim=1))

        feat = self.depth_decoder(x)  # 256
        params = feat * self.xscale + self.xbias
        params = torch.cat([
            torch.relu(params[:, :self.num_set]),
            torch.sigmoid(params[:, self.num_set:])
                            ], dim=1)
        upmask = self.mask_decoder(x) * 0.25
        return params, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder[-1].weight)
        nn.init.constant_(self.depth_decoder[-1].bias, 0)


def learned_upsample(content, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, C, H, W = content.shape
    mask = mask.reshape(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    content = torchf.pad(content, [1, 1, 1, 1], 'replicate')
    up_flow = torchf.unfold(content, [3, 3], padding=0)
    up_flow = up_flow.reshape(N, C, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, C, 8 * H, 8 * W)
