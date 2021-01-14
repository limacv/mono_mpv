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
    def __init__(self, incnl=3+2):
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
        self.resnet_backbone = nn.ModuleList([
            torchvision.models.resnet101(pretrained=True).conv1,
            torchvision.models.resnet101(pretrained=True).bn1,
            torchvision.models.resnet101(pretrained=True).relu,
            torchvision.models.resnet101(pretrained=True).maxpool,
            torchvision.models.resnet101(pretrained=True).layer1,
            torchvision.models.resnet101(pretrained=True).layer2,
        ])  # cnl = 256
        self.flow_encoder = nn.Sequential(
            conv(2, 32, 3),
            conv(32, 64, 3)
        )
        self.disp_last_encoder = nn.Sequential(
            conv(self.outcnl, 32, 3),
            conv(32, 64, 3)
        )
        self.down1 = conv(512 + 64, 512, 3)
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

        self.mask_1 = nn.Sequential(
            conv(512, 512, 3),
            conv(512, 64 * 9, 3),
        )
        self.depth_1 = conv(512 + 64, 512, 3)
        self.depth_1b = conv(512, 256, 3)
        self.depth_2 = conv(256, 128, 3)
        self.depth_2b = conv(128, 128, 3)
        self.depth_3 = conv(128, 64, 3)
        self.depth_3b = conv(64, 64, 3)
        self.output = nn.Conv2d(64, outcnl, 3, padding=1)
        self.meanout = nn.Sigmoid()

        self.register_buffer("outputscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        self.register_buffer("xscale", torch.ones(1, self.outcnl, 1, 1, dtype=torch.float32))
        self.outputscale[:, 1] = 0.3  # thickness
        if self.outcnl == 6:
            self.xscale[:, 3] = 0.3
            self.outputscale[:, 4] = 0.3

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_grad, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.resnet_backbone:
            im_feat = layer(im_feat)
        flow_feat = self.flow_encoder(flow_grad)
        disp_feat = self.disp_last_encoder(disp_warp)
        down1 = self.down1b(self.down1(self.down(torch.cat([im_feat, flow_feat], dim=1))))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.mid1(self.down(down3))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = x[..., :hei // 8, :wid // 8]

        feat = self.depth_1b(self.depth_1(torch.cat([x, disp_feat], dim=1)))  # 256
        feat = self.depth_2b(self.depth_2(feat))  # 128
        feat = self.depth_3b(self.depth_3(feat))  # 64
        feat = self.output(feat)
        feat = self.meanout(feat * self.xscale) * self.outputscale

        upmask = self.mask_1(x)
        return feat, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.output.bias, -1)


def learned_upsample(content, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, C, H, W = content.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    content = torchf.pad(content, [1, 1, 1, 1], 'replicate')
    up_flow = torchf.unfold(content, [3, 3], padding=0)
    up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, C, 8 * H, 8 * W)

