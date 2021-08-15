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


class MPI_V5Nset(nn.Module):
    """
    Predict sigma, d, t, d
    """
    def __init__(self, mpi_layers, backbone='cnn', num_set=2, thickmax=1):
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
        self.register_buffer("paramscale", torch.ones(1, self.num_set * 3, 1, 1, dtype=torch.float32))
        # output channel: sigma, depth, thickness
        self.xbias[:, :self.num_set] = 0.5
        self.xbias[:, 2 * self.num_set:] = -0.5
        self.paramscale[:, 2 * self.num_set:] = thickmax

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
        params = params * self.paramscale
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


class MPI_V6Nset(nn.Module):
    """
    Predict sigma, d, t, d
    """
    def __init__(self, mpi_layers, num_set=2):
        super().__init__()
        self.num_layers = mpi_layers
        self.num_set = num_set
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
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


class MPI_LDI(nn.Module):
    """
    Predict sigma, d, t, d
    """
    def __init__(self, mpi_layers, num_set=2):
        super().__init__()
        self.num_layers = mpi_layers
        self.num_set = num_set
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
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
        outparamnum = self.num_set * 2
        cnl1 = self.num_set * 128
        cnl2 = outparamnum * 64
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(512, cnl1, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(cnl1, cnl2, 3, padding=1, groups=self.num_set),
            nn.ReLU(True),
            nn.Conv2d(cnl2, outparamnum, 3, padding=1, groups=outparamnum),
        )

        self.register_buffer("oscale", torch.ones(1, self.num_set * 2, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness
        self.oscale[:, self.num_set:] = 5 / self.num_layers

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
        params = torch.sigmoid(feat) * self.oscale

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


class MPI_LDI_res(nn.Module):
    def __init__(self, mpi_layers, num_set=2):
        super().__init__()
        self.num_layers = mpi_layers
        self.num_set = num_set
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
        self.deepfeat_cnl = 256
        outparamnum = self.num_set * 2

        self.backbone = nn.ModuleList([
            conv(3, 64, 7, stride=2),
            ResidualBlock(64, 64, norm_fn='none', stride=1),

            ResidualBlock(64, 128, norm_fn='none', stride=2),
            ResidualBlock(128, 128, norm_fn='none', stride=1),

            ResidualBlock(128, 256, norm_fn='none', stride=2),
            ResidualBlock(256, 256, norm_fn='none', stride=1)
        ])  # cnl = 256

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
        self.up1b = conv(512, self.deepfeat_cnl, 3)

        self.mask_decoder = nn.Sequential(
            conv(self.deepfeat_cnl + self.imfeat_cnl + 256, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )

        self.depth_decoder = nn.Sequential(
            nn.Conv2d(self.deepfeat_cnl + self.imfeat_cnl + 256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=1, groups=self.num_set),
            nn.ReLU(True),
            nn.Conv2d(256, outparamnum, 3, padding=1, groups=outparamnum),
        )

        self.register_buffer("oscale", torch.ones(1, self.num_set * 2, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness
        self.oscale[:, self.num_set:] = 3 / (self.num_layers - 1)

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
        x = torch.cat(self.shapeto(x, feat_in), dim=1)

        feat = self.depth_decoder(x)  # 256
        params = torch.sigmoid(feat) * self.oscale

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


class MPI_LDIdiv(nn.Module):
    def __init__(self, mpi_layers, num_set=2):
        super().__init__()
        self.num_layers = mpi_layers
        self.num_set = num_set
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
        self.deepfeat_cnl = 256
        outparamnum = self.num_set * 2

        self.backbone = nn.ModuleList([
            conv(3, 64, 7, stride=2),
            ResidualBlock(64, 64, norm_fn='none', stride=1),

            ResidualBlock(64, 128, norm_fn='none', stride=2),
            ResidualBlock(128, 128, norm_fn='none', stride=1),

            ResidualBlock(128, 256, norm_fn='none', stride=2),
            ResidualBlock(256, 256, norm_fn='none', stride=1)
        ])  # cnl = 256

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
        self.up1b = conv(512, self.deepfeat_cnl, 3)

        self.mask_decoder = nn.Sequential(
            conv(self.deepfeat_cnl + self.imfeat_cnl + 256, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )

        self.depth_decoder = nn.Sequential(
            nn.Conv2d(self.deepfeat_cnl + self.imfeat_cnl + 256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 128, 3, padding=1, groups=self.num_set),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1, groups=self.num_set),
            nn.ReLU(True),
            nn.Conv2d(64, outparamnum, 3, padding=1, groups=self.num_set),
        )

        self.register_buffer("oscale", torch.ones(1, self.num_set * 2, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness
        self.oscale[:, self.num_set:] = 2 / (self.num_layers - 1)

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
        x = torch.cat(self.shapeto(x, feat_in), dim=1)

        feat = self.depth_decoder(x)  # 256
        params = torch.sigmoid(feat) * self.oscale

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


class MPI_LDIbig(nn.Module):
    def __init__(self, mpi_layers):
        super().__init__()
        self.num_set = 2
        self.num_layers = mpi_layers
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
        self.deepfeat_cnl = 256
        self.unet_out_cnl = self.deepfeat_cnl + self.imfeat_cnl + 256

        self.backbone = nn.ModuleList([
            conv(3, 64, 7, stride=2),
            ResidualBlock(64, 64, norm_fn='none', stride=1),

            ResidualBlock(64, 128, norm_fn='none', stride=2),
            ResidualBlock(128, 128, norm_fn='none', stride=1),

            ResidualBlock(128, 256, norm_fn='none', stride=2),
            ResidualBlock(256, 256, norm_fn='none', stride=1)
        ])  # cnl = 256

        self.down1 = conv(self.imfeat_cnl + 128 * 2, 384, 3)
        self.down1b = conv(384, 384, 3)
        self.down2 = conv(384, 384, 3)
        self.down2b = conv(384, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(512 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(384 + 384, 384, 3)
        self.up1b = conv(384, self.deepfeat_cnl, 3)

        self.mask_decoder = nn.Sequential(
            conv(self.unet_out_cnl, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )

        self.depth_decoder1 = nn.Sequential(
            conv(self.unet_out_cnl, 256),
            conv(256, 64),
            conv(64, 64),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.depth_decoder2 = nn.Sequential(
            conv(self.unet_out_cnl, 256),
            conv(256, 64),
            conv(64, 64),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.register_buffer("oscale", torch.ones(1, 2, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness
        self.oscale[:, 1] = 3 / (self.num_layers - 1)

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
        x = self.up(self.mid2(self.down(down3)))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = torch.cat(self.shapeto(x, feat_in), dim=1)

        feat1 = self.active(self.depth_decoder1(x))
        feat2 = self.active(self.depth_decoder2(x))

        params = torch.cat([feat1[:, 0:1], feat2[:, 0:1], feat1[:, 1:], feat2[:, 1:]], dim=1)

        upmask = self.mask_decoder(x) * 0.25
        return params, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder1[-1].weight)
        nn.init.constant_(self.depth_decoder1[-1].bias, 0)
        nn.init.xavier_normal_(self.depth_decoder2[-1].weight)
        nn.init.constant_(self.depth_decoder2[-1].bias, 0)

    def active(self, net):
        disp, thick = net.split(1, dim=1)
        disp = torch.sigmoid(disp)
        if self.training and np.random.randint(0, 2) == 0:
            thick = torch.sigmoid(thick)
        else:
            thick = torchf.hardsigmoid(thick)  # * 1.2
        # thick = torch.sigmoid(thick)
        return torch.cat([disp, thick], dim=1) * self.oscale


def active(net):
    return torch.sigmoid(net)


class RGBADNet(nn.Module):
    def __init__(self, mpi_layers, num_layers=2):
        super().__init__()
        self.is_ldi = mpi_layers == num_layers
        self.num_set = 2
        self.num_layers = mpi_layers
        self.ldi_layers = num_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up

        self.down1 = conv(3 + 3 + 1, 48, 7)
        self.down1b = conv(48, 48, 7)
        self.down2 = conv(48, 64, 5)
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
        self.up7 = conv(512 + 512, 512, 3)
        self.up7b = conv(512, 512, 3)
        self.up6 = conv(512 + 512, 512, 3)
        self.up6b = conv(512, 512, 3)
        self.up5 = conv(512 + 512, 512, 3)
        self.up5b = conv(512, 512, 3)
        self.up4 = conv(768, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(384, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(192, 64, 3)
        self.up2b = conv(64, 64, 3)

        self.depth_decoder = nn.Sequential(
            conv(512, 128, kernel_size=1),
            conv(128, 64),
            nn.Conv2d(64, self.ldi_layers, 3, padding=1)
        )
        self.alpha_decoder = nn.Sequential(
            conv(64 + 48, 64, 3),
            conv(64, 32, 3),
            conv(32, 16, 3),
            nn.Conv2d(16, self.ldi_layers - 1, 3, padding=1)
        )
        self.rgb_decoder = nn.Sequential(
            conv(64 + 48, 64, 3),
            conv(64, 64, 3),
            conv(64, 64, 3),
            nn.Conv2d(64, self.ldi_layers * 3, 3, padding=1)
        )
        self.active01 = lambda x: torch.sigmoid(x)
        self.active11 = lambda x: torch.tanh(x)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0

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
        disps = self.active01(self.depth_decoder(x))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = torch.cat(self.shapeto(x, down1), dim=1)
        alphas = self.active01(self.alpha_decoder(x))
        rgbs = self.active11(self.rgb_decoder(x))
        return disps, alphas, rgbs

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder[-1].weight)
        nn.init.constant_(self.depth_decoder[-1].bias[0], -0.5)
        nn.init.constant_(self.depth_decoder[-1].bias[1], 0.5)
        nn.init.xavier_normal_(self.alpha_decoder[-1].weight)
        nn.init.constant_(self.alpha_decoder[-1].bias, 0)
        nn.init.xavier_normal_(self.rgb_decoder[-1].weight)
        nn.init.constant_(self.rgb_decoder[-1].bias, 0)


class MPI_AB_up(nn.Module):
    def __init__(self, mpi_layers, num_set=2):
        super().__init__()
        self.num_layers = mpi_layers
        self.num_set = num_set
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
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

        cnl1 = self.num_set * 256
        cnl2 = self.num_set * 3 * 128
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(512, cnl1, 3, padding=1),
            nn.ReLU(True),
            nn.PixelShuffle(2),
            nn.Conv2d(cnl1 // 4, cnl2, 3, padding=1, groups=self.num_set),
            nn.ReLU(True),
            nn.PixelShuffle(2),
            nn.Conv2d(cnl2 // 4, self.num_set * 3 * 4, 3, padding=1, groups=self.num_set * 3),
            nn.PixelShuffle(2),
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
        return params, None

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder[-2].weight)
        nn.init.constant_(self.depth_decoder[-2].bias, 0)


class MPILDI_AB_alpha(nn.Module):
    def __init__(self, mpi_layers):
        super().__init__()
        self.num_set = 2
        self.num_layers = mpi_layers
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
        self.deepfeat_cnl = 256
        self.unet_out_cnl = self.deepfeat_cnl + self.imfeat_cnl + 256

        self.backbone = nn.ModuleList([
            conv(3, 64, 7, stride=2),
            ResidualBlock(64, 64, norm_fn='none', stride=1),

            ResidualBlock(64, 128, norm_fn='none', stride=2),
            ResidualBlock(128, 128, norm_fn='none', stride=1),

            ResidualBlock(128, 256, norm_fn='none', stride=2),
            ResidualBlock(256, 256, norm_fn='none', stride=1)
        ])  # cnl = 256

        self.down1 = conv(self.imfeat_cnl + 128 * 2, 384, 3)
        self.down1b = conv(384, 384, 3)
        self.down2 = conv(384, 384, 3)
        self.down2b = conv(384, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(512 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(384 + 384, 384, 3)
        self.up1b = conv(384, self.deepfeat_cnl, 3)

        self.mask_decoder = nn.Sequential(
            conv(self.unet_out_cnl, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )

        self.depth_decoder = nn.Sequential(
            conv(self.unet_out_cnl, 256),
            conv(256, 64),
            conv(64, 64),
            nn.Conv2d(64, self.num_layers - 1, 3, padding=1),
        )
        self.register_buffer("oscale", torch.ones(1, 2, 1, 1, dtype=torch.float32))
        self.outputbias = apply_harmonic_bias

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
        x = self.up(self.mid2(self.down(down3)))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = torch.cat(self.shapeto(x, feat_in), dim=1)

        alphas = self.outputbias(self.depth_decoder(x), self.num_layers)
        upmask = self.mask_decoder(x) * 0.25
        return alphas, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder[-1].weight)
        nn.init.constant_(self.depth_decoder[-1].bias, 0)


class MPILDI_AB_nonet(nn.Module):
    def __init__(self, mpi_layers):
        super().__init__()
        self.num_set = 2
        self.num_layers = mpi_layers
        down = nn.MaxPool2d(2, ceil_mode=True)
        self.down = down
        self.up = up
        self.imfeat_cnl = 256
        self.deepfeat_cnl = 256
        self.unet_out_cnl = self.deepfeat_cnl + self.imfeat_cnl

        self.backbone = nn.ModuleList([
            conv(3, 64, 7, stride=2),
            ResidualBlock(64, 64, norm_fn='none', stride=1),

            ResidualBlock(64, 128, norm_fn='none', stride=2),
            ResidualBlock(128, 128, norm_fn='none', stride=1),

            ResidualBlock(128, 256, norm_fn='none', stride=2),
            ResidualBlock(256, 256, norm_fn='none', stride=1)
        ])  # cnl = 256

        self.down1 = conv(self.imfeat_cnl, 384, 3)
        self.down1b = conv(384, 384, 3)
        self.down2 = conv(384, 384, 3)
        self.down2b = conv(384, 512, 3)
        self.down3 = conv(512, 512, 3)
        self.down3b = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up3 = conv(1024, 512, 3)
        self.up3b = conv(512, 512, 3)
        self.up2 = conv(512 + 512, 512, 3)
        self.up2b = conv(512, 384, 3)
        self.up1 = conv(384 + 384, 384, 3)
        self.up1b = conv(384, self.deepfeat_cnl, 3)

        self.mask_decoder = nn.Sequential(
            conv(self.unet_out_cnl, 512, 3),
            nn.Conv2d(512, 64 * 9, 1)
        )

        self.depth_decoder1 = nn.Sequential(
            conv(self.unet_out_cnl, 256),
            conv(256, 64),
            conv(64, 64),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.depth_decoder2 = nn.Sequential(
            conv(self.unet_out_cnl, 256),
            conv(256, 64),
            conv(64, 64),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.register_buffer("oscale", torch.ones(1, 2, 1, 1, dtype=torch.float32))
        # output channel: depth, thickness
        self.oscale[:, 1] = 2.5 / (self.num_layers - 1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img, flow_net, disp_warp):
        batchsz, _, hei, wid = img.shape
        assert hei % 8 == 0 and wid % 8 == 0
        im_feat = img
        for layer in self.backbone:
            im_feat = layer(im_feat)

        feat_in = im_feat
        down1 = self.down1b(self.down1(self.down(feat_in)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        x = self.up(self.mid2(self.down(down3)))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up(self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1))))
        x = torch.cat(self.shapeto(x, feat_in), dim=1)

        feat1 = torch.sigmoid(self.depth_decoder1(x)) * self.oscale  # 256
        feat2 = torch.sigmoid(self.depth_decoder2(x)) * self.oscale

        params = torch.cat([feat1[:, 0:1], feat2[:, 0:1], feat1[:, 1:], feat2[:, 1:]], dim=1)

        upmask = self.mask_decoder(x) * 0.25
        return params, upmask

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.depth_decoder1[-1].weight)
        nn.init.constant_(self.depth_decoder1[-1].bias, 0)
        nn.init.xavier_normal_(self.depth_decoder2[-1].weight)
        nn.init.constant_(self.depth_decoder2[-1].bias, 0)


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

