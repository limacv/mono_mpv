import torch.nn as nn
import torch
from ._modules import *


def initial_weights(model):
    if isinstance(model, nn.ModuleDict):
        for k, m in model.items():
            m.initial_weights()
    else:
        model.initial_weights()


class MPIReccuNet(nn.Module):
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

        self.mpidown1 = conv(self.num_layers - 1 + 3, 64, 3)
        self.mpidown2 = conv(64, 64, 3)
        self.mpidown3 = conv(64, 64, 3)
        self.mpidown4 = conv(64, 128, 3)
        self.mpidown5 = conv(128, 128, 3)
        self.mpidown6 = conv(128, 256, 3)
        self.mpidown7 = conv(256, 256, 3)

        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up7 = conv(1024 + 256, 512, 3)
        self.up7b = conv(512, 512, 3)
        self.up6 = conv(1024 + 256, 512, 3)
        self.up6b = conv(512, 512, 3)
        self.up5 = conv(1024 + 128, 512, 3)
        self.up5b = conv(512, 512, 3)
        self.up4 = conv(768 + 128, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(384 + 64, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(192 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.post1 = conv(96 + 64, 64, 3)
        self.post2 = conv(64, 64, 3)
        self.up1 = conv(64, 64, 3)
        self.up1b = conv(64, 64, 3)
        self.output = nn.Conv2d(64, self.num_layers - 1 + 3, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    @staticmethod
    def shapeto(x, tar, y):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar, y]

    def forward(self, img, mpi=None):
        if mpi is None:
            b, _, h, w = img.shape
            mpi = torch.zeros((b, self.num_layers - 1 + 3, h, w)).type_as(img)
            mpi[:, -3:] = img
        down1 = self.down1b(self.down1(img))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        down5 = self.down5b(self.down5(self.down(down4)))
        down6 = self.down6b(self.down6(self.down(down5)))
        down7 = self.down7b(self.down7(self.down(down6)))
        x = self.up(self.mid2(self.mid1(self.down(down7))))

        mpidown1 = self.mpidown1(mpi)
        mpidown2 = self.mpidown2(self.down(mpidown1))
        mpidown3 = self.mpidown3(self.down(mpidown2))
        mpidown4 = self.mpidown4(self.down(mpidown3))
        mpidown5 = self.mpidown5(self.down(mpidown4))
        mpidown6 = self.mpidown6(self.down(mpidown5))
        mpidown7 = self.mpidown7(self.down(mpidown6))

        x = self.up(self.up7b(self.up7(torch.cat(self.shapeto(x, down7, mpidown7), dim=1))))
        x = self.up(self.up6b(self.up6(torch.cat(self.shapeto(x, down6, mpidown6), dim=1))))
        x = self.up(self.up5b(self.up5(torch.cat(self.shapeto(x, down5, mpidown5), dim=1))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4, mpidown4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3, mpidown3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2, mpidown2), dim=1))))
        x = self.post2(self.post1(torch.cat(self.shapeto(x, down1, mpidown1), dim=1)))
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


class MPIRecuFlowNet(nn.Module):
    def __init__(self, mpi_layers):
        super().__init__()
        self.num_layers = mpi_layers
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(5, 64, 7)
        self.down1b = conv(64, 64, 7)
        self.down2 = conv(64, 64, 5)
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

        self.mpidown1 = conv(self.num_layers - 1 + 3, 64, 3)
        self.mpidown2 = conv(64, 64, 3)
        self.mpidown3 = conv(64, 64, 3)
        self.mpidown4 = conv(64, 128, 3)
        self.mpidown5 = conv(128, 128, 3)
        self.mpidown6 = conv(128, 256, 3)
        self.mpidown7 = conv(256, 256, 3)

        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up7 = conv(1024 + 256, 512, 3)
        self.up7b = conv(512, 512, 3)
        self.up6 = conv(1024 + 256, 512, 3)
        self.up6b = conv(512, 512, 3)
        self.up5 = conv(1024 + 128, 512, 3)
        self.up5b = conv(512, 512, 3)
        self.up4 = conv(768 + 128, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(384 + 64, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(192 + 64, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.post1 = conv(128 + 64, 64, 3)
        self.post2 = conv(64, 64, 3)
        self.up1 = conv(64, 64, 3)
        self.up1b = conv(64, 64, 3)
        self.output = nn.Conv2d(64, self.num_layers - 1 + 3, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    @staticmethod
    def shapeto(x, tar, y):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar, y]

    def forward(self, img, flow=None, mpi=None):
        if mpi is None:
            b, _, h, w = img.shape
            mpi = torch.zeros((b, self.num_layers - 1 + 3, h, w)).type_as(img)
            mpi[:, -3:] = img
        if flow is None:
            b, _, h, w = img.shape
            flow = torch.zeros((b, 2, h, w)).type_as(img)
        down1 = self.down1b(self.down1(torch.cat([img, flow], dim=1)))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        down5 = self.down5b(self.down5(self.down(down4)))
        down6 = self.down6b(self.down6(self.down(down5)))
        down7 = self.down7b(self.down7(self.down(down6)))
        x = self.up(self.mid2(self.mid1(self.down(down7))))

        mpidown1 = self.mpidown1(mpi)
        mpidown2 = self.mpidown2(self.down(mpidown1))
        mpidown3 = self.mpidown3(self.down(mpidown2))
        mpidown4 = self.mpidown4(self.down(mpidown3))
        mpidown5 = self.mpidown5(self.down(mpidown4))
        mpidown6 = self.mpidown6(self.down(mpidown5))
        mpidown7 = self.mpidown7(self.down(mpidown6))

        x = self.up(self.up7b(self.up7(torch.cat(self.shapeto(x, down7, mpidown7), dim=1))))
        x = self.up(self.up6b(self.up6(torch.cat(self.shapeto(x, down6, mpidown6), dim=1))))
        x = self.up(self.up5b(self.up5(torch.cat(self.shapeto(x, down5, mpidown5), dim=1))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4, mpidown4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3, mpidown3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2, mpidown2), dim=1))))
        x = self.post2(self.post1(torch.cat(self.shapeto(x, down1, mpidown1), dim=1)))
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


class MPIFuser(nn.Module):
    def __init__(self, planenum, outplanenum=None):
        self.num_layers = planenum
        self.out_num_layers = outplanenum if outplanenum is not None else planenum
        super().__init__()

        channels = [4 * 2, 16, 32, 64, 64, 128]
        outchannels = [128, 64*2, 64*2, 32*2, 16*2, self.out_num_layers]
        depthstride = [1, 1, 2, 1, 2]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(1, len(channels)):
            self.downs.append(
                nn.Sequential(
                    nn.Conv3d(channels[i - 1], channels[i], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channels[i], channels[i], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channels[i], channels[i], kernel_size=3, padding=1, stride=[depthstride[i - 1], 2, 2]),
                    nn.ReLU(inplace=True),
                )
            )
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(outchannels[i - 1], channels[-i - 1], kernel_size=3,
                                       stride=[depthstride[-i], 2, 2], padding=1,
                                       output_padding=[depthstride[-i]-1, 1, 1]),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(channels[-i - 1], channels[-i - 1], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(channels[-i - 1], channels[-i - 1], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.output = nn.Conv3d(channels[0], 4, kernel_size=3, padding=1)
        self.output_bias = apply_harmonic_bias_forfuse

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, mpi: torch.Tensor, mpi1: torch.Tensor):
        """
        mpi/mpi1: [B, 4, num_layers, H, W]
        """
        if mpi.shape[-3] == 4:
            mpi = mpi.permute(0, 2, 1, 3, 4)
            mpi1 = mpi1.permute(0, 2, 1, 3, 4)

        net = torch.cat([mpi, mpi1], dim=1)
        skip = []
        for down in self.downs:
            net = down(net)
            skip.append(net)

        out = skip.pop()
        for up in self.ups:
            if len(skip) > 0:
                out = torch.cat(self.shapeto(up(out), skip.pop()), dim=1)
            else:
                out = up(out)

        out = self.output(out)  # B x 4 x Layer x H x W
        out = self.output_bias(out)
        return out.permute(0, 2, 1, 3, 4)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
