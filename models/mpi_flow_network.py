"""
Models that can output flow / or multi plane flows
"""

import torch
import torch.nn as nn
import torch.nn.functional as torchf
from ._modules import *


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


class AFNet(nn.Module):
    """
    Appearance Multiplane flow with disparity input
    alpha input
    Multi plane flow input, output single plane appearance flow
    in: B x 3 (dx, dy, alpha) x H x W
    """
    def __init__(self, netcnl=6, hasmask=True):
        super().__init__()
        perframecnl = 4 if hasmask else 3
        self.hasmask = hasmask
        self.netcnl = netcnl
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(netcnl + 3 + perframecnl * 2, 64, 7)
        self.down1b = conv(64, 64, 5)
        self.down2 = conv(64, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 256, 3)
        self.down5 = conv(256, 384, 3)
        self.down5b = conv(384, 384, 3)
        self.mid1 = conv(384, 384, 3)
        self.mid2 = conv(384, 384, 3)
        self.up5 = conv(384 + 384, 384, 3)
        self.up5b = conv(384, 384, 3)
        self.up4 = conv(384 + 256, 256, 3)
        self.up4b = conv(256, 128, 3)
        self.up3 = conv(128 + 128, 128, 3)
        self.up3b = conv(128, 64, 3)
        self.up2 = conv(64 + 64, 64, 3)
        self.up2b = conv(64, 32, 3)
        self.up1 = conv(32 + 64, 32, 3)
        self.up1b = conv(32, 32, 3)
        self.output = nn.Conv2d(32, 3, 3, padding=1)
        self.act = nn.Sigmoid()

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, inpu):
        down1 = self.down1b(self.down1(inpu))
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
        return self.act(self.output(x) * 2)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class AFNet_AB_svdbg(nn.Module):
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
        self.down1 = conv(netcnl + 3, 64, 7)
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
        self.output = nn.Conv2d(32, 3, 3, padding=1)

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, net, img):
        x = torch.cat([net, img], dim=1)
        down1 = self.down1b(self.down1(x))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        x = self.up(self.mid2(self.mid1(self.down(down4))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.up1b(self.up1(torch.cat(self.shapeto(x, down1), dim=1)))
        return torch.sigmoid(self.output(x) * 2)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class InPaintNet(nn.Module):
    def __init__(self, netcnl, residual_blocks=8, hasmask=False):
        super(InPaintNet, self).__init__()
        perframecnl = 4 if hasmask else 3
        self.hasmask = hasmask
        self.netcnl = netcnl
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=netcnl + 3 + perframecnl * 2, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x

    def initial_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
