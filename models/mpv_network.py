from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as torchf
from ._modules import *


class MPVNet(nn.Module):
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
        self.merge3_in_1 = conv(128 + 128, 256)
        self.merge3_in_2 = conv(256, 256)
        self.merge3_in_3 = conv(256, 128)
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

    def forward(self, img, feat_in=None):
        down1 = self.down1b(self.down1(img))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        if feat_in is None:
            batchsz, _, hei3, wid3 = down3.shape
            feat_in = torch.zeros(batchsz, 128, hei3, wid3).type_as(down3)
        down3 = self.merge3_in_1(torch.cat([down3, feat_in], dim=1))
        down3 = self.merge3_in_2(down3)
        down3 = self.merge3_in_3(down3)
        down4 = self.down4b(self.down4(self.down(down3)))
        down5 = self.down5b(self.down5(self.down(down4)))
        down6 = self.down6b(self.down6(self.down(down5)))
        down7 = self.down7b(self.down7(self.down(down6)))
        x = self.up(self.mid2(self.mid1(self.down(down7))))
        x = self.up(self.up7b(self.up7(torch.cat([x, down7], dim=1))))
        x = self.up(self.up6b(self.up6(torch.cat([x, down6], dim=1))))
        x = self.up(self.up5b(self.up5(torch.cat([x, down5], dim=1))))
        x = self.up(self.up4b(self.up4(torch.cat([x, down4], dim=1))))
        feat_out = self.up3b(self.up3(torch.cat([x, down3], dim=1)))
        x = self.up(feat_out)
        x = self.up(self.up2b(self.up2(torch.cat([x, down2], dim=1))))
        x = self.post2(self.post1(torch.cat([x, down1], dim=1)))
        x = self.output(self.up1b(self.up1(x)))
        x = self.output_bias(x, self.num_layers)
        return x, feat_out

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
