# Crediate: the code is modified from https://github.com/yifjiang/relative-depth-using-pytorch

import torch
from torch import nn
from ._modules import inception, apply_harmonic_bias


class Channels1(nn.Module):
    def __init__(self):
        super(Channels1, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
            )
        )  # EE
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                nn.UpsamplingNearest2d(scale_factor=2)
            )
        )  # EEE

    def forward(self, x):
        return self.list[0](x) + self.list[1](x)


class Channels2(nn.Module):
    def __init__(self):
        super(Channels2, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])
            )
        )  # EF
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                Channels1(),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]]),
                nn.UpsamplingNearest2d(scale_factor=2)
            )
        )  # EE1EF

    def forward(self, x):
        return self.list[0](x) + self.list[1](x)


class Channels3(nn.Module):
    def __init__(self):
        super(Channels3, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                Channels2(),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                nn.UpsamplingNearest2d(scale_factor=2)
            )
        )  # BD2EG
        self.list.append(
            nn.Sequential(
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]])
            )
        )  # BC

    def forward(self, x):
        return self.list[0](x) + self.list[1](x)


class Channels4(nn.Module):
    def __init__(self):
        super(Channels4, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                Channels3(),
                inception(128, [[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]]),
                inception(128, [[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]]),
                nn.UpsamplingNearest2d(scale_factor=2)
            )
        )  # BB3BA
        self.list.append(
            nn.Sequential(
                inception(128, [[16], [3, 64, 16], [7, 64, 16], [11, 64, 16]])
            )
        )  # A

    def forward(self, x):
        return self.list[0](x) + self.list[1](x)


class Hourglass(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.seq = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding=3),
            nn.SyncBatchNorm(128),
            nn.ReLU(True),
            Channels4(),
        )
        self.output = nn.Conv2d(64, self.num_layers - 1 + 3, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    def forward(self, x):
        x = self.output(self.seq(x))
        return self.output_bias(x, self.num_layers)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)


class HourglassF(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.seq = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding=3),
            nn.SyncBatchNorm(128),
            nn.ReLU(True),
            Channels4(),
        )
        self.output = nn.Conv2d(64, self.num_layers - 1 + 3, 3, padding=1)
        self.output_bias = apply_harmonic_bias

    def forward(self, x):
        x = self.output(self.seq(x))
        return self.output_bias(x, self.num_layers)

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
