import torch.nn as nn
import torch
from ._modules import *


def initial_weights(model: nn.ModuleDict):
    for k, m in model.items():
        m.initial_weights()


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
