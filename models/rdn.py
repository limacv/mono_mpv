import torch
from torch import nn
import torch.nn.functional as torchf
from ._modules import apply_harmonic_bias


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, in_chnl=3, mpi_layers=24, num_features=64, growth_rate=64, num_blocks=3, num_layers=7):
        super(RDN, self).__init__()
        self.num_layers = mpi_layers
        self.num_blocks = num_blocks

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(in_chnl, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(num_features, growth_rate, num_layers)])
        for _ in range(self.num_blocks - 1):
            self.rdbs.append(RDB(growth_rate, growth_rate, num_layers))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(growth_rate * self.num_blocks, num_features, kernel_size=1),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        )
        self.outcnl = mpi_layers - 1 + 3
        self.upscale = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        self.output = nn.Conv2d(num_features, self.outcnl, kernel_size=3, padding=3 // 2)
        self.outputbias = apply_harmonic_bias

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = torchf.relu(self.upscale(x))
        x = self.outputbias(self.output(x), self.num_layers)
        return x

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
