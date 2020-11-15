import torch
import torch.nn as nn
import torch.nn.functional as torchf


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


def up(t: torch.Tensor):
    return torch.repeat_interleave(torch.repeat_interleave(t, 2, dim=-1), 2, dim=-2)


def apply_harmonic_bias(inputs: torch.Tensor, num_layers):
    alpha = torch.tensor(1.) / torch.arange(2, num_layers + 1, dtype=torch.float32)
    shift = torch.atanh(2. * alpha - 1.)
    no_shift = torch.zeros([inputs.shape[1] - (num_layers - 1)])
    shift = torch.cat([shift, no_shift], dim=-1).type_as(inputs).reshape(1, -1, 1, 1)
    out = inputs + shift
    return (torch.tanh(out) + 1.) / 2.0


class inception(nn.Module):
    def __init__(self, input_size, config):
        self.config = config
        super(inception, self).__init__()
        self.convs = nn.ModuleList()

        # Base 1*1 conv layer
        self.convs.append(nn.Sequential(
            nn.Conv2d(input_size, config[0][0], 1),
            nn.BatchNorm2d(config[0][0], affine=False),
            nn.ReLU(True),
        ))

        # Additional layers
        for i in range(1, len(config)):
            filt = config[i][0]
            pad = int((filt - 1) / 2)
            out_a = config[i][1]
            out_b = config[i][2]
            conv = nn.Sequential(
                nn.Conv2d(input_size, out_a, 1),
                nn.BatchNorm2d(out_a, affine=False),
                nn.ReLU(True),
                nn.Conv2d(out_a, out_b, filt, padding=pad),
                nn.BatchNorm2d(out_b, affine=False),
                nn.ReLU(True)
            )
            self.convs.append(conv)

    def __repr__(self):
        return "inception" + str(self.config)

    def forward(self, x):
        ret = []
        for conv in self.convs:
            ret.append(conv(x))
        # print(torch.cat(ret,dim=1))
        return torch.cat(ret, dim=1)
