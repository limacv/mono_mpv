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
    return inputs + shift
