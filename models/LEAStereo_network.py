import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from collections import namedtuple


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self._initialize_weights()

    def forward(self, x):
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv3d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(C_out)
        self._initialize_weights()

    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv3d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


PRIMITIVES = [
    'skip_connect',
    '3d_conv_3x3'
]
OPS = {
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    '3d_conv_3x3': lambda C, stride: ConvBR(C, C, 3, stride, 1)
}
OPS3 = {
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    '3d_conv_3x3': lambda C, stride: ConvBR3(C, C, 3, stride, 1)
}


class SkipModel3DCell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        super(SkipModel3DCell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_preprocess = ConvBR3(self.C_prev_prev, self.C_out, 1, 1, 0)
        self.preprocess = ConvBR3(self.C_prev, self.C_out, 1, 1, 0)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS3[primitive](self.C_out, stride=1)
            self._ops.append(op)

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_d = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_h = self.scale_dimension(s1.shape[3], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[4], self.scale)
            s1 = F.interpolate(s1, [feature_size_d, feature_size_h, feature_size_w], mode='trilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]) or (s0.shape[4] != s1.shape[4]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3], s1.shape[4]),
                                            mode='trilinear', align_corners=True)
        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
        return prev_input, concat_feature


class newModel2DCell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        super(newModel2DCell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_preprocess = ConvBR(self.C_prev_prev, self.C_out, 1, 1, 0)
        self.preprocess = ConvBR(self.C_prev, self.C_out, 1, 1, 0)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out, stride=1)
            self._ops.append(op)

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='bilinear', align_corners=True)

        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
        return prev_input, concat_feature


class ConvBR(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bn=True, relu=True):
        super(ConvBR, self).__init__()
        self.relu = relu
        self.use_bn = bn

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ConvBR3(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bn=True, relu=True):
        super(ConvBR3, self).__init__()
        self.relu = relu
        self.use_bn = bn

        self.conv = nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(C_out)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class newMatching(nn.Module):
    def __init__(self, network_arch, cell_arch, cell=SkipModel3DCell, args=None):
        super(newMatching, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = args.mat_step
        self._num_layers = args.mat_num_layers
        self._block_multiplier = args.mat_block_multiplier
        self._filter_multiplier = args.mat_filter_multiplier

        initial_fm = self._filter_multiplier * self._block_multiplier
        half_initial_fm = initial_fm // 2

        self.stem0 = ConvBR3(initial_fm * 2, initial_fm, 3, stride=1, padding=1)
        self.stem1 = ConvBR3(initial_fm, initial_fm, 3, stride=1, padding=1)

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()
            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier,
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier * filter_param_dict[level],
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 initial_fm / self._block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier * filter_param_dict[level],
                                 downup_sample, self.args)
                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_prev_level],
                                 self._filter_multiplier *
                                 filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier * filter_param_dict[level], downup_sample, self.args)

            self.cells += [_cell]

        self.last_3 = ConvBR3(initial_fm, 1, 3, 1, 1, bn=False, relu=False)
        self.last_6 = ConvBR3(initial_fm * 2, initial_fm, 1, 1, 0)
        self.last_12 = ConvBR3(initial_fm * 4, initial_fm * 2, 1, 1, 0)
        self.last_24 = ConvBR3(initial_fm * 8, initial_fm * 4, 1, 1, 0)

        self.conv1 = ConvBR3(initial_fm * 4, initial_fm * 2, 3, 1, 1)
        self.conv2 = ConvBR3(initial_fm * 4, initial_fm * 2, 3, 1, 1)

    def forward(self, x):
        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        out = (stem0, stem1)
        out0 = self.cells[0](out[0], out[1])
        out1 = self.cells[1](out0[0], out0[1])
        out2 = self.cells[2](out1[0], out1[1])
        out3 = self.cells[3](out2[0], out2[1])
        out4 = self.cells[4](out3[0], out3[1])

        out4_cat = self.conv1(torch.cat((out1[-1], out4[-1]), 1))
        out5 = self.cells[5](out4[0], out4_cat)
        out6 = self.cells[6](out5[0], out5[1])
        out7 = self.cells[7](out6[0], out6[1])
        out8 = self.cells[8](out7[0], out7[1])
        out8_cat = self.conv2(torch.cat((out4[-1], out8[-1]), 1))
        out9 = self.cells[9](out8[0], out8_cat)
        out10 = self.cells[10](out9[0], out9[1])
        out11 = self.cells[11](out10[0], out10[1])
        last_output = out11[-1]

        d, h, w = x.size()[2], x.size()[3], x.size()[4]
        upsample_6 = nn.Upsample(size=x.size()[2:], mode='trilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[d // 2, h // 2, w // 2], mode='trilinear', align_corners=True)
        upsample_24 = nn.Upsample(size=[d // 4, h // 4, w // 4], mode='trilinear', align_corners=True)

        if last_output.size()[3] == h:
            mat = self.last_3(last_output)
        elif last_output.size()[3] == h // 2:
            mat = self.last_3(upsample_6(self.last_6(last_output)))
        elif last_output.size()[3] == h // 4:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(last_output)))))
        elif last_output.size()[3] == h // 8:
            mat = self.last_3(
                upsample_6(self.last_6(upsample_12(self.last_12(upsample_24(self.last_24(last_output)))))))
        return mat


class newFeature(nn.Module):
    def __init__(self, network_arch, cell_arch, cell=newModel2DCell, args=None):
        super(newFeature, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier

        initial_fm = self._filter_multiplier * self._block_multiplier
        half_initial_fm = initial_fm // 2

        self.stem0 = ConvBR(3, half_initial_fm, 3, stride=1, padding=1)
        self.stem1 = ConvBR(half_initial_fm, initial_fm, 3, stride=3, padding=1)
        self.stem2 = ConvBR(initial_fm, initial_fm, 3, stride=1, padding=1)

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}

        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier,
                             self.cell_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 initial_fm / self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample, self.args)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)

            self.cells += [_cell]

        self.last_3 = ConvBR(initial_fm, initial_fm, 1, 1, 0, bn=False, relu=False)
        self.last_6 = ConvBR(initial_fm * 2, initial_fm, 1, 1, 0)
        self.last_12 = ConvBR(initial_fm * 4, initial_fm * 2, 1, 1, 0)
        self.last_24 = ConvBR(initial_fm * 8, initial_fm * 4, 1, 1, 0)

    def forward(self, x):
        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        stem2 = self.stem2(stem1)
        out = (stem1, stem2)

        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1])

        last_output = out[-1]

        h, w = stem2.size()[2], stem2.size()[3]
        upsample_6 = nn.Upsample(size=stem2.size()[2:], mode='bilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[h // 2, w // 2], mode='bilinear', align_corners=True)
        upsample_24 = nn.Upsample(size=[h // 4, w // 4], mode='bilinear', align_corners=True)

        if last_output.size()[2] == h:
            fea = self.last_3(last_output)
        elif last_output.size()[2] == h // 2:
            fea = self.last_3(upsample_6(self.last_6(last_output)))
        elif last_output.size()[2] == h // 4:
            fea = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(last_output)))))
        elif last_output.size()[2] == h // 8:
            fea = self.last_3(
                upsample_6(self.last_6(upsample_12(self.last_12(upsample_24(self.last_24(last_output)))))))

        return fea

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params


def network_layer_to_space(net_arch):
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    """
        return:
        network_space[layer][level][sample]:
        layer: 0 - 12
        level: sample_level {0: 1, 1: 2, 2: 4, 3: 8}
        sample: 0: down 1: None 2: Up
    """
    return space


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),
                                 [1, self.maxdisp, 1, 1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out


class Disp(nn.Module):
    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3] * 3, x.size()[4] * 3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)
        x = self.disparity(x)
        return x


class LEAStereo(nn.Module):
    def __init__(self, args):
        super(LEAStereo, self).__init__()

        network_path_fea, cell_arch_fea = np.load(args.net_arch_fea), np.load(args.cell_arch_fea)
        network_path_mat, cell_arch_mat = np.load(args.net_arch_mat), np.load(args.cell_arch_mat)
        print('Feature network path:{}\nMatching network path:{} \n'.format(network_path_fea, network_path_mat))

        network_arch_fea = network_layer_to_space(network_path_fea)
        network_arch_mat = network_layer_to_space(network_path_mat)

        self.maxdisp = args.maxdisp
        self.feature = newFeature(network_arch_fea, cell_arch_fea, args=args)
        self.matching = newMatching(network_arch_mat, cell_arch_mat, args=args)
        self.disp = Disp(self.maxdisp)

    def forward(self, x, y):
        x = self.feature(x)
        y = self.feature(y)

        cost = x.new().resize_(x.size()[0], x.size()[1] * 2, int(self.maxdisp / 3), x.size()[2],
                               x.size()[3]).zero_()
        for i in range(int(self.maxdisp / 3)):
            if i > 0:
                cost[:, :x.size()[1], i, :, i:] = x[:, :, :, i:]
                cost[:, x.size()[1]:, i, :, i:] = y[:, :, :, :-i]
            else:
                cost[:, :x.size()[1], i, :, i:] = x
                cost[:, x.size()[1]:, i, :, i:] = y

        cost = self.matching(cost)
        disp = self.disp(cost)
        return disp

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params
