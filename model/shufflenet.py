# -*- coding: utf-8 -*-
# @Time    : 2018/11/14 13:53
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : shufflenet.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
from collections import OrderedDict

from params import Args


def conv3x3(in_channels, out_channels, stride=1, groups=1):
    '''
    3x3 conv with padding
    :param in_channels:
    :param out_channels:
    :param stride:
    :param groups:
    :return:
    '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False, groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    '''
    1x1 conv with padding
    :param in_channels:
    :param out_channels:
    :param groups:
    :return:
    '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=1, groups=groups)

def channel_shuffle(x, groups):
    '''
    channel shuffle
    :param x:
    :param groups:
    :return:
    '''
    batch_size, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    '''
    Shuffle Unit
    '''
    def __init__(self, in_channels, out_channels, groups=3, grouped_conv=True, combine='add'):
        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv1x1_compress = self._make_grouped_conv1x1(self.in_channels,
                                                             self.bottleneck_channels,
                                                             self.first_1x1_groups,
                                                             batch_norm=True,
                                                             relu=True)

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(self.bottleneck_channels, self.bottleneck_channels,
                                         stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv1x1_expand = self._make_grouped_conv1x1(self.bottleneck_channels,
                                                           self.out_channels,
                                                           self.groups,
                                                           batch_norm=True,
                                                           relu=False)

        self.bypass_conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, batch_norm=True, relu=False):
        modules = OrderedDict()

        modules['conv1x1'] = conv1x1(in_channels, out_channels, groups)

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)

        if relu:
            modules['relu'] = nn.ReLU()

        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return modules['conv1x1']

    # @staticmethod
    def _add(self, x, out):
        # print(x.size())
        # print(out.size())
        x_channels = x.size()[1]
        out_channels = out.size()[1]
        if x_channels != out_channels:
            # x = nn.Conv2d(x_channels, out_channels, kernel_size=1, bias=False)(x)
            x = self.bypass_conv1x1(x)
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):
        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)

        out = self.g_conv1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv1x1_expand(out)

        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNet(nn.Module):
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        '''
        ShuffleNet in paper
        :param groups: number of groups to be used in grouped 1x1 convolutions in each shuffle unit.
        :param in_channels: number of channels in the input tensor.
        :param n_classes: number of classes.
        '''

        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stage_repeat = [3, 7, 3]

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 576]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(in_channels, self.stage_out_channels[1], stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(stage=2)
        # Stage 3
        self.stage3 = self._make_stage(stage=3)
        # Stage 4
        self.stage4 = self._make_stage(stage=4)

        # Global pooling:
        # Undefined as PyTorch's functional API can be used for on-the-fly
        # shape inference if input size is not ImageNet's 224x224

        # Fully-connected classification layer
        num_inputs = self.stage_out_channels[-1]
        self.fc5 = nn.Linear(num_inputs, 256)
        self.drop5 = nn.Dropout(0.5)
        self.fc6 = nn.Linear(256, self.num_classes)

        # initialize parameters
        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = 'ShuffleUnit_Stage{}'.format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(self.stage_out_channels[stage-1], self.stage_out_channels[stage],
                                   groups=self.groups, grouped_conv=grouped_conv, combine='concat')

        modules['{}_0'.format(stage_name)] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeat[stage-2]):
            name = '{}_{}'.format(stage_name, i+1)
            modules[name] = ShuffleUnit(self.stage_out_channels[stage],
                                 self.stage_out_channels[stage],
                                 groups=self.groups,
                                 grouped_conv=True,
                                 combine='add')

        return nn.Sequential(modules)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # global average pool
        x = F.avg_pool2d(x, x.data.size()[-2:])

        # flatten
        x = x.view(x.size()[0], -1)
        x = self.fc5(x)
        feature = x
        x = self.drop5(x)
        x = self.fc6(x)
        x = F.log_softmax(x, dim=-1)

        return x, feature

def conv_bn(input, output, stride):
    '''
    普通卷积模块（conv + bn + relu）
    :param input: 输入
    :param output: 输出
    :param stride: 步长
    :return: 普通卷积block
    '''

    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(output),
        # inplace，默认设置为False，表示新创建一个对象对其修改，也可以设置为True，表示直接对这个对象进行修改
        nn.ReLU(inplace=True)
    )


class ShuffleNet_v1(nn.Module):
    '''
    ShuffleNet custom
    '''
    def __init__(self, num_classes=107):
        super(ShuffleNet_v1, self).__init__()

        self.num_classes = num_classes

        self.conv_bn_1 = conv_bn(3, 32, 2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_shuffle_2 = ShuffleUnit(32, 64, groups=4, grouped_conv=True, combine='add')
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_shuffle_3 = ShuffleUnit(64, 128, groups=4, grouped_conv=True, combine='add')
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.drop_3 = nn.Dropout(Args.keep_prob)
        self.conv_shuffle_4 = ShuffleUnit(128, 256, groups=4, grouped_conv=True, combine='add')
        self.pool_4 = nn.MaxPool2d(kernel_size=2)
        self.drop_4 = nn.Dropout(Args.keep_prob)

        self.fc_5 = nn.Linear(11*11*256, 256)
        self.drop_5 = nn.Dropout(Args.keep_prob)
        self.fc_6 = nn.Linear(256, self.num_classes)

        # 初始化网络权重参数
        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_bn_1(x)
        x = self.pool_1(x)

        x = self.conv_shuffle_2(x)
        x = self.pool_2(x)

        x = self.conv_shuffle_3(x)
        x = self.pool_3(x)
        x = self.drop_3(x)

        x = self.conv_shuffle_4(x)
        x = self.pool_4(x)
        x = self.drop_4(x)

        x = x.view(-1, 11 * 11 * 256)
        x = self.fc_5(x)
        feature = x
        x = self.drop_5(x)

        x = self.fc_6(x)
        x = F.log_softmax(x, dim=-1)
        return x, feature


if __name__ == '__main__':
    # net = ShuffleNet()
    net = ShuffleNet_v1()

    summary(net, (3, 360, 360))

    # x = torch.randn((8, 3, 360, 360))
    # output, embed = net(x)
    # print('input: {}'.format(x.shape))
    # print('output: {}'.format(output.shape))
    # print('embedding: {}'.format(embed.shape))

