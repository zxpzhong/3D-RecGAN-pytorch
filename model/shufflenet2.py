# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 1:14
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : shufflenet2.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.nn.init as init

from params import Args


def conv3x3(in_channels, out_channels, stride, padding=1, groups=1):
    '''
    3x3 convolution
    :param in_channels: 输入通道数
    :param out_channels:  输出通道数
    :param stride:  步长
    :param padding:  补齐
    :param groups: 组卷积组数
    :return:
    '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    '''
    1x1 convolution
    :param in_channels: 输入通道数
    :param out_channels:  输出通道数
    :param stride:  步长
    :return:
    '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class ShuffleUnit(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super(ShuffleUnit, self).__init__()

        assert (stride == 1 or stride == 2), 'stride must be either 1 or 2.'

        if stride == 2:
            outputs = outputs // 2
            # stride=2，做下采样。左分支加入下采样模块。
            self.downsample = nn.Sequential(
                conv3x3(in_channels=inputs, out_channels=inputs, stride=2, padding=1, groups=inputs),
                nn.BatchNorm2d(inputs),
                conv1x1(in_channels=inputs, out_channels=outputs),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True)
            )
        elif stride == 1:
            # stride=1，不做下采样。加入channel_split。左分支不做处理。
            inputs = inputs // 2
            outputs = outputs // 2
            self.downsample = None

        # print('inputs: {}, outputs: {}'.format(inputs, outputs))
        self.conv1x1_1 = conv1x1(in_channels=inputs, out_channels=outputs)
        self.conv1x1_bn_1 = nn.BatchNorm2d(outputs)

        self.dwconv3x3 = conv3x3(in_channels=outputs, out_channels=outputs, stride=stride, groups=outputs)
        self.dwconv3x3_bn = nn.BatchNorm2d(outputs)

        self.conv1x1_2 = conv1x1(in_channels=outputs, out_channels=outputs)
        self.conv1x1_bn_2 = nn.BatchNorm2d(outputs)

        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _channel_split(features, ratio=0.5):
        split_idx = int(features.size(1) * ratio)
        return features[:, :split_idx], features[:, split_idx:]

    @staticmethod
    def _channel_shuffle(features, groups=2):
        batch_size, num_channels, height, width = features.data.size()

        channels_per_group = num_channels // groups

        # reshape
        features = features.view(batch_size, groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        features = torch.transpose(features, 1, 2).contiguous()

        # flatten
        features = features.view(batch_size, -1, height, width)

        return features

    def forward(self, x):
        if self.downsample:
            x1 = x
            x2 = x
        else:
            x1, x2 = self._channel_split(x)
            # print('x1: {}, x2: {}'.format(x1.shape, x2.shape))

        # right branch
        x2 = self.conv1x1_1(x2)
        x2 = self.conv1x1_bn_1(x2)
        x2 = self.relu(x2)
        # print('x2: {}'.format(x2.shape))

        x2 = self.dwconv3x3(x2)
        x2 = self.dwconv3x3_bn(x2)
        # print('x2: {}'.format(x2.shape))

        x2 = self.conv1x1_2(x2)
        x2 = self.conv1x1_bn_2(x2)
        x2 = self.relu(x2)
        # print('x2: {}'.format(x2.shape))

        # left branch
        if self.downsample:
            x1 = self.downsample(x1)

        # print('x1: {}, x2: {}'.format(x1.shape, x2.shape))
        x = torch.cat([x1, x2], dim=1)
        x = self._channel_shuffle(x)
        return x


class ShuffleNet2(nn.Module):
    def __init__(self, feature_dim, layers_num, num_classes=1000):
        super(ShuffleNet2, self).__init__()

        dim1, dim2, dim3, dim4, dim5 = feature_dim

        self.conv1 = conv3x3(in_channels=3, out_channels=dim1, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_layer(feature_dim[0], feature_dim[1], layers_num[0])
        self.stage3 = self._make_layer(feature_dim[1], feature_dim[2], layers_num[1])
        self.stage4 = self._make_layer(feature_dim[2], feature_dim[3], layers_num[2])

        self.conv5 = conv1x1(in_channels=feature_dim[3], out_channels=feature_dim[4])
        self.drop5 = nn.Dropout(0.5)
        # self.globalpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc6 = nn.Linear(feature_dim[4], 256)
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(256, num_classes)

    def _make_layer(self, dim1, dim2, blocks_num):
        layers = []
        layers.append(ShuffleUnit(dim1, dim2, stride=2))
        for i in range(blocks_num):
            layers.append(ShuffleUnit(dim2, dim2, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv5(x)
        x = self.drop5(x)
        # 全局池化
        x = nn.AvgPool2d(kernel_size=x.size(2), stride=1)(x)
        x = x.view(-1, x.size(1))

        x = self.fc6(x)
        feature = x
        x = self.drop6(x)

        x = self.fc7(x)
        x = F.log_softmax(x, dim=-1)

        return x, feature

default_config = {
    "0.5x": [24, 48,  96,  192, 1024],
    "1x":   [24, 116, 232, 464, 1024],
    "1.5x": [24, 176, 352, 704, 1024],
    "2x":   [24, 244, 488, 976, 2048]
}

class ShuffleNet2_Custom(nn.Module):
    def __init__(self, num_classes=107, mode="0.5x"):
        super(ShuffleNet2_Custom, self).__init__()

        self.net = ShuffleNet2(feature_dim=default_config[mode], layers_num=[3, 7, 3], num_classes=num_classes)

    def forward(self, x):
        x, feature = self.net(x)
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


class ShuffleNet2_mini(nn.Module):
    def __init__(self, num_classes=107):
        super(ShuffleNet2_mini, self).__init__()

        self.num_classes = num_classes

        self.conv_bn_1 = conv_bn(3, 32, 2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_shuffle_2 = ShuffleUnit(32, 32, 1)
        self.conv_bn_2 = conv_bn(32, 64, 1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_shuffle_3 = ShuffleUnit(64, 64, 1)
        self.conv_bn_3 = conv_bn(64, 128, 1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.drop_3 = nn.Dropout(Args.keep_prob)

        self.conv_shuffle_4 = ShuffleUnit(128, 128, 1)
        self.conv_bn_4 = conv_bn(128, 256, 1)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)
        self.drop_4 = nn.Dropout(Args.keep_prob)

        self.fc_5 = nn.Linear(11 * 11 * 256, 256)
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
        # print(x.shape)
        x = self.conv_bn_1(x)
        # print(x.shape)
        x = self.pool_1(x)
        # print(x.shape)

        x = self.conv_shuffle_2(x)
        # print(x.shape)
        x = self.conv_bn_2(x)
        x = self.pool_2(x)
        # print(x.shape)

        x = self.conv_shuffle_3(x)
        # print(x.shape)
        x = self.conv_bn_3(x)
        x = self.pool_3(x)
        x = self.drop_3(x)

        x = self.conv_shuffle_4(x)
        x = self.conv_bn_4(x)
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
    # net = ShuffleNet2(feature_dim=default_config['0.5x'], layers_num=[3, 7, 3], num_classes=107)
    # net = ShuffleNet2_Custom(num_classes=107, mode="0.5x")
    net = ShuffleNet2_mini(num_classes=107)

    summary(net, (3, 360, 360))

