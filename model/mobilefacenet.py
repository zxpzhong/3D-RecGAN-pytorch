# -*- coding:UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

from params import Args
import sys
import platform
if platform.python_version().split('.')[0] == '2':
    sys.path.append('./')
    from center_loss import CenterLoss
else:
    from Model.center_loss import CenterLoss

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
        nn.PReLU()
    )

def conv_dw(input, output, stride):
    '''
    深度可分离卷积模块(depthwise separable convolution):
        depthwise convolution:
            dw_conv + bn + relu
        pointwise convolution:
            1*1 conv + bn + relu
    :param input: 输入
    :param output: 输出
    :param stride: 步长
    :return: 深度可分离卷积block
    '''

    return nn.Sequential(
        nn.Conv2d(input, input, kernel_size=3, stride=stride, padding=1, groups=input, bias=False),
        nn.BatchNorm2d(input),
        nn.PReLU(),

        nn.Conv2d(input, output, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(output),
        nn.PReLU()
    )

def global_dw_conv(input, output, kernel_size):
    '''
    GDC 全局深度可分离卷积
    :param input: 输入
    :param output: 输出
    :param kernel_size: 卷积核尺寸，等于图片大小
    :return:
    '''
    return nn.Sequential(
        nn.Conv2d(input, input, kernel_size=kernel_size, stride=1, padding=0, groups=input, bias=False),
        nn.BatchNorm2d(input),
        nn.PReLU(),

        nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(output),
        nn.PReLU()
    )

class Bottleneck(nn.Module):
    '''
    Inverted Residual Block
    '''
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_factor = expansion_factor
        self.kernel_size = kernel_size
        self.stride = stride

        if self.stride != 1 and self.stride != 2:
            raise ValueError('Stride should be 1 or 2')

        # Inverted Residual Block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, padding=kernel_size // 2,
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            # Linear Bottleneck，这里不接ReLU6
            # nn.ReLU6(inplace=True)
        )

        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        self.bypass_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )


        # self.if_match_bypss = True if in_channels != out_channels else False
        # if self.if_match_bypss:
        #     self.bypass_conv = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )

    def forward(self, x):

        # print('\n')
        # print('in_channels: {}'.format(self.in_channels))
        # print('out_channels: {}'.format(self.out_channels))
        # print('expansion_factor: {}'.format(self.expansion_factor))
        # print('kernel_size: {}'.format(self.kernel_size))
        # print('stride: {}'.format(self.stride))

        # print('x: {}'.format(x.shape))
        output = self.block(x)
        # print('output: {}'.format(output.shape))
        # if self.if_match_bypss:
        #     return output + self.bypass_conv(x)
        # else:
        #     return output + x

        if self.stride == 2:
            return output
        else:
            if self.in_channels != self.out_channels:
                return output + self.bypass_conv(x)
            else:
                return output + x

def bottleneck_sequence(in_channels, out_channels, num_units, expansion_factor=6, kernel_size=3, initial_stride=2):
    '''
    bottleneck序列
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param num_units:   bottleneck单元数
    :param expansion_factor: 扩张参数
    :param kernel_size:  卷积核尺寸，默认为3
    :param initial_stride:  初始步长
    :return:
    '''
    # print(kernel_size)
    bottleneck_arr = [
        Bottleneck(in_channels, out_channels, expansion_factor, kernel_size, initial_stride)
    ]

    for i in range(num_units - 1):
        bottleneck_arr.append(
            Bottleneck(out_channels, out_channels, expansion_factor, kernel_size, 1)
        )

    return bottleneck_arr

class MobileFaceNet(nn.Module):
    def __init__(self, num_classes=107):
        super(MobileFaceNet, self).__init__()

        self.num_classes = num_classes

        self.network_sequence = []
        self.network_sequence.append(
            conv_bn(input=3, output=64, stride=2)
        )
        self.network_sequence.append(
            conv_dw(input=64, output=64, stride=1)
        )
        self.bottleneck_dict = [
            {'in_channels': 64,  'out_channels': 64,  'num_units': 5, 'expansion_factor': 2, 'kernel_size': 3, 'initial_stride': 2},
            {'in_channels': 64,  'out_channels': 128, 'num_units': 1, 'expansion_factor': 4, 'kernel_size': 3, 'initial_stride': 2},
            {'in_channels': 128, 'out_channels': 128, 'num_units': 3, 'expansion_factor': 2, 'kernel_size': 3, 'initial_stride': 2},
            {'in_channels': 128, 'out_channels': 128, 'num_units': 3, 'expansion_factor': 2, 'kernel_size': 3, 'initial_stride': 1},
            {'in_channels': 128, 'out_channels': 128, 'num_units': 1, 'expansion_factor': 4, 'kernel_size': 3, 'initial_stride': 2},
            {'in_channels': 128, 'out_channels': 128, 'num_units': 2, 'expansion_factor': 2, 'kernel_size': 3, 'initial_stride': 1},
        ]
        for i in range(0, 6):
            self.network_sequence.extend(
                bottleneck_sequence(self.bottleneck_dict[i]['in_channels'],
                                    self.bottleneck_dict[i]['out_channels'],
                                    self.bottleneck_dict[i]['num_units'],
                                    self.bottleneck_dict[i]['expansion_factor'],
                                    self.bottleneck_dict[i]['kernel_size'],
                                    self.bottleneck_dict[i]['initial_stride'])
            )
        self.network_base = nn.Sequential(*self.network_sequence)

        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.PReLU()

        self.gdconv_2 = global_dw_conv(input=256, output=256, kernel_size=12)
        self.drop_2 = nn.Dropout(Args.keep_prob)

        # linear conv 1*1
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.relu_3 = nn.PReLU()
        self.drop_3 = nn.Dropout(Args.keep_prob)

        self.conv_4 = nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1, stride=1)

        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.network_base(x)
        # print('x: {}'.format(x.shape))

        x = self.relu_1(self.bn_1(self.conv_1(x)))

        x = self.drop_2(self.gdconv_2(x))

        x = self.relu_3(self.bn_3(self.conv_3(x)))
        feature = x.view(-1, 256)
        x = self.drop_3(x)

        x = self.conv_4(x)
        x = x.view(-1, self.num_classes)
        x = F.log_softmax(x, dim=-1)

        return x, feature

if __name__ == '__main__':
    net = MobileFaceNet(num_classes=107)

    # summary(net, (3, 360, 360))

    data = torch.rand((8, 3, 360, 360))
    output, embed = net(data)
    print('input: {}'.format(data.shape))
    print('output: {}'.format(output.shape))
    # print(output)

    # embed = net.get_embedding(data)
    print('embedding: {}'.format(embed.shape))

    loss = CenterLoss(num_classes=107, feat_dim=256)
    labels = torch.Tensor(np.random.randint(low=0, high=107, size=8)).long()
    print(labels.shape)
    loss_out = loss(embed, labels)
    print(loss_out)