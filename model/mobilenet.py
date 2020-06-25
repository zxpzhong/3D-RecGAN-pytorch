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
        nn.ReLU(inplace=True)
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
        nn.ReLU(inplace=True),

        nn.Conv2d(input, output, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True)
    )

class MobileNet_v1(nn.Module):
    '''
    MobileNet网络
    '''
    def __init__(self, num_classes=107):
        '''
        构造函数
        :param num_classes: 总类别数
        '''
        super(MobileNet_v1, self).__init__()

        self.num_classes = num_classes

        self.conv_bn_1 = conv_bn(3, 32, 2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_dw_2 = conv_dw(32, 64, 1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_dw_3 = conv_dw(64, 128, 1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.drop_3 = nn.Dropout(Args.keep_prob)
        self.conv_dw_4 = conv_dw(128, 256, 1)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)
        self.drop_4 = nn.Dropout(Args.keep_prob)

        self.fc_5 = nn.Linear(5*17*256, 256)
        self.drop_5 = nn.Dropout(Args.keep_prob)
        self.fc_6 = nn.Linear(256, self.num_classes)

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
        '''
        前向传播
        :param x: 输入数据，[batch_size, channels, img_height, img_width]
        :return: 预测结果(softmax)，[batch_size, num_classes]
        '''
        x = self.conv_bn_1(x)
        x = self.pool_1(x)

        x = self.conv_dw_2(x)
        x = self.pool_2(x)

        x = self.conv_dw_3(x)
        x = self.pool_3(x)
        x = self.drop_3(x)

        x = self.conv_dw_4(x)
        x = self.pool_4(x)
        x = self.drop_4(x)

        x = x.view(-1, 5*17*256)
        x = self.fc_5(x)
        feature = x
        x = self.drop_5(x)

        x = self.fc_6(x)
        x = F.log_softmax(x, dim=-1)
        return x, feature

    # def get_embedding(self, x):
    #     x = self.conv_bn_1(x)
    #     x = self.pool_1(x)
    #
    #     x = self.conv_dw_2(x)
    #     x = self.pool_2(x)
    #
    #     x = self.conv_dw_3(x)
    #     x = self.pool_3(x)
    #     x = self.drop_3(x)
    #
    #     x = self.conv_dw_4(x)
    #     x = self.pool_4(x)
    #     x = self.drop_4(x)
    #
    #     x = x.view(-1, 11 * 11 * 256)
    #     x = self.fc_5(x)
    #     return x

if __name__ == '__main__':
    net = MobileNet_v1(num_classes=107)

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
