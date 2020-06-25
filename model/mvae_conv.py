from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

from efficientnet_pytorch import EfficientNet
from Model.unet_model import *
from Model.resnet_all import *
from Model.mobilenetv3 import *

class Bottleneck(nn.Module):
    '''
    Inverted Residual Block
    '''
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2):
        super(Bottleneck, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError('Stride should be 1 or 2')

        # Inverted Residual Block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, padding=int((kernel_size-1)/2),
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # Linear Bottleneck，这里不接ReLU6
            # nn.ReLU6(inplace=True)
        )

        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        self.if_match_bypss = True if in_channels != out_channels else False
        # if self.if_match_bypss:
        if True:
            self.bypass_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        output = self.block(x)
        # if self.if_match_bypss:
        if True:
            return output + self.bypass_conv(x)
        else:
            return output + x

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
        nn.LeakyReLU(inplace=True)
    )

def conv_bottleneck(input, output, stride):
    return Bottleneck(in_channels=input, out_channels=output, stride=stride)

def debug(str):
    if False:
        print(str)


d = 2

class MV2Encoder(nn.Module):
    def __init__(self,latent_num = 256):
        super(MV2Encoder, self).__init__()

        self.latent_num = latent_num

        # self.conv_bn_1 = conv_bn(1, d, 2)
        # self.pool_1 = nn.MaxPool2d(kernel_size=2)
        # self.conv_bottleneck_2 = conv_bottleneck(d, d*2, 2)
        # self.pool_2 = nn.MaxPool2d(kernel_size=2)
        # self.conv_bottleneck_3 = conv_bottleneck(d*2, d*4, 2)
        # # self.pool_3 = nn.MaxPool2d(kernel_size=2)
        # self.conv_bottleneck_4 = conv_bottleneck(d*4, d*8, 2)
        # self.conv_bottleneck_4_1 = conv_bottleneck(d*8, d*32, 2)
        # # self.pool_4 = nn.MaxPool2d(kernel_size=2)
        # self.conv_bottleneck_5 = conv_bottleneck(d*32, d*128, 2)
        # # self.conv_bottleneck_5_1 = conv_bottleneck(d*32, d*64, 2)
        #
        # # self.conv_bottleneck_6 = conv_bottleneck(d * 64, d * 128, 2)
        # # self.pool_5 = nn.MaxPool2d(kernel_size=2)
        # # self.fc_5 = nn.Linear(13824, self.latent_num)
        # # self.drop_5 = nn.Dropout(0.5)
        d = 4
        self.inc = inconv(1, d)
        self.down1 = Bottleneck(d, d*2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.down2 = Bottleneck(d*2, d*4)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.down3 = Bottleneck(d*4, d*8)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.down4 = Bottleneck(d*8, d*16)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)
        self.down5 = Bottleneck(d*16, d*32)
        self.pool_5 = nn.MaxPool2d(kernel_size=2)


        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self,x):
        # x = self.conv_bn_1(input)
        # x = self.pool_1(x)
        # x = self.conv_bottleneck_2(x)
        # x = self.pool_2(x)
        # x = self.conv_bottleneck_3(x)
        # # x = self.pool_3(x)
        # x = self.conv_bottleneck_4(x)
        # x = self.conv_bottleneck_4_1(x)
        # x = self.pool_4(x)
        # x = self.conv_bottleneck_5(x)
        # x = self.conv_bottleneck_5_1(x)

        # x = self.conv_bottleneck_6(x)
        # x = self.pool_5(x)
        # x = x.view(x.shape[0], -1)
        # feature = self.fc_5(x)
        # feature = self.drop_5(feature)
        # print(x.shape)

        x = self.inc(x)
        x = self.pool_1(x)
        debug('x1 shape is {}'.format(x.shape))
        x = self.down1(x)
        x = self.pool_1(x)
        debug('x2 shape is {}'.format(x.shape))
        x = self.down2(x)
        x = self.pool_1(x)
        debug('x3 shape is {}'.format(x.shape))
        x = self.down3(x)
        x = self.pool_1(x)
        debug('x4 shape is {}'.format(x.shape))
        x = self.down4(x)
        x = self.pool_1(x)
        debug('x5 shape is {}'.format(x.shape))
        x = self.down5(x)
        x = self.pool_1(x)
        debug('x6 shape is {}'.format(x.shape))
        return x



class MV2Discriminator(nn.Module):
    def __init__(self,latent_num = 256):
        super(MV2Discriminator, self).__init__()

        self.latent_num = latent_num

        self.conv_bn_1 = conv_bn(1, d, 2)
        # self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_2 = conv_bottleneck(d, d*2, 2)
        # self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_3 = conv_bottleneck(d*2, d*4, 2)
        # self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_4 = conv_bottleneck(d*4, d*8, 2)
        # self.conv_bottleneck_4_1 = conv_bottleneck(256, 256, 2)
        # self.pool_4 = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_5 = conv_bottleneck(d*8, d*16, 2)
        # self.conv_bottleneck_5_1 = conv_bottleneck(512, 512, 2)
        # self.pool_5 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(self.latent_num, 2)
        # self.drop_5 = nn.Dropout(0.5)

        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self,input):
        x = self.conv_bn_1(input)
        # x = self.pool_1(x)
        x = self.conv_bottleneck_2(x)
        # x = self.pool_2(x)
        x = self.conv_bottleneck_3(x)
        # x = self.pool_3(x)
        x = self.conv_bottleneck_4(x)
        # x = self.conv_bottleneck_4_1(x)
        # x = self.pool_4(x)
        x = self.conv_bottleneck_5(x)
        # x = self.conv_bottleneck_5_1(x)
        # x = self.pool_5(x)
        # x = x.view(x.shape[0], -1)
        feature = self.fc(x)
        # feature = self.drop_5(feature)
        # print(x.shape)
        return feature


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()

        # self.deconv_7_bn = nn.BatchNorm2d(d * 32)
        # self.deconv_7 = nn.ConvTranspose2d(d*64, d*32, 4, 2, 1)
        # self.deconv_6_bn = nn.BatchNorm2d(d*8)
        # self.deconv_6 = nn.ConvTranspose2d(d*32, d*8, 4, 2, 1)
        # self.deconv1 = nn.ConvTranspose2d(d*8, d*4, 7, 5, 1)
        # self.deconv1_bn = nn.BatchNorm2d(d*4)
        # self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 7, 5, 1)
        # self.deconv2_bn = nn.BatchNorm2d(d * 2)
        # self.deconv3 = nn.ConvTranspose2d(d * 2, d, 2, 1, 1)
        # self.deconv3_bn = nn.BatchNorm2d(d)
        # self.deconv4 = nn.ConvTranspose2d(d, 1, 2, 1, 0)

        self.deconv_8_bn = nn.BatchNorm2d(d * 64)
        self.deconv_8 = nn.ConvTranspose2d(d*128, d*64, 4, 4, 0)


        self.deconv_7_bn = nn.BatchNorm2d(d * 32)
        self.deconv_7 = nn.ConvTranspose2d(d*64, d*32, 4, 4, 0)
        self.deconv_6_bn = nn.BatchNorm2d(d*2)
        self.deconv_6 = nn.ConvTranspose2d(d*32, d*2, 2, 2, 0)

        self.deconv1 = nn.ConvTranspose2d(d*2, d*1, 2, 2, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*1)
        self.deconv2 = nn.ConvTranspose2d(d*1, 1, 2, 2, 0)
        self.deconv2_bn = nn.BatchNorm2d(1)
        self.deconv3 = nn.ConvTranspose2d(1, 1, 2, 2, 0)
        # self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 2, 2, 0)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        # self.deconv5 = nn.ConvTranspose2d(d, 1, 2, 2, 0)
        # self.deconv5_bn = nn.BatchNorm2d(1)
        # self.deconv6 = nn.ConvTranspose2d(1, 1, 2, 2, 0)

        # self.conv1 = nn.Conv2d(d,2,3,1,1)
        # self.conv1_bn = nn.BatchNorm2d(2)
        # self.conv2 = nn.Conv2d(2, 1, 3, 1, 1)

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
        # debug('input : {}'.format(input.size()))
        # x = input.view(input.size()[0], 128, 1, 3)
        debug('x : {}'.format(x.size()))
        x = F.relu(self.deconv_8_bn(self.deconv_8(x)))
        debug('x : {}'.format(x.size()))
        x = F.relu((self.deconv_7(x)))
        debug('x : {}'.format(x.size()))
        x = F.relu((self.deconv_6(x)))
        debug('x : {}'.format(x.size()))
        x = F.relu((self.deconv1(x)))
        debug('x : {}'.format(x.size()))
        x = F.relu((self.deconv2(x)))
        debug('x : {}'.format(x.size()))
        # x = F.relu((self.deconv3(x)))
        # debug('x : {}'.format(x.size()))
        # x = F.relu((self.deconv4(x)))
        # debug('x : {}'.format(x.size()))
        # x = F.relu(self.deconv5_bn(self.deconv5(x)))
        # debug('x : {}'.format(x.size()))
        x = self.deconv3(x)
        # debug('x : {}'.format(x.size()))
        # x = F.relu(self.conv1_bn(self.conv1(x)))
        # debug('x : {}'.format(x.size()))
        # x = F.relu((self.conv2(x)))
        debug('x : {}'.format(x.size()))
        # return F.sigmoid(x)
        return x


class FcClassify(nn.Module):
    def __init__(self,latent_num,classes):
        super(FcClassify, self).__init__()

        self.feature_classification = nn.Linear(latent_num, classes)

        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        logits = self.feature_classification(feature)
        logits = F.log_softmax(logits,dim=-1)
        return logits


# class Encoder_eff(nn.Module):
#     def __init__(self,latent_num):
#         super(FcClassify, self).__init__()
#         self.eff = EfficientNet.from_pretrained('efficientnet-b2').cuda()
#         self.feature_trans = nn.Linear(90112,latent_num)
#     def forward(self, x):
#         x = self.eff.extract_features(x)
#         x = self.feature_trans(x)
#         return logits

class Encoder_MV3(nn.Module):
    def __init__(self,latent_num):
        super(Encoder_MV3, self).__init__()
        self.encoder = mobilenetv3(mode='large')
    def forward(self, x):
        x = self.encoder(x)
        return x
    def extract_features(self, x):
        x = self.encoder(x)
        return x
class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class Encoder_eff(nn.Module):
    def __init__(self):
        super(Encoder_eff, self).__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b1')
        # self.conv_bottleneck_2 = conv_bottleneck(1280, 512, 2)
        # self.conv_bottleneck_3 = conv_bottleneck(512, 128, 2)
        # self.conv_bottleneck_4 = conv_bottleneck(512, 128, 2)
        # self.sonnet = ADDneck(1408, 1024)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.gapool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.encoder.extract_features(x)
        # x = self.maxpool(x)
        # x = self.conv_bottleneck_2(x)
        # x = self.maxpool(x)
        # x = self.conv_bottleneck_3(x)
        # x = self.maxpool(x)
        # x = self.conv_bottleneck_4(x)
        # x = self.sonnet(x)
        # x = self.avgpool(x)
        x = self.gapool(x)
        x = x.view(x.size(0), -1)
        return x
    def extract_features(self, x):
        x = self.encoder.extract_features(x)
        # x = self.maxpool(x)
        # x = self.conv_bottleneck_2(x)
        # x = self.maxpool(x)
        # x = self.conv_bottleneck_3(x)
        # x = self.maxpool(x)
        # x = self.conv_bottleneck_4(x)
        # x = self.sonnet(x)
        # x = self.avgpool(x)
        x = self.gapool(x)
        x = x.view(x.size(0), -1)
        return x

class AE():
    def __init__(self, classes , latent_num = 256):
        # self.encoder = Encoder_MV3(latent_num).cuda()
        # self.encoder = MV2Encoder(latent_num = latent_num).cuda()
        # self.encoder = resnet152(pretrained=True, get_feature=True).cuda()
        # self.encoder.fc = torch.nn.Linear(2048, classes).cuda()
        # self.encoder = EfficientNet.from_pretrained('efficientnet-b2').cuda()
        self.encoder = Encoder_eff().cuda()
        # 换用resnet18预训练过的试试，图像size已经和imagenet数据集一样大了
        self.decoder = ConvDecoder().cuda()

        # self.unet = UNet(1,1).cuda()
        self.classify = FcClassify(latent_num=latent_num,classes=classes).cuda()
        self.discriminator = MV2Discriminator(latent_num=latent_num).cuda()

    def train_mode(self):
        # self.unet.train()
        self.encoder.train()
        self.decoder.train()
        self.classify.train()
        self.discriminator.train()

    def eval_mode(self):
        # self.unet.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.classify.eval()
        self.discriminator.eval()

    def forward(self,x):
        # feature,reconstruction = self.unet(x)
        feature = self.encoder.extract_features(x)
        # reconstruction = self.decoder(feature)
        # print(reconstruction.shape)
        reconstruction = None
        feature = feature.view(feature.shape[0],-1)
        debug("feature : {}".format( feature.shape))
        logits = self.classify(feature)
        return reconstruction,feature,logits

    def save_model(self,ckpt_dir,epoch):
        ckpt_name = os.path.join(ckpt_dir, 'params-{}-encoder.ckpt'.format(epoch))
        torch.save(self.encoder.state_dict(), ckpt_name)

        ckpt_name = os.path.join(ckpt_dir, 'params-{}-decoder.ckpt'.format(epoch))
        torch.save(self.decoder.state_dict(), ckpt_name)

        # ckpt_name = os.path.join(ckpt_dir, 'params-{}-unet.ckpt'.format(epoch))
        # torch.save(self.unet.state_dict(), ckpt_name)

        ckpt_name = os.path.join(ckpt_dir, 'params-{}-classify.ckpt'.format(epoch))
        torch.save(self.classify.state_dict(), ckpt_name)
