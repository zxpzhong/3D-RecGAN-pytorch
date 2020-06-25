# -*- coding: utf-8 -*-
# @Time    : 2018/11/16 20:43
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : center_git_loss.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from params import Args

class CenterGitLoss(nn.Module):
    '''
    Center Loss and Git Loss

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Calefati, A., et al. (2018). Git Loss for Deep Face Recognition.

    '''
    def __init__(self, num_classes=107, feat_dim=256, update=True):
        super(CenterGitLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.update = update

        # 中心点
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(Args.device))

        # centers的梯度
        # self.delta_centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(Args.device))

        # centers的学习率
        self.lr = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        feat_dim = x.size(1)
        # x: (batch_size, feat_dim)
        # centers: (num_classes, feat_dim)

        if self.update:
            # 更新Center
            delta_centers = torch.zeros(self.num_classes, self.feat_dim).to(Args.device)
            for j in range(self.num_classes):
                # c_j = self.centers[j]
                # A, B = 0, 0
                A = torch.zeros(feat_dim).to(Args.device)
                B = torch.zeros(1).to(Args.device)
                for i in range(batch_size):
                    if labels[i] == j:
                        A += self.centers[j] - x[i]
                        B += 1
                delta_centers[j] = A / (1 + B)
                del A, B
            # delta_centers = nn.Parameter(delta_centers).to(Args.device)

            # 更新centers
            # print(self.centers.shape)
            # print(self.delta_centers.shape)
            # print(type(self.centers))
            # print(type(self.delta_centers))
            # print(type(self.lr))
            self.centers = nn.Parameter(self.centers - self.lr * delta_centers).to(Args.device)
            # print(self.centers)

        # 转成矩阵形式操作（一开始还真没看懂。。。反应过来就很好理解了）
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        center_loss_distmat =  distmat
        git_loss_distmat = 1 / (1 + distmat)

        classes = torch.arange(self.num_classes).long()
        # classes = torch.arange(self.num_classes)
        # classes = torch.arange(self.num_classes)
        classes = classes.to(Args.device)
        # 增加一维，并转换为(batch_size, num_classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        center_loss_dist = []
        git_loss_dist = []
        for i in range(batch_size):
            value = center_loss_distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)
            center_loss_dist.append(value)

            value = git_loss_distmat[i].sum() - git_loss_distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)
            git_loss_dist.append(value)
        center_loss_dist = torch.cat(center_loss_dist)
        git_loss_dist = torch.cat(git_loss_dist)
        center_loss = center_loss_dist.mean()
        git_loss = git_loss_dist.mean()

        return center_loss, git_loss

if __name__ == '__main__':
    loss = CenterGitLoss()

    test_data = torch.randn(8, 256)
    test_label = torch.Tensor(np.random.randint(0, 107, size=(8))).long()
    center_loss, git_loss = loss(test_data, test_label)

    print("center_loss: {}, git_loss: {}".format(center_loss, git_loss))
