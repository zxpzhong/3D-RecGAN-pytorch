import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        # 初始化
        nn.init.xavier_normal_(self.centers)
        # nn.init.constant_(self.centers, 0)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # torch.pow(x, 2).sum(dim=1, keepdim=True)将特征的每一个元素平方，然后按照第1维进行加和（也就是单个特征自身），并且keepdim保持维度，即加和后仍然为列向量
        # .expand(batch_size, self.num_classes)再将上述的平方和列向量保持行数不变，列数拉成类别数那么多
        # 然后对于类中心也执行和特征一样的操作，之后再加和
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        # print(mask.shape)
        # print(distmat.shape)
        # 选出 classes正确的 ，取出其距离值
        dist = distmat * mask.float()
        # print(dist.shape)
        # exit()
        # 类内求平均距离值
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size


        # 改进： 类间样本也设置一个阈值，然后让类间距离变大
        # 1.0 取出所有classes不正确的
        # dist = distmat * (~mask).float()
        # # 2.0 取出不正确且距离过小的
        # dist = dist * (dist < 10).float()
        # loss = loss - dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # print(distmat)
        # print(distmat[~mask])
        #
        # # print(dist.min())
        # 这个符号貌似有问题吧？？？？
        # loss = loss - distmat[~mask].min()

        # 改进：直接从类中心考虑，计算类中心的最小距离


        return loss
