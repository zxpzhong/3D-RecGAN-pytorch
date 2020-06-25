import torch
import numpy
import torch.nn as nn
import torch.nn.functional as func

def cos_distance(feature1, feature2):
    '''
    计算两个特征向量之间的余弦距离
    :param feature1: [1, feat_dim]
    :param feature2: [1, feat_dim]
    :return: distance: 余弦距离
    '''
    # reshape成[1, feat_dim]
    feature1 = feature1.reshape(-1, feature1.size(-1))
    feature2 = feature2.reshape(-1, feature2.size(-1))
    return torch.sum(feature1 * feature2, -1) / (torch.norm(feature1) * torch.norm(feature2))

def batch_cos_distance(feature1, feature2):
    '''
    计算两组特征向量之间的余弦距离
    :param feature1:  [batch_size, feat_dim]
    :param feature2:  [batch_size, feat_dim]
    :return: distances: [batch_size]
    '''
    batch_size = feature1.size(0)
    # feat_dim = feature1.size(1)
    distances = []
    for i in range(batch_size):
        distances.append(cos_distance(feature1[i], feature2[i]))
    return torch.Tensor(distances)

class TripletCosinLoss(nn.Module):
    def __init__(self, t1, t2, beta):
        super(TripletCosinLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.beta = beta
        return

    def forward(self, anchor, positive, negative):
        matched = 1 - batch_cos_distance(anchor, positive)
        mismatched1 = 1 - batch_cos_distance(anchor, negative)
        # mismatched2 = batch_cos_distance(positive, negative)

        # part_1 = torch.clamp(matched - mismatched1, min=self.t1)
        # part_2 = torch.clamp(matched - mismatched2, min=self.t1)
        # part_3 = torch.clamp(matched, min=self.t2)
        # dist_hinge = part_1 + part_2 +self.beta * part_3
        # dist_hinge = matched1 + mismatched1 + mismatched2

        dist_hinge = 0.5*torch.clamp(0.5+matched-mismatched1,min=0)

        loss = torch.mean(dist_hinge)
        return loss