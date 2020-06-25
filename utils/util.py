import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


# ------

# -*- coding:UTF-8 -*-
import os
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def L2_distance(feature1, feature2):
    '''
    计算两个特征向量之间的余弦距离
    :param feature1: [1, feat_dim]
    :param feature2: [1, feat_dim]
    :return: distance: 余弦距离
    '''
    return torch.norm(feature1 - feature2, 2)


def batch_L2_distance(feature1, feature2):
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
        distances.append(L2_distance(feature1[i], feature2[i]))
    return torch.Tensor(distances)

from tqdm import tqdm
# 1表示同类，0表示异类
def calc_eer(distances, label):
    '''
    计算等误率
    :param distances:  余弦距离矩阵，[batch_size]
    :param label:  标签，[batch_size]；1，表示同类；0，表示异类
    :return:
    '''
    # 将tensor转化为numpy
    distances_np = np.array(distances)
    label_np = np.array(label.cpu())

    batch_size = label.size(0)
    minV = 100
    bestThresh = 0

    max_dist = np.max(distances_np)
    min_dist = np.min(distances_np)
    threshold_list = np.linspace(min_dist, max_dist, num=100)

    intra_cnt_final = 0
    inter_cnt_final = 0
    intra_len_final = 0
    inter_len_final = 0

    for threshold in (threshold_list):
        intra_cnt = 0
        intra_len = 0
        inter_cnt = 0
        inter_len = 0
        for i in (range(batch_size)):
            # intra
            # 注意是余弦距离，越大越相近，所以这里若小于则错误了
            if label_np[i] == 1:
                intra_len += 1
                if distances_np[i] < threshold:
                    intra_cnt += 1
            elif label_np[i] == 0:
                inter_len += 1
                if distances_np[i] > threshold:
                    inter_cnt += 1

        fr = intra_cnt / intra_len
        fa = inter_cnt / inter_len

        if abs(fr - fa) < minV:
            minV = abs(fr - fa)
            eer = (fr + fa) / 2
            bestThresh = threshold

            intra_cnt_final = intra_cnt
            inter_cnt_final = inter_cnt
            intra_len_final = intra_len
            inter_len_final = inter_len
    # print('eer : {}, bestThresh : {},'.format(eer,bestThresh))
    # print("intra_cnt is : {} , inter_cnt is {} , intra_len is {} , inter_len is {}".format(intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final))

    return intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final,eer, bestThresh, minV


if __name__ == '__main__':
    f1 = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    f2 = f1 * 2
    dists = batch_cos_distance(f1, f2)
    print(dists)
