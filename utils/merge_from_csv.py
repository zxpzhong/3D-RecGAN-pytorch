import sys
import numpy as np
import os
import csv
from csv_op import read_from_csv
import torch
sys.path.append('../')
from utils import AverageMeter, batch_cos_distance, calc_eer

if __name__ == '__main__':
    dis1 = read_from_csv('/home/zf/pycharm_tmp/double_img_vein/mobilenetV1/checkpoints/NLLCENTER_MOBILEV2/test_front_norotate_dis.csv')
    dis2 = read_from_csv('/home/zf/pycharm_tmp/double_img_vein/mobilenetV1/checkpoints/NLLCENTER_MOBILEV2/test_back_norotate_dis.csv')
    label = read_from_csv('/home/zf/pycharm_tmp/double_img_vein/mobilenetV1/csv/test_back_norotate.csv')

    img_dist_list = np.array(dis1['distance'])
    dep_dist_list = np.array(dis2['distance'])
    label_list = np.array(label['flag'])

    # 寻找不同权重
    merge_weight_list = np.linspace(0, 1, num=50)
    best_eer = 1
    best_thresh = 100
    best_weight = 0
    for i, weight in enumerate(merge_weight_list):
        merge_distance = weight * img_dist_list + (1 - weight) * dep_dist_list

        eer, thresh, minV = calc_eer(torch.Tensor(merge_distance), torch.Tensor(label_list))

        print('weight: {:.4f}, eer: {:.4f}, thresh: {:.4f}, minV: {:.4f} ({} / {})'.format(weight, eer, thresh, minV,
                                                                                                      i+1, merge_weight_list.shape[0]))

        if eer < best_eer:
            best_eer = eer
            best_thresh = thresh
            best_weight = weight

    print('Final  :   best_eer:{}, best_thresh:{}, best_weight:{} '.format(best_eer, best_thresh, best_weight))
