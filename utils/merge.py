# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 12:14
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : merge.py
# @Software: PyCharm

import os
import sys
sys.path.append('/home/zf/pycharm_tmp/3D_VEIN/fv_img_dep_dl-dev/fv_img_dep_dl/')
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging  # 引入logging模块
# logging.basicConfig(level=logging.NOTSET)  # 设置日志级别
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from params import Args
from Model.mobilenet import MobileNet_v1
from Model.mobilenet2 import MobileNet_v2
from Model.mobilefacenet import MobileFaceNet
from datasets import normalization
from utils import cos_distance, calc_eer


def merge():


    # 导入数据集路径
    dataset_dir = Args.dir
    fv_img_dir = os.path.join(dataset_dir, 'fv_img')
    fv_dep_dir = os.path.join(dataset_dir, 'fv_dep')

    # 测试集csv文件
    pwd = os.getcwd()
    csv_dir = os.path.join(pwd, 'csv')
    csv_path = os.path.join(csv_dir, 'test_file.csv')

    # 模型参数文件所在目录
    img_model_dir = os.path.join(pwd, 'img_model_params')
    img_model_params_path = os.path.join(img_model_dir, 'params.ckpt')
    dep_model_dir = os.path.join(pwd, 'dep_model_params')
    dep_model_params_path = os.path.join(dep_model_dir, 'params.ckpt')

    # 导入测试集
    test_df = pd.read_csv(csv_path)
    logging.info('test_df: {}'.format(test_df.shape))
    sample1_list = test_df['sample1']
    sample2_list = test_df['sample2']
    label_list = test_df['label']

    # 创建网络模型
    logging.info('Creating model...')
    img_model = MobileNet_v1(num_classes=75).to(Args.device)
    dep_model = MobileNet_v1(num_classes=75).to(Args.device)
    # model = MobileNet_v2(num_classes=int(train_dataset.num_classes)).to(Args.device)
    # model = MobileFaceNet(num_classes=int(train_dataset.num_classes)).to(Args.device)
    img_model.eval()
    dep_model.eval()
    logging.info('Done!')

    # 模型导入参数
    if torch.cuda.is_available():
        img_model.load_state_dict(torch.load(img_model_params_path))
        dep_model.load_state_dict(torch.load(dep_model_params_path))
    else:
        img_model.load_state_dict(torch.load(img_model_params_path, map_location='cpu'))
        dep_model.load_state_dict(torch.load(dep_model_params_path, map_location='cpu'))

    # 测试，随便读取一组sample看看
    # test_sample1 = sample1_list[0]
    # test_sample1_path = os.path.join(fv_img_dir, test_sample1)
    # test_sample2 = sample2_list[0]
    # test_sample2_path = os.path.join(fv_img_dir, test_sample2)
    # test_label = label_list[0]
    # sample1_img = torch.Tensor(normalization(cv2.imread(test_sample1_path))).unsqueeze(0)
    # sample2_img = torch.Tensor(normalization(cv2.imread(test_sample2_path))).unsqueeze(0)
    # sample1_img = sample1_img.permute(0, 3, 1, 2)
    # sample2_img = sample2_img.permute(0, 3, 1, 2)
    #
    # _, feature1 = img_model(sample1_img)
    # _, feature2 = dep_model(sample2_img)
    #
    # logging.info(feature1.shape)
    # logging.info(feature2.shape)
    #
    # print(feature1)
    # print(feature2)

    feature_dict = {'img_cos_distance': [],
                    'dep_cos_distance': [],
                    'label': []}

    # 遍历整个测试集，分别生成纹理图和深度图的余弦距离
    for i in range(label_list.shape[0]):
        sample1 = sample1_list[i]
        sample2 = sample2_list[i]
        # label为1表示同类，label为0表示异类
        label = label_list[i]

        logging.info('{} -- {} ({} / {})'.format(sample1, sample2, i+1, label_list.shape[0]))

        # 纹理图路径
        img_sample1_path = os.path.join(fv_img_dir, sample1)
        img_sample2_path = os.path.join(fv_img_dir, sample2)

        # 深度图路径
        dep_sample1_path = os.path.join(fv_dep_dir, sample1)
        dep_sample2_path = os.path.join(fv_dep_dir, sample2)

        img_sample1 = torch.Tensor(normalization(cv2.imread(img_sample1_path))).unsqueeze(0).permute(0, 3, 1, 2)
        img_sample2 = torch.Tensor(normalization(cv2.imread(img_sample2_path))).unsqueeze(0).permute(0, 3, 1, 2)

        dep_sample1 = torch.Tensor(normalization(cv2.imread(dep_sample1_path))).unsqueeze(0).permute(0, 3, 1, 2)
        dep_sample2 = torch.Tensor(normalization(cv2.imread(dep_sample2_path))).unsqueeze(0).permute(0, 3, 1, 2)

        _, img_feature1 = img_model(img_sample1)
        _, img_feature2 = img_model(img_sample2)
        _, dep_feature1 = dep_model(dep_sample1)
        _, dep_feature2 = dep_model(dep_sample2)

        img_cos_distance = cos_distance(img_feature1, img_feature2)[0]
        dep_cos_distance = cos_distance(dep_feature1, dep_feature2)[0]

        logging.info('img_cos_distance: {:.4f}'.format(img_cos_distance.detach().numpy()))
        logging.info('dep_cos_distance: {:.4f}'.format(dep_cos_distance.detach().numpy()))

        feature_dict['img_cos_distance'].append(img_cos_distance.detach().numpy())
        feature_dict['dep_cos_distance'].append(dep_cos_distance.detach().numpy())
        feature_dict['label'].append(label)

    # 将距离保存到csv文件中
    dist_csv_path = os.path.join(csv_dir, 'cos_dist.csv')
    dist_df = pd.DataFrame(feature_dict)
    dist_df.to_csv(dist_csv_path)


def search_best_weight():
    pwd = os.getcwd()
    csv_dir = os.path.join(pwd, 'csv')
    dist_csv_path = os.path.join(csv_dir, 'cos_dist.csv')
    feature_dict = pd.read_csv(dist_csv_path)

    img_dist_list = np.array(feature_dict['img_cos_distance'])
    dep_dist_list = np.array(feature_dict['dep_cos_distance'])
    ellipse_distance_list = np.array(feature_dict['ellipse_distance'])
    label_list = np.array(feature_dict['label'])

    # 寻找不同权重
    merge_weight_list = np.linspace(0, 1, num=50)
    merge_weight_list2 = np.linspace(0, 1, num=50)
    best_eer = 1
    best_thresh = 100
    best_weight = 0
    best_weight2 = 0
    for i, weight in enumerate(merge_weight_list):
        for j, weight2 in enumerate(merge_weight_list2):
            merge_distance = weight * img_dist_list + (1 - weight) * dep_dist_list + (1-weight2) * ellipse_distance_list

            eer, thresh, minV = calc_eer(torch.Tensor(merge_distance), torch.Tensor(label_list))

            logging.info('weight: {:.4f}, eer: {:.4f}, thresh: {:.4f}, minV: {:.4f} ({} / {})'.format(weight, eer, thresh, minV,
                                                                                                      i+1, merge_weight_list.shape[0]))

            if eer < best_eer:
                best_eer = eer
                best_thresh = thresh
                best_weight = weight
                best_weight2 = weight2

    return best_eer, best_thresh, best_weight,best_weight2


def search_best_weight_ori():
    pwd = os.getcwd()
    csv_dir = os.path.join(pwd, 'csv')
    dist_csv_path = os.path.join(csv_dir, 'cos_dist.csv')
    feature_dict = pd.read_csv(dist_csv_path)

    img_dist_list = np.array(feature_dict['img_cos_distance'])
    dep_dist_list = np.array(feature_dict['ellipse_distance'])
    label_list = np.array(feature_dict['label'])

    # 寻找不同权重
    merge_weight_list = np.linspace(0, 1, num=50)
    best_eer = 1
    best_thresh = 100
    best_weight = 0
    for i, weight in enumerate(merge_weight_list):
        merge_distance = weight * img_dist_list + (1 - weight) * dep_dist_list

        eer, thresh, minV = calc_eer(torch.Tensor(merge_distance), torch.Tensor(label_list))

        logging.info('weight: {:.4f}, eer: {:.4f}, thresh: {:.4f}, minV: {:.4f} ({} / {})'.format(weight, eer, thresh, minV,
                                                                                                      i+1, merge_weight_list.shape[0]))

        if eer < best_eer:
            best_eer = eer
            best_thresh = thresh
            best_weight = weight

    return best_eer, best_thresh, best_weight,0


if __name__ == '__main__':

    # merge()

    best_eer, best_thresh, weight,weight2 = search_best_weight_ori()
    logging.info('\nbest_eer: {:.4f}, best_thresh: {:.4f}, weight: {:.4f}，weight2: {:.4f}\n'.format(best_eer, best_thresh, weight,weight2))

