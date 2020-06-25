# -*- coding:UTF-8 -*-
import os
import torch
import logging  # 引入logging模块
logging.basicConfig(level=logging.NOTSET)  # 设置日志级别

class Args:
    # 数据库路径
    dir = '/home/data_ssd/ywl_DataSets/seg_zf/data/MMCBNU_6000_FVDataSet_ROI_Seg_maxcurv_pcurv_gabor_thin'
    train_dir = os.path.join(dir)
    test_dir = '/home/data_ssd/ywl_DataSets/seg_zf/data/MMCBNU_6000_FVDataSet_ROI_Seg_maxcurv_pcurv_gabor_thin'
    root_dir = os.getcwd()
    # 总类别数：人数*手指数
    subjects = 600
    # 采集次数
    times = 10

    # 训练测试比例
    test_rate = 0.8
    # 参与训练手指数目
    train_subject = int(subjects * test_rate)+1

    test_subject = subjects - train_subject
    # 数据扩增次数
    aug_num = 24

    # 特征维度
    nb_feature = 256

    # 超参数
    keep_prob = 0.5
    batch_size = 16
    num_epoches = 4000
    learning_rate = 0.0002
    learning_rate_cent = 0.00001
    weight_cent = 0.025


    print_freq = 100
    test_freq = 4
    save_freq = 4
print(Args.train_subject)