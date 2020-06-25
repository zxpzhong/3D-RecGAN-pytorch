import sys
sys.path.append('../')
from scipy.special import comb, perm
import numpy as np
import os
import torch
import csv
from tqdm import tqdm

class Args:
    # # 数据库路径
    # dir = '/home/data_ssd/ywl_DataSets/seg_zf/data/MMCBNU_6000_FVDataSet_ROI_Seg_maxcurv_pcurv_gabor_thin'
    # train_dir = os.path.join(dir)
    # test_dir = '/home/data_ssd/ywl_DataSets/seg_zf/data/MMCBNU_6000_FVDataSet_ROI_Seg_maxcurv_pcurv_gabor_thin'
    dir = '/home/data_ssd/ywl_DataSets/seg_zf/data/MMCBNU_6000_FVDataSet_ROI'
    train_dir = os.path.join(dir)
    test_dir = '/home/data_ssd/ywl_DataSets/seg_zf/data/MMCBNU_6000_FVDataSet_ROI'
    root_dir = os.getcwd()
    # 总类别数：人数*手指数
    subjects = 600
    # 采集次数
    times = 10

    # 训练测试比例
    test_rate = 0.9
    # 参与训练手指数目
    train_subject = int(subjects * test_rate)+1
    # train_subject = 0
    test_subject = subjects - train_subject

def creat_train_file_MMCBNU():
    '''
    韩国全北大学数据库，命名方式为 person(1-100)-left/right(1/2)-finger(1-3)-num(1-10).bmp
    如果是分类任务，那么训练集上的label=(person-1)*6+(left/right -1 )*3+(finger-1)
    写入csv格式：	number  flag	img_path	label
    number:0,1,2.....
    flag:train
    img_path:绝对路径
    label:类别
    :return:
    '''

    with open(os.path.join(Args.root_dir,'csv','train_MMCBNU_91.csv'), "w", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerow(["number", "flag", "img_path", "label"])
        number = 0
        for person in range(1,101):
            for lr in range(1,3):
                for finger in range(1,4):
                    label = (person - 1) * 6 + (lr - 1) * 3 + (finger - 1)
                    if not label > Args.train_subject-1:
                        for num in range(1, 11):
                            flag = 'train'
                            img_path = os.path.join(Args.train_dir,'{}_{}_{}_{}.bmp'.format(person, lr, finger,num))
                            csvwriter.writerow([str(number), flag, img_path, label])
                            number = number + 1

def creat_test_set_MMCBNU():
    '''
    写入csv格式：number  	img_path	label
    number：序号 0，1，2，3。。。
    label：是否为同一类0为不同类，1为同一类
    img1_path：样本一的地址
    img2_path: 样本二的地址
    :return:
    '''
    count = 0
    with open(os.path.join(Args.root_dir,'csv','test_MMCBNU_91.csv'), "w", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        csvwriter.writerow(["number", "flag", "img1_path", "img2_path"])
        for finger_num in tqdm(range(0, Args.subjects)):
            if finger_num > Args.train_subject-1:
                # for finger_num in tqdm(range(1, 3)):
                for capture_time in range(1, 10 + 1):
                    #遍历类内所有样本
                    # 寻找类内样本
                    current_name = '{}_{}_{}_{}.bmp'.format(finger_num//6+1,finger_num%6//3+1,finger_num%6%3+1,capture_time)
                    for other_capturte_time in range(capture_time, 10 + 1):
                        if (other_capturte_time != capture_time):
                            intra_name = '{}_{}_{}_{}.bmp'.format(finger_num // 6 + 1, finger_num % 6 // 3 + 1,
                                                                    finger_num % 6 % 3+1, other_capturte_time)
                            # 寻找类间样本，随便找一个样本'
                            random_finger_num = np.random.randint(Args.train_subject+1,Args.subjects)
                            while random_finger_num == finger_num:
                                random_finger_num = np.random.randint(Args.train_subject + 1, Args.subjects)
                            random_capturte_time = np.random.randint(1, 10 + 1)
                            inter_name = '{}_{}_{}_{}.bmp'.format(random_finger_num // 6 + 1, random_finger_num % 6 // 3 + 1,
                                                                  random_finger_num % 6 % 3+1, random_capturte_time)
                            # 将类内匹配对写入csv
                            csvwriter.writerow([str(count), '1', os.path.join(Args.test_dir,current_name), os.path.join(Args.test_dir,intra_name)])
                            count = count+1
                            # 将类间匹配对写入csv
                            csvwriter.writerow([str(count), '0', os.path.join(Args.test_dir,current_name), os.path.join(Args.test_dir,inter_name)])
                            count = count + 1




def creat_train_file_MMCBNU_saimese_train():
    '''
    写入csv格式：number  	img_path	label
    number：序号 0，1，2，3。。。
    label：是否为同一类0为不同类，1为同一类
    img1_path：样本一的地址
    img2_path: 样本二的地址
    :return:
    '''
    count = 0
    with open(os.path.join(Args.root_dir,'csv','train_MMCBNU_vein_feature_siamese.csv'), "w", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        csvwriter.writerow(["number", "flag", "img1_path", "img2_path"])
        for finger_num in tqdm(range(0, Args.subjects)):
            if not finger_num > Args.train_subject-1:
                # for finger_num in tqdm(range(1, 3)):
                for capture_time in range(1, 10 + 1):
                    #遍历类内所有样本
                    # 寻找类内样本
                    current_name = '{}_{}_{}_{}.bmp.npy'.format(finger_num//6+1,finger_num%6//3+1,finger_num%6%3+1,capture_time)
                    for other_capturte_time in range(capture_time, 10 + 1):
                        if (other_capturte_time != capture_time):
                            intra_name = '{}_{}_{}_{}.bmp.npy'.format(finger_num // 6 + 1, finger_num % 6 // 3 + 1,
                                                                    finger_num % 6 % 3+1, other_capturte_time)
                            # 寻找类间样本，随便找一个样本'
                            random_finger_num = np.random.randint(0,Args.train_subject)
                            while random_finger_num == finger_num:
                                random_finger_num = np.random.randint(0,Args.train_subject)
                            random_capturte_time = np.random.randint(1, 10 + 1)
                            inter_name = '{}_{}_{}_{}.bmp.npy'.format(random_finger_num // 6 + 1, random_finger_num % 6 // 3 + 1,
                                                                  random_finger_num % 6 % 3+1, random_capturte_time)
                            # 将类内匹配对写入csv
                            csvwriter.writerow([str(count), '1', os.path.join(Args.test_dir,current_name), os.path.join(Args.test_dir,intra_name)])
                            count = count+1
                            # 将类间匹配对写入csv
                            csvwriter.writerow([str(count), '0', os.path.join(Args.test_dir,current_name), os.path.join(Args.test_dir,inter_name)])
                            count = count + 1



def creat_train_file_MMCBNU_saimese_test():
    '''
    写入csv格式：number  	img_path	label
    number：序号 0，1，2，3。。。
    label：是否为同一类0为不同类，1为同一类
    img1_path：样本一的地址
    img2_path: 样本二的地址
    :return:
    '''
    count = 0
    with open(os.path.join(Args.root_dir,'csv','test_MMCBNU_vein_feature_siamese.csv'), "w", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        csvwriter.writerow(["number", "flag", "img1_path", "img2_path"])
        for finger_num in tqdm(range(0, Args.subjects)):
            if finger_num > Args.train_subject-1:
                # for finger_num in tqdm(range(1, 3)):
                for capture_time in range(1, 10 + 1):
                    #遍历类内所有样本
                    # 寻找类内样本
                    current_name = '{}_{}_{}_{}.bmp.npy'.format(finger_num//6+1,finger_num%6//3+1,finger_num%6%3+1,capture_time)
                    for other_capturte_time in range(capture_time, 10 + 1):
                        if (other_capturte_time != capture_time):
                            intra_name = '{}_{}_{}_{}.bmp.npy'.format(finger_num // 6 + 1, finger_num % 6 // 3 + 1,
                                                                    finger_num % 6 % 3+1, other_capturte_time)
                            # 寻找类间样本，随便找一个样本'
                            random_finger_num = np.random.randint(Args.train_subject,Args.subjects)
                            while random_finger_num == finger_num:
                                random_finger_num = np.random.randint(Args.train_subject, Args.subjects)
                            random_capturte_time = np.random.randint(1, 10 + 1)
                            inter_name = '{}_{}_{}_{}.bmp.npy'.format(random_finger_num // 6 + 1, random_finger_num % 6 // 3 + 1,
                                                                  random_finger_num % 6 % 3+1, random_capturte_time)
                            # 将类内匹配对写入csv
                            csvwriter.writerow([str(count), '1', os.path.join(Args.test_dir,current_name), os.path.join(Args.test_dir,intra_name)])
                            count = count+1
                            # 将类间匹配对写入csv
                            csvwriter.writerow([str(count), '0', os.path.join(Args.test_dir,current_name), os.path.join(Args.test_dir,inter_name)])
                            count = count + 1



import shutil
def data_rename():
    pass
    src_dir = '/home/data_ssd/ywl_DataSets/seg_zf/data/MMCBNU_6000_FVDataSet_ROI'
    dest_dir = '/home/data_ssd/zf/vein/MMCBNU_6000_FVDataSet_ROI'

    for finger_num in tqdm(range(0, Args.subjects)):
        # if finger_num > Args.train_subject - 1:
        if True:
            # for finger_num in tqdm(range(1, 3)):
            for capture_time in range(1, 10 + 1):
                # 遍历类内所有样本
                # 寻找类内样本
                current_name = '{}_{}_{}_{}.bmp'.format(finger_num // 6 + 1, finger_num % 6 // 3 + 1,
                                                        finger_num % 6 % 3 + 1, capture_time)
                new_name = '{}-{}.bmp'.format(finger_num,capture_time)
                print(os.path.join(src_dir,current_name))
                print(os.path.join(src_dir,new_name))
                shutil.copy(os.path.join(src_dir,current_name),os.path.join(dest_dir,new_name))

        # if finger_num > 2:
        #     exit()


if __name__ == '__main__':
    # creat_train_set()
    creat_train_file_MMCBNU()
    creat_test_set_MMCBNU()
    # creat_test_set(AB='A')
    # creat_train_file_MMCBNU_saimese_train()
    # creat_train_file_MMCBNU_saimese_test()
    # data_rename()
    pass