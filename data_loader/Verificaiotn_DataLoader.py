'''
Verificaiotn_DataLoader++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import torch
import pandas as pd
from torchvision import transforms as T
from PIL import Image


from torchvision import datasets, transforms
from base import BaseDataLoader


class Verificaiotn_DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir,test_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, verification=True):
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.dataset = Train_Dataset(self.data_dir)
        if not self.test_dir == None:
            self.test_dataset = Test_Dataset(self.test_dir)
        else:
            self.test_dataset = None
        super().__init__(self.dataset,self.test_dataset, batch_size, shuffle, validation_split, num_workers,verification)
# 路径替换
final_data_root = '/home/data/finger_vein'
def path_change(str):
    new_str = str.replace('/home/data_ssd/ywl_DataSets/seg_zf/data',final_data_root)
    return new_str

transform = T.Compose([
    T.Resize([256, 256]),
    T.RandomCrop(224),
    T.Grayscale(),
    T.RandomRotation(10),
    T.RandomAffine(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    T.RandomGrayscale(),
    T.RandomPerspective(0.2,0.2),
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize([0.5], [0.5]),  # 标准化至[-1, 1]，规定均值和标准差
])
'''
PIL读取出来的图像默认就已经是0-1范围了！！！！！！！！，不用再归一化
'''
transform_notrans = T.Compose([
    T.Grayscale(),
    T.Resize([224,224]), # 缩放图片(Image)
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
    T.Normalize([0.5], [0.5]),  # 标准化至[-1, 1]，规定均值和标准差
])

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self,csv_file):
        csv_path = os.path.join('data/csv/', csv_file)
        self.imgwidth, self.imgheight = 224,224
        df = pd.read_csv(csv_path)
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # print(path_change(self.df['img_path'][index]))
        if not os.path.isfile(path_change(self.df.iloc[index]['img_path'])):
            print(path_change(self.df['img_path'][index]))
            print('file not exist')
            exit()
        # print(self.df['img_path'][index])
        pil_img = Image.open(path_change(self.df.iloc[index]['img_path'])).convert("L")
        pil_img = transform(pil_img)
        pil_img = torch.cat([pil_img, pil_img, pil_img], 0)
        label = int(self.df.iloc[index]['label'])
        return pil_img,label


class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self,csv_file):
        csv_path = os.path.join('data/csv',csv_file)
        self.df = pd.read_csv(csv_path)
        # self.imgwidth, self.imgheight = imgwidth, imgheight

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if not os.path.isfile(path_change(self.df['img1_path'][index])):
            print(path_change(self.df['img1_path'][index]))
            print('file not exist')
            exit()
        if not os.path.isfile(path_change(self.df['img2_path'][index])):
            print(path_change(self.df['img1_path'][index]))
            print('file not exist')
            exit()
        img1 = Image.open(path_change(self.df['img1_path'][index])).convert("L")
        img2 = Image.open(path_change(self.df['img2_path'][index])).convert("L")
        img1 = transform_notrans(img1)
        img1 = torch.cat([img1, img1, img1], 0)
        img2 = transform_notrans(img2)
        img2 = torch.cat([img2, img2, img2], 0)
        label = int(self.df['flag'][index])
        return img1,img2,label