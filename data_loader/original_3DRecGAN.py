'''
original_3DRecGAN++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import torch
import pandas as pd
from base import BaseDataLoader
from base import DataPrefetcher
import numpy as np
import re
import matplotlib.pyplot as plt

class Data:
    def __init__(self,path):
        ###############################################################
        self.resolution = 64
        config={}
        # chair/stool/toilet
        config['train_names'] = ['chair']
        for name in config['train_names']:
            config['X_train_'+name] = path+name+'/train_25d/voxel_grids_64/'
            config['Y_train_'+name] = path+name+'/train_3d/voxel_grids_64/'
        config['test_names']=['chair']
        for name in config['test_names']:
            config['X_test_'+name] = path+name+'/test_25d/voxel_grids_64/'
            config['Y_test_'+name] = path+name+'/test_3d/voxel_grids_64/'
        self.config = config
        self.train_names = config['train_names']
        self.test_names = config['test_names']

        self.X_train_files, self.Y_train_files = self.load_X_Y_files_paths_all( self.train_names,label='train')
        self.X_test_files, self.Y_test_files = self.load_X_Y_files_paths_all(self.test_names,label='test')
        print ('X_train_files:',len(self.X_train_files))
        print ('X_test_files:',len(self.X_test_files))


    @staticmethod
    def plotFromVoxels(voxels):
        if len(voxels.shape)>3:
            x_d = voxels.shape[0]
            y_d = voxels.shape[1]
            z_d = voxels.shape[2]
            v = voxels[:,:,:,0]
            v = np.reshape(v,(x_d,y_d,z_d))
        else:
            v = voxels
        x, y, z = v.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        plt.show()

    def load_X_Y_files_paths_all(self, obj_names, label='train'):
        x_str=''
        y_str=''
        if label =='train':
            x_str='X_train_'
            y_str ='Y_train_'

        elif label == 'test':
            x_str = 'X_test_'
            y_str = 'Y_test_'

        else:
            print ('label error!!')
            exit()

        X_data_files_all = []
        Y_data_files_all = []
        for name in obj_names:
            X_folder = self.config[x_str + name]
            Y_folder = self.config[y_str + name]
            X_data_files, Y_data_files = self.load_X_Y_files_paths(X_folder, Y_folder)

            for X_f, Y_f in zip(X_data_files, Y_data_files):
                if X_f[0:15] != Y_f[0:15]:
                    print ('index inconsistent!!')
                    exit()
                X_data_files_all.append(X_folder + X_f)
                Y_data_files_all.append(Y_folder + Y_f)
        return X_data_files_all, Y_data_files_all

    def load_X_Y_files_paths(self,X_folder, Y_folder):
        X_data_files = [X_f for X_f in sorted(os.listdir(X_folder))]
        Y_data_files = [Y_f for Y_f in sorted(os.listdir(Y_folder))]
        return X_data_files, Y_data_files

    def voxel_grid_padding(self,a):
        x_d = a.shape[0]
        y_d = a.shape[1]
        z_d = a.shape[2]
        channel = a.shape[3]
        resolution = self.resolution
        size = [resolution, resolution, resolution,channel]
        b = np.zeros(size)

        bx_s = 0;bx_e = size[0];by_s = 0;by_e = size[1];bz_s = 0; bz_e = size[2]
        ax_s = 0;ax_e = x_d;ay_s = 0;ay_e = y_d;az_s = 0;az_e = z_d
        if x_d > size[0]:
            ax_s = int((x_d - size[0]) / 2)
            ax_e = int((x_d - size[0]) / 2) + size[0]
        else:
            bx_s = int((size[0] - x_d) / 2)
            bx_e = int((size[0] - x_d) / 2) + x_d

        if y_d > size[1]:
            ay_s = int((y_d - size[1]) / 2)
            ay_e = int((y_d - size[1]) / 2) + size[1]
        else:
            by_s = int((size[1] - y_d) / 2)
            by_e = int((size[1] - y_d) / 2) + y_d

        if z_d > size[2]:
            az_s = int((z_d - size[2]) / 2)
            az_e = int((z_d - size[2]) / 2) + size[2]
        else:
            bz_s = int((size[2] - z_d) / 2)
            bz_e = int((size[2] - z_d) / 2) + z_d
        b[bx_s:bx_e, by_s:by_e, bz_s:bz_e,:] = a[ax_s:ax_e, ay_s:ay_e, az_s:az_e, :]

        return b

    def load_single_voxel_grid(self,path):
        temp = re.split('_', path.split('.')[-2])
        x_d = int(temp[len(temp) - 3])
        y_d = int(temp[len(temp) - 2])
        z_d = int(temp[len(temp) - 1])

        a = np.loadtxt(path)
        if len(a)<=0:
            print ('load_single_voxel_grid error:', path)
            exit()

        voxel_grid = np.zeros((x_d, y_d, z_d,1))
        for i in a:
            voxel_grid[int(i[0]), int(i[1]), int(i[2]),0] = 1 # occupied

        #Data.plotFromVoxels(voxel_grid)
        voxel_grid = self.voxel_grid_padding(voxel_grid)
        voxel_grid = voxel_grid.transpose([3,0,1,2])
        return voxel_grid

    def load_X_Y_voxel_grids(self,X_data_files, Y_data_files):
        if len(X_data_files) !=self.batch_size or len(Y_data_files)!=self.batch_size:
            print ('load_X_Y_voxel_grids error:', X_data_files, Y_data_files)
            exit()

        X_voxel_grids = []
        Y_voxel_grids = []
        index = -1
        for X_f, Y_f in zip(X_data_files, Y_data_files):
            index += 1
            X_voxel_grid = self.load_single_voxel_grid(X_f)
            X_voxel_grids.append(X_voxel_grid)

            Y_voxel_grid = self.load_single_voxel_grid(Y_f)
            Y_voxel_grids.append(Y_voxel_grid)

        X_voxel_grids = np.asarray(X_voxel_grids)
        Y_voxel_grids = np.asarray(Y_voxel_grids)
        return X_voxel_grids, Y_voxel_grids


class train_original_3DRecGAN(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True):
        self.dataset = Train_Dataset(data_dir)
        self.batch_size = batch_size
        super().__init__(self.dataset, batch_size, shuffle, num_workers)
        
class test_original_3DRecGAN(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True):
        self.dataset = Test_Dataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, num_workers)


class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self,path):
        self.root = path
        # fetch training data
        self.data = Data(self.root)
        
    def __len__(self):
        return len(self.data.X_train_files)

    def __getitem__(self, index):
        # return training data
        X = self.data.load_single_voxel_grid(self.data.X_train_files[index])
        Y = self.data.load_single_voxel_grid(self.data.Y_train_files[index])
        
        return X,Y

class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self,path):
        self.root = path
        # fetch test data
        self.data = Data(self.root)

    def __len__(self):
        return len(self.data.X_test_files)

    def __getitem__(self, index):
        X = self.data.load_single_voxel_grid(self.data.X_test_files[index])
        Y = self.data.load_single_voxel_grid(self.data.Y_test_files[index])
        return X,Y