import os
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker,calc_eer
from utils.numpy3D import numpy_2_ply
import torch.nn.functional as F
from tqdm import tqdm
from utils.metric import IOU_metric,cross_entropy


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer_G,optimizer_D, config, data_loader,
                 valid_data_loader=None, lr_scheduler_G=None,lr_scheduler_D=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer_G,optimizer_D, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader.dataset)
            # self.len_epoch = len(self.data_loader.loader_)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.log_step = int(data_loader.batch_size)*10

        self.train_metrics = MetricTracker('loss_G','loss_D', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('IOU','CE', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
        # AE reconstruction loss
        # self.AE_loss = F.cross_entropy
        
        # optimizer
        self.Generator_opt = optimizer_G
        self.Discriminator_opt = optimizer_D
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        loss_D = torch.Tensor([0])
        loss_G = torch.Tensor([0])
        for batch_idx, (X, Y) in enumerate(tqdm(self.data_loader)):
            X, Y = X.to(self.device).float(), Y.to(self.device).float()
            
            real_labels = torch.ones(X.shape[0]).to(self.device)
            fake_labels = torch.zeros(X.shape[0]).to(self.device)
            
            self.optimizer_D.zero_grad()
            # train D
            Y_fake,dis_fake,dis_real = self.model(X,Y,train_D=True)
            # print('##########')
            # print(dis_real.reshape(dis_real.shape[0],-1))
            # print(dis_fake.reshape(dis_real.shape[0],-1))
            d_real_loss = F.binary_cross_entropy(dis_real, real_labels)
            d_fake_loss = F.binary_cross_entropy(dis_fake, fake_labels)
            loss_D = d_real_loss + d_fake_loss
            loss_D.backward()
            self.Discriminator_opt.step()
            
            self.optimizer_G.zero_grad()
            # train AE(G)
            Y_fake,dis_fake = self.model(X)
            # [bs, 262144]
            Y_fake_ = Y_fake.reshape(Y_fake.shape[0],-1)
            Y_ = Y.reshape(Y.shape[0],-1)
            ae_loss = cross_entropy(Y_fake_,Y_)/Y_fake_.shape[1]
            g_fake_loss = F.binary_cross_entropy(dis_fake, real_labels)
            # ae_loss = torch.mean(torch.abs(Y_fake-Y))
            # ae_loss = F.mse_loss(Y_fake,Y)
            # discriminator score
            loss_G = ae_loss+g_fake_loss
            loss_G.backward()
            self.Generator_opt.step()
            
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss_G', loss_G.item())
                self.train_metrics.update('loss_D', loss_D.item())
                self.logger.debug('Train Epoch: {} {} loss_G: {:.6f} loss_D: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_G.item(),
                    loss_D.item()
                    ))
            if batch_idx == self.len_epoch:
                break
            batch_idx+=1
            
        log = self.train_metrics.result()

        # save model 
        if epoch % self.save_period == 0:
            self._save_checkpoint(epoch, save_best=True)
        
        if epoch%self.config['trainer']['save_period'] == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            X_list = []
            Y_fake_list = []
            Y_list = []
            length = 20
            IOU = 0
            CE = 0
            count = 0
            for batch_idx, (X,Y) in enumerate(tqdm(self.valid_data_loader)):
                X,Y = X.to(self.device).float(),Y.to(self.device).float()
                Y_fake = self.model.module.unet(X)
                if batch_idx < length:
                    for i in range(Y_fake.shape[0]):
                        Y_fake_list.append(Y_fake[i].cpu().numpy())
                        X_list.append(X[i].cpu().numpy())
                        Y_list.append(Y[i].cpu().numpy())
                
                # cal test metric
                for i in range(Y_fake.shape[0]):
                    IOU+=IOU_metric(Y_fake[i],Y[i])
                    count+=1
                # output voxel dimension : 64
                CE+=cross_entropy(Y_fake,Y)
                    
            # save test set reconstruction 3D voxel
            Y_fake_array = np.array(Y_fake_list).transpose([0,2,3,4,1])
            X_array = np.array(X_list).transpose([0,2,3,4,1])
            Y_array = np.array(Y_list).transpose([0,2,3,4,1])
            print('saving ply......')
            for i in tqdm(range(Y_fake_array.shape[0])):
                numpy_2_ply(Y_fake_array[i],os.path.join(self.config.save_dir,'Y_fake_epoch_{}_{}.ply'.format(epoch,i)),threshold=0.5)
                if i == 0:
                    numpy_2_ply(X_array[i],os.path.join(self.config.save_dir,'X_{}.ply'.format(epoch,i)),threshold=0.5)
                    numpy_2_ply(Y_array[i],os.path.join(self.config.save_dir,'Y_{}.ply'.format(epoch,i)),threshold=0.5)

            # log 
            self.logger.debug("Test IOU is : {} , CE is {} ".format(IOU/count,CE/count))
            self.writer.set_step((epoch - 1), 'valid')
            self.valid_metrics.update('IOU', IOU)
            self.valid_metrics.update('CE', CE)
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
