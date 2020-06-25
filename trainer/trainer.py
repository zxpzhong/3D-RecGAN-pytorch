import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker,calc_eer
import torch.nn.functional as F
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None,veri_mode = False, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.veri_mode = veri_mode
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader.loader_)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        batch = self.data_loader.next()
        # for batch_idx, (data, target) in enumerate(self.data_loader):
        batch_idx = 0
        while batch is not None:
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            if batch_idx == self.len_epoch:
                break
            batch_idx+=1
            batch = self.data_loader.next()
            
            
        log = self.train_metrics.result()

        if self.do_validation and epoch%self.config['trainer']['save_period'] == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
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
            if self.veri_mode == False:
                for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)
                    loss = self.criterion(output, target)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output, target))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            else:
                distances = []
                distance_data_list = []
                labels = []
                for batch_idx, (data1,data2, target) in enumerate(self.valid_data_loader):
                    data1,data2, target = data1.to(self.device),data2.to(self.device), target.to(self.device)

                    output1 = self.model.extract_feature(data1)
                    output2 = self.model.extract_feature(data2)
                    dis = F.cosine_similarity(output1, output2).cpu()
                    distances.append(dis)
                    distance_data_list.append(np.array(dis))
                    labels.append(target)
                    
                # cat all distances
                distances = torch.cat(distances)
                # cat all labels
                label = torch.cat(labels)
                # cal eer
                intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final,eer, bestThresh, minV = calc_eer(distances, label)
                self.logger.debug('eer : {}, bestThresh : {},'.format(eer,bestThresh))
                self.logger.debug("intra_cnt is : {} , inter_cnt is {} , intra_len is {} , inter_len is {}".format(intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final))
                self.writer.set_step((epoch - 1), 'valid')
                self.valid_metrics.update('loss', eer)
                self.writer.add_image('input', make_grid(data1.cpu(), nrow=8, normalize=True))
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
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
