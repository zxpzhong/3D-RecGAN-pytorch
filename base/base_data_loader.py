import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
class DataPrefetcher():
    def __init__(self, loader,device):
        self.device = device
        self.loader_ = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': self.num_workers
        }
        # super().__init__(sampler=self.sampler, **self.init_kwargs)
        return DataPrefetcher(DataLoader(**self.init_kwargs),device=torch.device('cuda'))
