from torchvision import datasets, transforms
from base import BaseDataLoader

'''
dataloader import from data_loader dir
'''
from data_loader.Verificaiotn_DataLoader import Verificaiotn_DataLoader
from data_loader.taobao_image_loader import taobao_image_loader
'''
MnistDataLoader++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir,test_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, verification=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset,None, batch_size, shuffle, validation_split, num_workers,verification)


