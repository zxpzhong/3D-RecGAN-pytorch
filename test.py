import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.metric import IOU_metric,cross_entropy

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['test_data_loader']['type'])(
        config['test_data_loader']['args']['data_dir'],
        batch_size=2,
        shuffle=False,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    IOU = 0
    CE = 0
    count = 0
    
    with torch.no_grad():
        for i, (X, Y) in enumerate(tqdm(data_loader)):
            X, Y = X.to(device), Y.to(device)
            Y_fake = model.module.unet(X)
            #
            # save sample images, or do something with output here
            #
            # cal test metric
            for i in range(Y_fake.shape[0]):
                IOU+=IOU_metric(Y_fake[i],Y[i])
                count+=1
            # output voxel dimension : 64
            CE+=cross_entropy(Y_fake,Y)/(64*64*64)
    log = {'IOU': IOU / count,'CE': CE / count}
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--log', default='None', type=str,
                    help='log name')
    config = ConfigParser.from_args(args)
    main(config)
