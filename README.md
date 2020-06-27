# An unofficial pyotrch implement of 3D-RecGAN (ICCV Workshops 2017)

The official implement (python2.7+**tensorflow**):https://github.com/Yang7879/3D-RecGAN

Bo Yang, Hongkai Wen, Sen Wang, Ronald Clark, Andrew Markham, Niki Trigoni. In ICCV Workshops, 2017.

https://arxiv.org/abs/1708.07969

## Requirements

- Python \>= 3.5 (3.6 recommended)
- Training : [pytorch](https://github.com/pytorch/pytorch)>=1.0
- torchvision>=0.4.0
- tqdm

## Process

<img src="3d_recgan_sample.png" alt="3d_recgan_sample" style="zoom: 33%;" />

## Data(Provided by author)

https://drive.google.com/open?id=1n4qQzSd_S6Isd6WjKD_sq6LKqn4tiQm9

Data are also available at Baidu Pan:

https://pan.baidu.com/s/165IXaA_JISCwGzTUCiuPig 提取码: gbp2

## Run

`python train.py -c config.json`

## Experiments

### Per-category IoU and CE Loss

Results in paper:

|                   |  IOU  |       |        | CE Loss  |       |        |
| :---------------: | :---: | :---: | :----: | :------: | :---: | :----: |
| trained/tested on | chair | stool | toilet | 3D-RecAE | stool | toilet |
|     3D-RecAE      | 0.633 | 0.488 | 0.520  |  0.069   | 0.085 | 0.166  |
|     3D-RecGAN     | 0.661 | 0.501 | 0.569  |  0.074   | 0.083 | 0.157  |

Re-implement:

|                   |  IOU  |       |        | CE Loss  |       |        |
| :---------------: | :---: | :---: | :----: | :------: | :---: | :----: |
| trained/tested on | chair | stool | toilet | 3D-RecAE | stool | toilet |
|     3D-RecAE      |   *   |   *   |   *    |    *     |   *   |   *    |
|     3D-RecGAN     |   *   |   *   |   *    |    *     |   *   |   *    |

## Todo

- [ ] RecGAN : Per-category IoU and CE Loss
- [ ] RecAE : Per-category IoU and CE Loss
- [ ] Multi-category IoU and CE Loss
- [ ] Cross-category IoU and CE Loss

## Reference repo

- https://github.com/Yang7879/3D-RecGAN
- https://github.com/wolny/pytorch-3dunet
- https://github.com/moemen95/Pytorch-Project-Template