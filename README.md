# An unofficial pyotrch implement of 3D-RecGAN (ICCV Workshops 2017)

The official implement (python2.7+**tensorflow**):https://github.com/Yang7879/3D-RecGAN

Bo Yang, Hongkai Wen, Sen Wang, Ronald Clark, Andrew Markham, Niki Trigoni. In ICCV Workshops, 2017.

https://arxiv.org/abs/1708.07969

## 1. Requirements

- Python \>= 3.5 (3.6 recommended)
- Training : [pytorch](https://github.com/pytorch/pytorch)>=1.0
- torchvision>=0.4.0
- tqdm

## 2. Process

<img src="3d_recgan_sample.png" alt="3d_recgan_sample" style="zoom: 33%;" />

## Data(Provided by author)

https://drive.google.com/open?id=1n4qQzSd_S6Isd6WjKD_sq6LKqn4tiQm9

Data are also available at Baidu Pan:

https://pan.baidu.com/s/165IXaA_JISCwGzTUCiuPig 提取码: gbp2

## 3. Run
### 3.1 Train

`python train.py -c config.json`

### 3.2 Test

`python test.py -c config.json`

## 4. Experiments

### 4.1 Per-category IoU and CE Loss

#### 4.1.1 Results in paper :

|                   |  IOU  |       |        | CE Loss  |       |        |
| :---------------: | :---: | :---: | :----: | :------: | :---: | :----: |
| trained/tested on | chair | stool | toilet | 3D-RecAE | stool | toilet |
| 3D-RecAE(CE loss) | 0.633 | 0.488 | 0.520  |  0.069   | 0.085 | 0.166  |
|     3D-RecGAN     | 0.661 | 0.501 | 0.569  |  0.074   | 0.083 | 0.157  |

#### 4.1.2 Re-implement:

|                   |  IOU  |       |        | CE Loss  |       |        |
| :---------------: | :---: | :---: | :----: | :------: | :---: | :----: |
| trained/tested on | chair | stool | toilet | 3D-RecAE | stool | toilet |
| 3D-RecAE(CE loss) | 0.5931|   *   |   *    | 0.0547   |   *   |   *    |
| 3D-RecAE(L1 loss) | 0.5171|   *   |   *    | 0.4769   |   *   |   *    |
|     3D-RecGAN     |   *   |   *   |   *    |    *     |   *   |   *    |

## 5. Todo

- [x] RecGAN : Per-category IoU and CE Loss
- [ ] RecAE : Per-category IoU and CE Loss
- [ ] Multi-category IoU and CE Loss
- [ ] Cross-category IoU and CE Loss

## 6. Reference repo

- https://github.com/Yang7879/3D-RecGAN
- https://github.com/wolny/pytorch-3dunet
- https://github.com/moemen95/Pytorch-Project-Template