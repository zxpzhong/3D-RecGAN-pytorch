'''
使用imgaug对数据库进行扩充后，单独储存
'''
import sys
sys.path.append('../')
import os
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import argparse
import cv2 as cv
import cv2
import skimage
from skimage import exposure
import random
from PIL import Image
from params import Args
from tqdm import tqdm
def random_rotation(image, angle_range=5):
    height, width = image.shape[:2]
    random_angle = np.random.uniform(-angle_range, angle_range)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), random_angle, 1)
    image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return image


def random_shift(image, wrg, hrg):
    height, width = image.shape[:2]
    tx = np.random.uniform(-wrg, wrg) * width
    ty = np.random.uniform(-hrg, hrg) * height
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty]])
    image = cv2.warpAffine(image, translation_matrix, (width, height))
    return image


def random_zoom(image, zoom_range):
    height, width = image.shape[:2]
    zx, zy = 1 + np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0]])
    image = cv2.warpAffine(image, zoom_matrix, (int(width * zy), int(height * zx)))
    new_h, new_w = image.shape[:2]

    x_range = new_h - height
    y_range = new_w - width

    x_start = np.random.randint(x_range) if x_range > 0 else 0
    y_start = np.random.randint(y_range) if y_range > 0 else 0

    image = image[x_start:x_start + height, y_start:y_start + width]

    return image


def random_shear(image, intensity):
    height, width = image.shape[:2]

    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0]])

    image = cv2.warpAffine(image, shear_matrix, (width, height))
    return image


def random_channel_shift(image, intensity=0.1, channel_axis=2):
    x = np.rollaxis(image, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x.astype(np.uint8)


def adjust_gamma(image, gamma=0.05, gain=1):
    gamma = np.random.uniform(1 - gamma, 1 + gamma, 1)[0]
    image = exposure.adjust_gamma(image, gamma)
    #     image = ((image / 256) ** gamma) * 256 * gain
    return image


class NoiseGenerator(object):

    def generated_noise(self, img):
        """
        Choose which noise to produce.
        :param img(.jpg):
        """
        noise_count = random.randint(1, 2)
        for i in range(noise_count):
            index_noise = random.randint(0, 4)
            if index_noise == 0:
                img = self.gaussian_img(img)
            elif index_noise == 1:
                img = self.salt_img(img)
            elif index_noise == 2:
                img = self.rotate_whole_img(img)
            elif index_noise == 3:
                img = self.erode_img(img)
        return img

    def gaussian_img(self, img):
        """
        gaussion noise
        """
        im = cv2.GaussianBlur(img, (3, 3), 2)
        return im

    def salt_img(self, img):
        """
           salt noise
           the number of sale dot is n
        """
        n = int(img.shape[0] * img.shape[1] * 0.001)
        ilist = np.random.randint(0, img.shape[1], n)
        jlist = np.random.randint(0, img.shape[0], n)
        for k in range(n):
            i = ilist[k]
            j = jlist[k]
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j:j + 1, i:i + 1, :] = 255
        return img

    def rotate_whole_img(self, img):
        """
        rotate whole image
        rotate angle is 0 - 20
        """
        angle = np.random.randint(0, 20)
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        im = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        return im

    def erode_img(self, img):
        """
           erode noise
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        im = cv2.erode(img, kernel)
        return im


def imageaug(image):
    ng = NoiseGenerator()
    image = adjust_gamma(image, gamma=0.03)
    image = random_shear(image, intensity=0.02)
    image = random_shift(image, wrg=0.04, hrg=0.08)
    image = random_rotation(image, angle_range=1)
    image = random_zoom(image, (0, 0.05))
    image = random_channel_shift(image, 0.5)
    image = ng.generated_noise(image)
    return image

'''
双面纹理数据库说明：
finger_num  -  roll_flag   -   capture_time    -    left_right    .bmp

finger_num  :  采集手指号码，同一finger_num即为同一手指，一共有100根手指
roll_flag   :  旋转标志，1为正放，2为向左旋转，3为向右旋转
capture_time： 采集次数，每根手指每个旋转状态采集6次
left_right  :  A表示左侧摄像头，B表示右侧摄像头
'''

def augment(args):
    all_ori_imgs = os.listdir(args.input_dir)
    for finger_num in tqdm(range(1,Args.train_subject+1)):
    # for finger_num in tqdm(range(1, 3)):
        for roll_flag in range(1,3+1):
            for capture_time in range(1,6+1):
                img_name = '{}-{}-{}-A.bmp'.format(finger_num,roll_flag,capture_time)
                img = cv.imread(os.path.join(args.input_dir,img_name))
                for count in range(1,24+1):
                    images_aug = imageaug(img)
                    new_img_name = '{}-{}-{}-{}.jpg'.format(finger_num*2-1, roll_flag, capture_time,count)
                    cv.imwrite(os.path.join(args.output_dir, new_img_name), images_aug)

                img_name = '{}-{}-{}-B.bmp'.format(finger_num,roll_flag,capture_time)
                img = cv.imread(os.path.join(args.input_dir,img_name))
                for count in range(1,24+1):
                    images_aug = imageaug(img)
                    new_img_name = '{}-{}-{}-{}.jpg'.format(finger_num*2, roll_flag, capture_time,count)
                    cv.imwrite(os.path.join(args.output_dir, new_img_name), images_aug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', dest='input_dir', type=str, default='/mnt/Disk/felix/database/double_img_roi',help='Input the ori data path')
    parser.add_argument('--output_dir', dest='output_dir', type=str,default='/mnt/Disk/felix/database/double_img_roi_aug/train', help='Input the output_dir path')

    args = parser.parse_args()

    if args.input_dir == None or args.output_dir == None:
        print('error')
        exit(-1)

    print(args)

    augment(args)

