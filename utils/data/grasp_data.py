import numpy as np
import cv2
from utils.data import get_dataset
import torch
import torch.utils.data
import math
import random
import os
import copy

from utils.data.structure.img import Image
from utils.data.structure.grasp import GraspMat, drawGrasp1



class GraspDatasetBase(torch.utils.data.Dataset):
    def __init__(self, output_size, angle_k, include_depth=False, include_rgb=True,
                 argument=False):
        """
        :param output_size: int 输入网络的图像尺寸
        :param angle_k: 抓取角的分类数
        :param include_depth: 网络输入是否包括深度图
        :param include_rgb: 网络输入是否包括RGB图
        :param random_rotate: 是否随机旋转
        :param random_zoom: 是否随机缩放      # 后期可加是否随机平移
        """

        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.angle_k = angle_k
        self.argument = argument

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        """
        numpy转tensor
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __getitem__(self, idx):
        # 读取img和标签
        label_name = self.grasp_files[idx]
        rgb_name = label_name.replace('grasp.mat', 'r.png')

        image = Image(rgb_name)
        label = GraspMat(label_name)
        # 数据增强
        if self.argument:
            # resize
            scale = np.random.uniform(0.9, 1.1)
            image.rescale(scale)
            label.rescale(scale)
            # rotate
            rota = 30
            rota = np.random.uniform(-1 * rota, rota)
            image.rotate(rota)
            label.rotate(rota)
            # crop
            dist = 30   # 50
            crop_bbox = image.crop(self.output_size, dist)
            label.crop(crop_bbox)
            # flip
            flip = True if np.random.rand() < 0.5 else False
            if flip:
                image.flip()
                label.flip()
            # color
            image.color()
        else:
            # crop
            crop_bbox = image.crop(self.output_size)
            label.crop(crop_bbox)

        # img归一化
        image.nomalise()
        img = image.img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
        # 获取target
        label.decode(angle_cls=self.angle_k)
        target = label.grasp    # (2 + angle_k, 320, 320)

        img = self.numpy_to_torch(img)
        target = self.numpy_to_torch(target)

        return img, target


    def __len__(self):
        return len(self.grasp_files)


if __name__ == '__main__':
    angle_cls = 120
    dataset = 'cornell'
    dataset_path = '/home/wangdx/dataset/cornell_grasp/img/'
    # 加载训练集
    print('Loading Dataset...')
    Dataset = get_dataset(dataset)
    train_dataset = Dataset(dataset_path,
                            mode='num',
                            start=0, end=20,
                            test_mode='image-wise',
                            data='train',
                            argument=True,
                            output_size=320,
                            angle_k=angle_cls)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)

    print('>> dataset: {}'.format(len(train_data)))

    count = 0
    max_w = 0
    for x, y in train_data:
        count += 1
        img = x[0].cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)    # (3, h, w) -> (h, w, 3)
        im = np.zeros(img.shape, dtype=np.uint8)
        im[:, :, 0] = img[:, :, 0]
        im[:, :, 1] = img[:, :, 1]
        im[:, :, 2] = img[:, :, 2]

        able = y[0, 0, :, :].cpu().numpy()          # (320, 320)
        angles = y[0, 1:-1, :, :].cpu().numpy()     # (angle_k, 320, 320)
        widths = y[0, -1, :, :].cpu().numpy()       # (320, 320)

        rows, cols = np.where(able == 1)    # 抓取点
        for i in range(rows.shape[0]):
            row, col = rows[i], cols[i]
            width = widths[row, col] * 200
            angle = np.argmax(angles[:, row, col]) / angle_cls * 2 * np.pi
            drawGrasp1(im, [row, col, angle, width])

        cv2.imwrite('/home/wangdx/research/grasp_detection/img/test/' + str(count) + '.png', im)