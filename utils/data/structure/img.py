# -*- coding: UTF-8 -*-
"""===============================================
@Author : wangdx
@Date   : 2020/9/1 21:37
==============================================="""
import cv2
import mmcv
import math
import numpy as np

class Image:
    """
    语义分割mask类，本项目中用于affordance分割
    """
    def __init__(self, file):
        self.img = cv2.imread(file)

    def height(self):
        return self.img.shape[0]

    def width(self):
        return self.img.shape[1]

    def crop(self, size, dist=-1):
        """
        裁剪 self.grasp

        args:
            size: int
            dist: int
        return:
            crop_x1, ...
        """
        if dist > 0:
            x_offset = np.random.randint(-1 * dist, dist)
            y_offset = np.random.randint(-1 * dist, dist)
        else:
            x_offset = 0
            y_offset = 0

        crop_x1 = int((self.width() - size) / 2 + x_offset)
        crop_y1 = int((self.height() - size) / 2 + y_offset)
        crop_x2 = crop_x1 + size
        crop_y2 = crop_y1 + size

        self.img = self.img[crop_y1:crop_y2, crop_x1:crop_x2, :]

        return crop_x1, crop_y1, crop_x2, crop_y2


    def rescale(self, scale, interpolation='bilinear'):
        self.img = mmcv.imrescale(self.img, scale, interpolation=interpolation)

    def rotate(self, rota):
        """
        旋转 rota (弧度)
        """
        self.img = mmcv.imrotate(self.img, rota, border_value=(255, 255, 255))


    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical')

        self.img = mmcv.imflip(self.img, direction=flip_direction)


    def _Hue(self, img, bHue, gHue, rHue):
        # 1.计算三通道灰度平均值
        imgB = img[:, :, 0]
        imgG = img[:, :, 1]
        imgR = img[:, :, 2]

        # 下述3行代码控制白平衡或者冷暖色调，下例中增加了b的分量，会生成冷色调的图像，
        # 如要实现白平衡，则把两个+10都去掉；如要生成暖色调，则增加r的分量即可。
        bAve = cv2.mean(imgB)[0] + bHue
        gAve = cv2.mean(imgG)[0] + gHue
        rAve = cv2.mean(imgR)[0] + rHue
        aveGray = (int)(bAve + gAve + rAve) / 3

        # 2计算每个通道的增益系数
        bCoef = aveGray / bAve
        gCoef = aveGray / gAve
        rCoef = aveGray / rAve

        # 3使用增益系数
        imgB = np.expand_dims(np.floor((imgB * bCoef)), axis=2)
        imgG = np.expand_dims(np.floor((imgG * gCoef)), axis=2)
        imgR = np.expand_dims(np.floor((imgR * rCoef)), axis=2)

        dst = np.concatenate((imgB, imgG, imgR), axis=2)
        dst = np.clip(dst, 0, 255).astype(np.uint8)

        return dst


    def color(self, hue=10):
        """
        色调hue、亮度 增强
        """

        # 调节色调
        hue = np.random.uniform(-1 * hue, hue)

        if hue == 0:
            # 一般的概率保持原样 / 白平衡
            if np.random.rand() < 0.5:
                # 白平衡
                self.img = self._Hue(self.img, hue, hue, hue)
        else:
            # 冷暖色调
            bHue = hue if hue > 0 else 0
            gHue = abs(hue)
            rHue = -1 * hue if hue < 0 else 0
            self.img = self._Hue(self.img, bHue, gHue, rHue)


        # 调节亮度
        bright = np.random.uniform(-40, 10)
        imgZero = np.zeros(self.img.shape, self.img.dtype)
        self.img = cv2.addWeighted(self.img, 1, imgZero, 2, bright)


    def nomalise(self):
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()

