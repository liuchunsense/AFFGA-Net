# -*- coding: UTF-8 -*-
"""===============================================
@Author : wangdx
@Date   : 2020/9/1 21:37
==============================================="""
import mmcv
import numpy as np
import cv2
import math
import scipy.io as scio
# from mmcv.ops.roi_align import roi_align

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasp(img, label, offset, interval=20):
    """
    绘制抓取标签
        label: (4, h, w)
        offset: (row, col)
    :return:
    """

    grasp_confidence = label[0, :, :]   # 抓取置信度
    grasp_mode = label[1, :, :]         # 抓取模式 0-无约束抓取 1-单向抓取 2-对称抓取
    grasp_angle = label[2, :, :]        # 抓取角
    grasp_width = label[3, :, :]        # 抓取宽度

    # 绘制抓取点
    grasp_point_rows, grasp_point_cols = np.where(grasp_confidence > 0)
    grasp_point_rows = grasp_point_rows + offset[0]
    grasp_point_cols = grasp_point_cols + offset[1]
    img[grasp_point_rows, grasp_point_cols, :] = [0, 255, 0]

    # 绘制抓取角和抓取宽度
    n = 0
    for i, _ in enumerate(grasp_point_rows):
        n += 1
        if n % interval != 0:
            continue
        row, col = grasp_point_rows[i] - offset[0], grasp_point_cols[i] - offset[1]
        width = grasp_width[row, col] * 150. / 2
        angle = grasp_angle[row, col]   # 弧度
        mode = grasp_mode[row, col]

        row, col = row + offset[0], col + offset[1]

        if mode == 0.:      # 无约束抓取
            cv2.circle(img, (col, row), int(width), (255, 245, 0), 1)

        elif mode == 1.:    # 单向抓取
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)

        elif mode == 2.:    # 对称抓取
            angle2 = calcAngle2(angle)
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)

        else:
            raise ValueError

    return img

def drawGrasp1(img, grasp):
    """
    绘制抓取标签
        grasp: [row, col, angle, width]
    :return:
    """

    row, col = int(grasp[0]), int(grasp[1])
    cv2.circle(img, (int(grasp[1]), int(grasp[0])), 2, (0, 255, 0), -1)
    angle = grasp[2]   # 弧度
    width = grasp[3] / 2

    k = math.tan(angle)

    if k == 0:
        dx = width
        dy = 0
    else:
        dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
        dy = k * dx

    if angle < math.pi:
        cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (255, 245, 0), 1)
    else:
        cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (255, 245, 0), 1)


    return img


def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             flag=cv2.INTER_NEAREST,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=flag, borderValue=border_value)
    return rotated

class GraspMat:
    """
    语义分割mask类，本项目中用于affordance分割
    抓取宽度: 原始宽度/200
    """
    def __init__(self, file):
        self.grasp = scio.loadmat(file)['A']   # (4, 480, 640)

    def height(self):
        return self.grasp.shape[1]

    def width(self):
        return self.grasp.shape[2]

    def crop(self, bbox):
        """
        裁剪 self.grasp

        args:
            bbox: list(x1, y1, x2, y2)
        """
        self.grasp = self.grasp[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def rescale(self, scale, interpolation='nearest'):
        ori_shape = self.grasp.shape[1]
        self.grasp = np.stack([
            mmcv.imrescale(grasp, scale, interpolation=interpolation)
            for grasp in self.grasp
        ])
        new_shape = self.grasp.shape[1]
        ratio = new_shape / ori_shape
        # 抓取宽度同时缩放
        self.grasp[3, :, :] = self.grasp[3, :, :] * ratio


    def rotate(self, rota):
        """
        rota: 角度
        """
        self.grasp = np.stack([imrotate(grasp, rota) for grasp in self.grasp])
        # 角度旋转
        rota = rota / 180 * np.pi
        self.grasp[2, :, :] -= rota
        self.grasp[2, :, :] = self.grasp[2, :, :] % (np.pi * 2)
        self.grasp[2, :, :] *= self.grasp[0, :, :]

    def _flipAngle(self, angle_mat, confidence_mat):
        """
        水平翻转angle
        Args:
            angle_mat: (h, w) 弧度
            confidence_mat: (h, w) 抓取置信度
        Returns:
        """
        # 全部水平翻转
        angle_out = (angle_mat // math.pi) * 2 * math.pi + math.pi - angle_mat
        # 将非抓取区域的抓取角置0
        angle_out = angle_out * confidence_mat
        # 所有角度对2π求余
        angle_out = angle_out % (2 * math.pi)

        return angle_out

    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical')

        self.grasp = np.stack([
            mmcv.imflip(grasp, direction=flip_direction)
            for grasp in self.grasp
        ])
        # 抓取角翻转，除了位置翻转，角度值也需要翻转
        self.grasp[2, :, :] = self._flipAngle(self.grasp[2, :, :], self.grasp[0, :, :])

    def _decode(self, mat, angle_cls):
        """
        解析 grasp_mat
        Args:
            mat: np.ndarray (4, h, w)
            angle_cls: 抓取角类别数，36/72/120

        Returns:
                (1 + angle_cls + 1, h, w)  float
        """
        h, w = mat.shape[1:]
        grasp_confidence = mat[0, :, :]
        grasp_mode = mat[1, :, :]
        grasp_angle = mat[2, :, :]
        grasp_width = mat[3, :, :]

        angle_mat = np.zeros((angle_cls, h, w), dtype=np.float)     # -1:不属于抓取点
        grasp_point = np.where(grasp_confidence > 0)
        for i, _ in enumerate(grasp_point[0]):
            row, col = grasp_point[0][i], grasp_point[1][i]
            angle = grasp_angle[row, col]  # 弧度
            mode = grasp_mode[row, col]

            angle_mat[-1, row, col] = 0.

            if mode == 1.:  # 无约束抓取
                angle_mat[:, row, col] = 1.
            elif mode == 2.:  # 单向抓取
                angle1 = int(angle / (2 * np.pi) * angle_cls)  # 弧度转类别
                angle_mat[angle1, row, col] = 1.
            elif mode == 3.:  # 对称抓取
                angle1 = int(angle / (2 * np.pi) * angle_cls)  # 弧度转类别
                angle2 = angle + np.pi - int((angle + np.pi) // (2 * np.pi)) * 2 * np.pi
                angle2 = int(angle2 / (2 * np.pi) * angle_cls)  # 弧度转类别
                angle_mat[angle1, row, col] = 1.
                angle_mat[angle2, row, col] = 1.
            else:
                print('mode error')
                raise ValueError

        grasp_confidence = np.expand_dims(grasp_confidence, axis=0)
        grasp_width = np.expand_dims(grasp_width, axis=0)
        ret_mat = np.zeros(shape=(122, 320, 320))   # 122
        ret_mat[0, :, :] = grasp_confidence
        ret_mat[1:-1, :, :] = angle_mat
        ret_mat[-1, :, :] = grasp_width / 200.

        # print(np.unique(grasp_width))

        return ret_mat


    def decode(self, angle_cls):
        """
        (4, H, W) -> (angle_cls+2, H, W)
        """
        self.grasp = self._decode(self.grasp, angle_cls)
