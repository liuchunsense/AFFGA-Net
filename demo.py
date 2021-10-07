# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：demo.py
@ IDE ：PyCharm
@ Function :
"""

import cv2
import os
import torch
import math
from utils.affga import AFFGA


def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode):
    """
    绘制grasp
    file: img路径
    grasps: list()	元素是 [row, col, angle, width]
    mode: arrow / region
    """
    assert mode in ['arrow', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        if mode == 'arrow':
            width = width / 2
            angle2 = calcAngle2(angle)
            k = math.tan(angle)
            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.arrowedLine(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1, 8, 0, 0.5)
            else:
                cv2.arrowedLine(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1, 8, 0, 0.5)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)
            
            color_b = 255 / num * i
            color_r = 0
            color_g = -255 / num * i + 255
            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)
        
        else:
            color_b = 255 / num * i
            color_r = 0
            color_g = -255 / num * i + 255
            img[row, col] = [color_b, color_g, color_r]

    return img

def drawRect(img, rect):
    """
    绘制矩形
    rect: [x1, y1, x2, y2]
    """
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


if __name__ == '__main__':
    # 模型路径
    model = 'path_to_pretrained_model'
    input_path = 'demo/input'
    output_path = 'demo/output'

    # 运行设备
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    # 初始化
    affga = AFFGA(model, device=device_name)
    with torch.no_grad():
        for file in os.listdir(input_path):

            print('processing ', file)

            img_file = os.path.join(input_path, file)
            img = cv2.imread(img_file)

            grasps, x1, y1 = affga.predict(img, device, mode='peak', thresh=0.5, peak_dist=2)	# 预测
            im_rest = drawGrasps(img, grasps, mode='arrow')  # 绘制预测结果
            rect = [x1, y1, x1 + 320, y1 + 320]
            drawRect(im_rest, rect)

            # 保存
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            save_file = os.path.join(output_path, file)
            cv2.imwrite(save_file, im_rest)

    print('FPS: ', affga.fps())
