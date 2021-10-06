# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/31 18:58
@Auth ： 王德鑫
@File ：loss.py
@IDE ：PyCharm
@Function: Loss
"""
import math
import time
import torch
import torch.nn.functional as F


def compute_loss(net, xc, target, device):
    able_target = target[:, 0, :, :].to(device)        # (-1, 320, 320)
    angle_target = target[:, 1:-1, :, :].to(device)    # (-1, angle_k, 320, 320)
    width_target = target[:, -1, :, :].to(device)      # (-1, 320, 320)

    able_pred, angle_pred, width_pred = net(xc)         # shape 同上

    # 置信度损失
    able_pred = torch.sigmoid(able_pred)
    able_loss = F.binary_cross_entropy(able_pred.squeeze(), able_target.squeeze())

    # 抓取角损失
    angle_pred = torch.sigmoid(angle_pred)
    angle_loss = F.binary_cross_entropy(angle_pred.squeeze(), angle_target.squeeze())

    # 抓取宽度损失
    width_pred = torch.sigmoid(width_pred)
    width_loss = F.binary_cross_entropy(width_pred.squeeze(), width_target.squeeze())

    return {
        'loss': able_loss + angle_loss * 10 + width_loss,
        'losses': {
            'able_loss': able_loss,
            'angle_loss': angle_loss * 10,
            'width_loss': width_loss,
        },
        'pred': {
            'able': able_pred,      # [-1, 1, 320, 320]
            'angle': angle_pred,    # [-1, angle_k, 320, 320]
            'width': width_pred,    # [-1, 1, 320, 320]
        }
    }


def get_pred(net, xc):
    able_pred, angle_pred, width_pred = net(xc)
    """
        able_pred:  (-1, 1, 300, 300)  
        angle_pred: (-1, angle_k+1, 300, 300)
        width_pred: (-1, 1, 300, 300)
    """
    able_pred = torch.sigmoid(able_pred)
    angle_pred = torch.sigmoid(angle_pred)
    width_pred = torch.sigmoid(width_pred)
    return able_pred, angle_pred, width_pred
