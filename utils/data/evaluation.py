import cv2
import math
from skimage.draw import polygon
from skimage.feature import peak_local_max
import torch.nn.functional as F
import numpy as np


def length(pt1, pt2):
    """
    计算两点间的欧氏距离
    :param pt1: [row, col]
    :param pt2: [row, col]
    :return:
    """
    return pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), 0.5)


def diff(k, label):
    """
    计算cls与label的差值
    :param k: int 不大于label的长度
    :param label: 一维数组 array (k, )  label为多标签的标注类别
    :return: min_diff: 最小的差值 int    clss_list: 角度GT的类别 len=1/2/angle_k
    """
    clss = np.argwhere(label == 1)
    clss = np.reshape(clss, newshape=(clss.shape[0],))
    clss_list = list(clss)
    min_diff = label.shape[0] + 1

    for cls in clss_list:
        min_diff = min(min_diff, abs(cls - k))

    return min_diff, clss_list


def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    :param array: 二维array
    :param thresh: float阈值
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs



def rect_loc(row, col, angle, height, bottom):
    """
    计算矩形的四个角的坐标[row, col]
    :param row:矩形中点 row
    :param col:矩形中点 col
    :param angle: 抓取角 弧度
    :param height: 抓取宽度
    :param bottom: 抓取器尺寸
    :param angle_k: 抓取角分类数
    :return:
    """
    xo = np.cos(angle)
    yo = np.sin(angle)

    y1 = row + height / 2 * yo
    x1 = col - height / 2 * xo
    y2 = row - height / 2 * yo
    x2 = col + height / 2 * xo

    return np.array(
        [
         [y1 - bottom/2 * xo, x1 - bottom/2 * yo],
         [y2 - bottom/2 * xo, x2 - bottom/2 * yo],
         [y2 + bottom/2 * xo, x2 + bottom/2 * yo],
         [y1 + bottom/2 * xo, x1 + bottom/2 * yo],
         ]
    ).astype(np.int)



def polygon_iou(polygon_1, polygon_2):
    """
    计算两个多边形的IOU
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :param polygon_2: 同上
    :return:
    """
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0

    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    return intersection / union


def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi



def evaluation(able_out, angle_out, width_out, target, angle_k, eval_mode, angle_th=30, iou_th=0.25, bottom=30, desc='1'):
    """
    评估预测结果
    :param able_out: 抓取置信度     (320, 320)
    :param angle_out: 抓取角       (320, 320)
    :param width_out: 抓取宽度      (320, 320)
    :param target: (1, 2+angle_k, 320, 320)
    :param angle_k: 抓取角分类数
    :param eval_mode: 评估模式：'peak':只选取峰值进行评估；'all':所有超过阈值的都进行评估
    :param angle_th: 角度 阈值
    :param desc:
    :return:
    1、得到graspable最大的预测点p。
    2、以p为中点，th为半径做圆，搜索圆内的label（th=30）
    3、与任意一个label同时满足以下两个条件，认为预测正确：
        1、偏转角小于30°（k<=3）
        2、IOU>0.25
    """

    mon_pt = False
    mon_angle = False

    rows = able_out.shape[0]    # 320
    cols = able_out.shape[1]    # 320

    # label预处理
    able_target = target[0, 0, :, :].cpu().numpy()          # (320, 320)
    angles_target = target[0, 1:-1, :, :].cpu().numpy()     # (angle_k, 320, 320)
    width_target = target[0, -1, :, :].cpu().numpy() * 200.        # (320, 320)

    # 搜索超过抓取置信度的待评估点
    threshold_abs = 0.5 # 0.3
    if eval_mode == 'peak':
        min_distance = 10
        pred_pts = peak_local_max(able_out, min_distance=min_distance, threshold_abs=threshold_abs)
        while pred_pts.shape[0] > 50:
            threshold_abs += 0.05
            pred_pts = peak_local_max(able_out, min_distance=min_distance, threshold_abs=threshold_abs)
            if threshold_abs >= 0.95:
                break
        while pred_pts.shape[0] > 50:
            min_distance += 2
            pred_pts = peak_local_max(able_out, min_distance=min_distance, threshold_abs=threshold_abs)
            if min_distance >= 30:
                break
        # print('threshold_abs={}, min_distance={}, 极大值数量={}'.format(threshold_abs, min_distance, pred_pts.shape[0]))
    elif eval_mode == 'all':
        pred_pts = arg_thresh(able_out, threshold_abs)
        while pred_pts.shape[0] > 50:
            threshold_abs += 0.05
            pred_pts = arg_thresh(able_out, threshold_abs)
            if threshold_abs >= 0.95:
                break
    elif eval_mode == 'max':
        loc = np.argmax(able_out)
        row = loc // able_out.shape[0]
        col = loc % able_out.shape[0]
        pred_pts = np.array([[row, col]])

    else:
        raise Exception("无效的评估选项，您只能在['peak', 'all', 'max']中选择", eval_mode)

    if desc != '1':
        return 0

    thresh = 30  # 搜索圆半径    50
    for idx in range(pred_pts.shape[0]):
        row_pred, col_pred = pred_pts[idx]
        angle_pred_cls = angle_out[row_pred, col_pred]  # 预测的抓取角类别
        width_pred = width_out[row_pred, col_pred]  # 预测的宽度
        angle_pred = angle_pred_cls / angle_k * 2 * math.pi

        rect_pred = rect_loc(row_pred, col_pred, angle_pred, width_pred, bottom) 

        # 2、以p为中点，th为半径做圆，搜索圆内的label（th=30）
        for row in range(rows):
            for col in range(cols):
                # 筛选抓取点
                if able_target[row, col] != 1.:
                    continue

                # 当前点在搜索圆内
                if length([row, col], [row_pred, col_pred]) > thresh:
                    continue
                mon_pt = True

                angle_label = angles_target[:, row, col]  # 抓取角标签 (angle_k,)
                angle_diff, angle_label = diff(angle_pred_cls, angle_label)  # 抓取角偏差， 抓取角GT的类别 len=1/2/angle_k
                angle_diff = angle_diff / angle_k * 360.

                # 1、偏转角小于30°
                if angle_diff > angle_th:
                    continue

                # 2、iou大于0.25
                width_label = width_target[row, col]
                for angle in angle_label:
                    angle = angle / angle_k * 2 * math.pi
                    rect_label = rect_loc(row, col, angle, width_label, bottom)
                    iou = polygon_iou(rect_label, rect_pred)  # 计算矩形框表示法的IOU
                    if iou >= iou_th:
                        # print('!!! 预测正确 !!!')
                        return True

    # print('mon_pt={}, mon_angle={}'.format(mon_pt, mon_angle))
    return False




