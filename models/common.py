import torch
import numpy as np
from skimage.filters import gaussian


def post_process_output(able_pred, angle_pred, width_pred):
    """
    :param able_pred:  (1, 2, 320, 320)      (as torch Tensors)
    :param angle_pred: (1, angle_k, 320, 320)     (as torch Tensors)
    """

    # 抓取置信度
    able_pred = able_pred.squeeze().cpu().numpy()    # (320, 320)
    able_pred = gaussian(able_pred, 1.0, preserve_range=True)

    # 抓取角
    angle_pred = np.argmax(angle_pred.cpu().numpy().squeeze(), 0)   # (320, 320)

    # 抓取宽度
    width_pred = width_pred.squeeze().cpu().numpy() * 200.  # (320, 320)
    width_pred = gaussian(width_pred, 1.0, preserve_range=True)

    return able_pred, angle_pred, width_pred

