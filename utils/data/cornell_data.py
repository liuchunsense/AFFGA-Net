import os
import numpy as np
import glob

from .grasp_data import GraspDatasetBase


class CornellDataset(GraspDatasetBase):
    """
    加载cornell数据集
    """
    def __init__(self, file_path, data_list, data='train', num=-1, test_mode='image-wise', **kwargs):
        """
        :param file_path: Cornell 数据集路径
        :param data_list: 数据集列表文件夹
        :param num: 参与训练或测试的数据量，-1：全部，num：前num个
        :param test_mode: 测试模式 image-wise 或 object-wise
        :param data: train 或 test
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        graspf = []
        if test_mode in ['image-wise', 'object-wise', 'all-wise']:
            train_list_f = os.path.join(file_path, '..',  'train-test', data_list, test_mode + '-' + data + '.txt')
            with open(train_list_f) as f:
                names = f.readlines()
                for name in names:
                    name = name.strip()
                    graspf.append(os.path.join(file_path, name + 'grasp.mat'))
        else:
            raise SystemError('测试模式无效，您只能在[image-wise, object-wise, all-wise]中选择', test_mode)
        graspf.sort()  # 从小到大排序

        if num < 0:
            self.grasp_files = graspf
        else:
            self.grasp_files = graspf[:num]
        
        if len(self.grasp_files) == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
