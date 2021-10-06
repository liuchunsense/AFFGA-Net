import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.affga.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class _HASPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_HASPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class HASP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        """
        :param backbone: resnet
        :param output_stride: 16
        :param BatchNorm:
        """
        super(HASP, self).__init__()

        # HASP
        self.hasp1_1 = _HASPModule(2048, 512, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.hasp1_2 = _HASPModule(2048, 256, 3, padding=6, dilation=6, BatchNorm=BatchNorm)
        self.hasp2_1 = _HASPModule(2048, 512, 3, padding=12, dilation=12, BatchNorm=BatchNorm)
        self.hasp2_2 = _HASPModule(2048, 256, 3, padding=18, dilation=18, BatchNorm=BatchNorm)

        self.conv_small = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5))

        self.conv_big = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False),
                                      BatchNorm(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.conv_all = nn.Sequential(nn.Conv2d(1536, 256, 1, bias=False),
                                      BatchNorm(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self._init_weight()

    def forward(self, x):
        x1_1 = self.hasp1_1(x)
        x1_2 = self.hasp1_2(x)
        x2_1 = self.hasp2_1(x)
        x2_2 = self.hasp2_2(x)

        x_small = torch.cat((x1_1, x1_2), dim=1)
        x_big = torch.cat((x2_1, x2_2), dim=1)
        x_all = torch.cat((x1_1, x1_2, x2_1, x2_2), dim=1)

        x_small = self.conv_small(x_small)
        x_big = self.conv_big(x_big)
        x_all = self.conv_all(x_all)

        return x_small, x_big, x_all

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_hasp(backbone, output_stride, BatchNorm):
    return HASP(backbone, output_stride, BatchNorm)
