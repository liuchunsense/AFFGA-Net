import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.affga.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, upSize, angle_cls):
        """
        :param num_classes:
        :param backbone:
        :param BatchNorm:
        :param upSize: 320
        """
        super(Decoder, self).__init__()

        self.upSize = upSize
        self.angleLabel = angle_cls

        # feat_low 卷积
        self.conv_1 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                    BatchNorm(48),
                                    nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Conv2d(512, 48, 1, bias=False),
                                    BatchNorm(48),
                                    nn.ReLU())

        # hasp_small 卷积
        self.conv_hasp_small = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      BatchNorm(256),
                                      nn.ReLU())

        # hasp_mid 卷积
        self.conv_hasp_mid = nn.Sequential(nn.Conv2d(352, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())

        # hasp_big 卷积
        self.conv_hasp_big = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           BatchNorm(256),
                                           nn.ReLU())

        # 抓取置信度预测
        self.able_conv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),

                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),

                                       nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                       nn.ReLU(),

                                       nn.Conv2d(128, 1, kernel_size=1, stride=1))

        # 角度预测
        self.angle_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(256, self.angleLabel, kernel_size=1, stride=1))

        # 抓取宽度预测
        self.width_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(128),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(128, 1, kernel_size=1, stride=1))

        self._init_weight()


    def forward(self, feat_1, hasp_small, hasp_big, hasp_all):
        """
        :param feat_low: Res_1 的输出特征            (-1, 256, 80, 80)
        :param hasp_small: rate = {1, 6}            (-1, 256, 20, 20)
        :param hasp_big: rate = {12, 18}            (-1, 256, 20, 20)
        :param hasp_all: rate = {1, 6, 12, 18}      (-1, 256, 20, 20)
        """
        # feat_1 卷积
        feat_1 = self.conv_1(feat_1)

        # 特征融合
        hasp_small = F.interpolate(hasp_small, size=feat_1.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        hasp_small = torch.cat((hasp_small, feat_1), dim=1)
        hasp_small = self.conv_hasp_small(hasp_small)

        hasp_big = F.interpolate(hasp_big, size=feat_1.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）

        input_able = torch.cat((hasp_small, hasp_big), dim=1)

        # angle width 获取输入
        hasp_all = F.interpolate(hasp_all, size=feat_1.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        hasp_all = torch.cat((hasp_all, feat_1), dim=1)

        # 预测
        able_pred = self.able_conv(input_able)
        angle_pred = self.angle_conv(hasp_all)
        width_pred = self.width_conv(hasp_all)

        return able_pred, angle_pred, width_pred

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm, upSize, angle_cls):
    return Decoder(num_classes, backbone, BatchNorm, upSize, angle_cls)
