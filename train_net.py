import datetime
import os
import sys
import argparse
import logging

import cv2
import time
import numpy as np

import random
import torch
import torch.utils.data
import torch.optim as optim

import tensorboardX
from torchsummary import summary

from utils.data.evaluation import evaluation
from utils.data import get_dataset
from utils.saver import Saver
from models import get_network
from models.common import post_process_output

from models.loss import compute_loss

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Train AFFGA-Net')

    # 数据集
    parser.add_argument('--dataset-path', default='/home/wangdx/dataset/cornell_clutter/data/', type=str, help='Path to dataset')
    parser.add_argument('--test-mode', type=str, default='all-wise', choices=['image-wise', 'object-wise', 'all-wise'], help='测试方式')
    parser.add_argument('--data-list', type=str, default='train-test-all', choices=['train-test-origin', 'train-test-single', 'train-test-mutil'], help='数据列表文件夹')

    # 训练超参数
    parser.add_argument('--finetune', type=bool, default=False, help='是否微调') 
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size') 
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='学习率衰减模式')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    # 抓取表示超参数
    parser.add_argument('--angle-cls', type=int, default=120, help='抓取角分类数')
    parser.add_argument('--bottom', type=int, default=30, help='抓取器尺寸')
    parser.add_argument('--eval-mode', type=str, default='max', choices=['peak', 'all', 'max'], help='抓取评估方法')

    # 保存地址
    parser.add_argument('--outdir', type=str, default='output', help='Training Output Directory')
    parser.add_argument('--modeldir', type=str, default='models', help='model保存地址')
    parser.add_argument('--logdir', type=str, default='tensorboard', help='summary保存文件夹')
    parser.add_argument('--imgdir', type=str, default='img', help='中间预测图保存文件夹')
    parser.add_argument('--max_models', type=int, default='5', help='最大保存的模型数')

    # cuda
    parser.add_argument('--device-name', type=str, default='cuda:1', choices=['cpu', 'cuda:0', 'cuda:1'], help='是否使用GPU')

    # description
    parser.add_argument('--description', type=str, default='train2', help='Training description')

    # 从已有网络继续训练
    parser.add_argument('--goon-train', type=bool, default=False, help='是否从已有网络继续训练')
    parser.add_argument('--model', type=str, default='path_to_pretrained_model', help='保存的模型')
    parser.add_argument('--start-epoch', type=int, default=0, help='继续训练开始的epoch')

    args = parser.parse_args()

    return args


def validate(net, device, val_data, saver, args):
    """
    Run validation.
    :param net: 网络
    :param device:
    :param val_data: 验证数据集
    :param saver: 保存器
    :param args:
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'graspable': 0,
        'fail': [],
        'losses': {
        }
    }

    ld = len(val_data)

    with torch.no_grad():     # 不计算梯度，不反向传播
        batch_idx = 0
        for x, y in val_data:
            batch_idx += 1
            print ("\r Validating... {:.2f}".format(batch_idx/ld), end="")

            lossd = compute_loss(net, x.to(device), y.to(device), device)

            # 统计损失
            loss = lossd['loss']    # 损失和
            results['loss'] += loss.item()/ld       # 损失累加
            for ln, l in lossd['losses'].items():   # 添加单项损失
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()/ld

            # 输出值预处理
            able_out, angle_out, width_out = post_process_output(lossd['pred']['able'], lossd['pred']['angle'], lossd['pred']['width'])

            # 保存预测图
            # saver.save_img(epoch, batch_idx, [able_out_0, able_out_1, yc])

            # 评估
            results['graspable'] += np.max(able_out)/ld

            ret = evaluation(able_out, angle_out, width_out, y, args.angle_cls, args.eval_mode, desc='1')
            if ret:
                results['correct'] += 1
            else:
                results['failed'] += 1
                results['fail'].append(batch_idx)

    return results


def train(epoch, net, device, train_data, optimizer):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param optimizer: Optimizer
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    sum_batch = len(train_data)
    for x, y in train_data:
        """
        x = (-1, 3, 320, 320)
        y = (-1, 2 + angle_k, 320, 320)
        """
        batch_idx += 1

        # 计算损失
        lossd = compute_loss(net, x.to(device), y.to(device), device)

        loss = lossd['loss']        # 损失和

        logging.info('Epoch: {}, '
                     'Batch: {}/{}, '
                     'able_loss: {:.5f}, '
                     'angle_loss: {:.5f}, '
                     'width_loss: {:.5f}, '
                     'Loss: {:0.5f}'.format(
            epoch, batch_idx, sum_batch,
            lossd['losses']['able_loss'], lossd['losses']['angle_loss'], lossd['losses']['width_loss'], loss.item()))

        # 统计损失
        results['loss'] += loss.item()
        for ln, l in lossd['losses'].items():
            if ln not in results['losses']:
                results['losses'][ln] = 0
            results['losses'][ln] += l.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    results['loss'] /= batch_idx    # 计算一个epoch的损失均值
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def datasetloaders(Dataset, args):
    train_dataset = Dataset(args.dataset_path,
                            data_list=args.data_list, 
                            data='train',
                            num=-1, 
                            test_mode=args.test_mode,
                            output_size=320,
                            angle_k=args.angle_cls,
                            argument=True)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    train_val_dataset = Dataset(args.dataset_path,
                                data_list=args.data_list, 
                                data='train',
                                num=-1,
                                test_mode=args.test_mode,
                                output_size=320,
                                angle_k=args.angle_cls)
    train_val_data = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)

    # 加载测试集
    val_dataset = Dataset(args.dataset_path,
                          data_list=args.data_list, 
                          data='test',
                          num=-1,
                          test_mode=args.test_mode,
                          output_size=320,
                          angle_k=args.angle_cls)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)

    return train_data, train_val_data, val_data


def run():
    args = parse_args()

    # 设置保存器
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    saver = Saver(args.outdir, args.logdir, args.modeldir, args.imgdir, net_desc)
    # 初始化tensorboard 保存器
    tb = saver.save_summary()

    # 加载数据集
    logging.info('Loading Dataset...')
    Dataset = get_dataset()
    train_data, train_val_data, val_data = datasetloaders(Dataset, args)

    print('>> train dataset: {}'.format(len(train_data) * args.batch_size))
    print('>> train_val dataset: {}'.format(len(train_val_data)))
    print('>> test dataset: {}'.format(len(val_data)))

    # 加载网络
    logging.info('Loading Network...')
    device_name = args.device_name if torch.cuda.is_available() else "cpu"
    if args.goon_train:
        # 从已有网络继续训练
        net = torch.load(args.model, map_location=torch.device(device_name))
    else:
        # 新建网络训练
        affga = get_network()
        net = affga(angle_cls=args.angle_cls, device=device_name)
    device = torch.device(device_name)      # 指定运行设备
    net = net.to(device)

    # 微调，优化器
    if args.finetune:
        print('>> 微调网络')
        for para in net.backbone.parameters():
            para.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr*0.5)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.5)     # 学习率衰减
    logging.info('optimizer Done')

    # 打印网络结构
    # summary(net, (3, 320, 320))            # 将网络结构信息输出到终端
    # saver.save_arch(net, (3, 320, 320))    # 保存至文件 output/arch.txt

    # 训练
    best_acc = 0.0
    start_epoch = args.start_epoch if args.goon_train else 0
    for _ in range(start_epoch):
        scheduler.step()
    for epoch in range(args.epochs)[start_epoch:]:
        logging.info('Beginning Epoch {:02d}, lr={}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

        # 训练
        train_results = train(epoch, net, device, train_data, optimizer)
        scheduler.step()

        # 保存训练日志
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        if epoch % 5 == 0:
            # 使用测试集验证
            logging.info('>>> Validating...')
            test_results = validate(net, device, val_data, saver, args)

            # 打印日志
            print('>>> test_graspable = {:.5f}'.format(test_results['graspable']))
            print('>>> test_acc: %d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                                       test_results['correct']/(test_results['correct']+test_results['failed'])))
            print('>>> pred fail idx：', test_results['fail'])

            # 保存测试集日志
            tb.add_scalar('loss/val_IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
            tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
            tb.add_scalar('pred/val_graspable', test_results['graspable'], epoch)
            for n, l in test_results['losses'].items():
                tb.add_scalar('val_loss/' + n, l, epoch)

            # 使用训练集进行验证
            if epoch % 50 == 0:
                train_val_results = validate(net, device, train_val_data, saver, args)
                print('>>> train_graspable = {:.5f}'.format(train_val_results['graspable']))
                print('>>> train_acc: %d/%d = %f' % (train_val_results['correct'], train_val_results['correct'] + train_val_results['failed'],
                                                            train_val_results['correct'] / (train_val_results['correct'] + train_val_results['failed'])))
                tb.add_scalar('loss/train_val_IOU', train_val_results['correct'] / (train_val_results['correct'] + train_val_results['failed']), epoch)
                tb.add_scalar('loss/train_val_loss', train_val_results['loss'], epoch)
                tb.add_scalar('pred/train_val_graspable', train_val_results['graspable'], epoch)
                for n, l in train_val_results['losses'].items():
                    tb.add_scalar('train_val_loss/' + n, l, epoch)

            # 保存模型
            accuracy = test_results['correct'] / (test_results['correct'] + test_results['failed'])
            if accuracy >= best_acc :
                print('>>> save model: ', 'epoch_%04d_iou_%0.4f' % (epoch, accuracy))
                saver.save_model(net, 'epoch_%04d_iou_%0.4f' % (epoch, accuracy))
                best_acc = accuracy
            else:
                print('>>> save model: ', 'epoch_%04d_iou_%0.4f_' % (epoch, accuracy))
                saver.save_model(net, 'epoch_%04d_iou_%0.4f_' % (epoch, accuracy))
                saver.remove_model(args.max_models)  # 删除多余的模型

    tb.close()


if __name__ == '__main__':
    run()
