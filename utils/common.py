"""
通用工具函数模块

包含训练和评估过程中常用的工具类和函数：
1. AverageMeter: 计算和存储平均值
2. checkpoint: 模型检查点管理
3. accuracy: 准确率计算
4. get_logger: 日志记录器
"""

from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import os

import torch
import logging


class AverageMeter(object):
    """
    计算并存储当前值和平均值

    用于跟踪损失、准确率等指标的移动平均
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计信息"""
        self.val = 0.0      # 当前值
        self.avg = 0.0      # 平均值
        self.sum = 0.0      # 累计和
        self.count = 0      # 样本数

    def update(self, val, n=1):
        """
        更新统计信息

        参数:
            val: 新的数值
            n: 该数值对应的样本数（默认1）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ensure_path(directory):
    """
    确保目录存在，如不存在则创建

    参数:
        directory: 目录路径
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

def mkdir(path):
    """
    递归创建目录

    参数:
        path: 目录路径
    """
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)


class checkpoint():
    """
    模型检查点管理器

    功能:
    1. 创建和管理实验目录结构
    2. 保存模型检查点
    3. 记录配置参数
    4. 保存最优模型
    """
    def __init__(self, args):
        """
        初始化检查点管理器

        参数:
            args: 命令行参数对象

        创建的目录结构:
            job_dir/
            ├── checkpoint/     # 模型检查点
            ├── run/           # 运行日志
            └── config.txt     # 配置文件
        """
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        # 如果设置了reset，删除现有目录
        if args.reset:
            os.system('rm -rf ' + str(args.job_dir))

        def _make_dir(path):
            """创建目录的内部函数"""
            if not os.path.exists(path):
                os.makedirs(path)

        # 创建所需目录
        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        # 保存配置信息到文件
        config_dir = self.job_dir / 'config.txt'
        if not os.path.exists(config_dir):
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

    def save_model(self, state, epoch, is_best):
        """
        保存模型检查点

        参数:
            state: 模型状态字典（包含model, optimizer, epoch等信息）
            epoch: 当前epoch数
            is_best: 是否是最优模型
        """
        save_path = f'{self.ckpt_dir}/model_epoch_{epoch}.pth'
        torch.save(state, save_path)

        # 如果是最优模型，额外保存一份
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pth')

    def save_honey_model(self, state):
        """
        保存ABC算法搜索到的最优剪枝模型

        参数:
            state: 模型状态字典
        """
        save_path = f'{self.ckpt_dir}/bestmodel_after_bee.pth'
        torch.save(state, save_path)


def get_logger(file_path):
    """
    创建并配置日志记录器

    参数:
        file_path: 日志文件保存路径

    返回:
        logger: 配置好的日志记录器对象

    日志会同时输出到文件和控制台
    """
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')

    # 文件处理器
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def accuracy(output, target, topk=(1,)):
    """
    计算Top-K准确率

    参数:
        output: 模型输出logits，shape: (batch_size, num_classes)
        target: 真实标签，shape: (batch_size,)
        topk: 要计算的top-k值元组，例如(1, 5)表示计算Top-1和Top-5准确率

    返回:
        res: Top-K准确率列表，每个元素对应一个k值

    示例:
        >>> output = model(input)
        >>> top1, top5 = accuracy(output, target, topk=(1, 5))
        >>> print(f'Top-1: {top1:.2f}%, Top-5: {top5:.2f}%')
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 获取top-k预测
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # 判断预测是否正确
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # 计算top-k准确率
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
