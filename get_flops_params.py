"""
模型FLOPs和参数量计算工具

功能:
1. 比较原始模型和剪枝后模型的FLOPs、参数量和通道数
2. 计算剪枝压缩率
3. 支持多种网络架构(VGG, ResNet, GoogLeNet, DenseNet)

使用示例:
python get_flops_params.py 
    --arch vgg_cifar `
    --cfg vgg16 `
    --honey "5,5,5,5,5,5,5,5,5,5,5,5,5"
"""

import torch
import torch.nn as nn
import argparse
import utils.common as utils
from importlib import import_module
from thop import profile  # 用于计算FLOPs和参数量

parser = argparse.ArgumentParser(description='计算剪枝模型的FLOPs和参数量')


parser.add_argument(
    '--arch',
    type=str,
    default='vgg_cifar',
    choices=('vgg_cifar','resnet_cifar','vgg','resnet','densenet','googlenet','vgglayerwise'),
    help='网络架构类型')
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    help='数据集名称(cifar10/imagenet)')
parser.add_argument(
    '--cfg',
    type=str,
    default='resnet56',
    help='具体的网络配置(如vgg16, resnet56等)')
parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='使用的GPU ID列表',
)
parser.add_argument(
    '--depth',
    type=int,
    default=None,
    help='网络深度(用于某些架构)')
parser.add_argument(
    '--honey',
    type=str,
    default=None,
    help='蜜蜂编码（剪枝配置），格式: "5,5,5,5,5,5,5,5,5,5,5,5,5"')
args = parser.parse_args()
honey = list(map(int,args.honey.split(', ')))  # 将字符串解析为整数列表



device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

print('==> Building model..')

# 构建原始模型和剪枝后的模型
if args.arch == 'vgg_cifar':
    orimodel = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)  # 原始VGG
    model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=honey).to(device)  # 剪枝后的VGG
elif args.arch == 'resnet_cifar':
    orimodel = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=honey).to(device)
elif args.arch == 'vgg':
    orimodel = import_module(f'model.{args.arch}').VGG(num_classes=1000).to(device)
    model = import_module(f'model.{args.arch}').BeeVGG(honeysource=honey, num_classes = 1000).to(device)
elif args.arch == 'resnet':
    orimodel = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=honey).to(device)
elif args.arch == 'googlenet':
    orimodel = import_module(f'model.{args.arch}').googlenet().to(device)
    model = import_module(f'model.{args.arch}').googlenet(honey=honey).to(device)
elif args.arch == 'densenet':
    orimodel = import_module(f'model.{args.arch}').densenet().to(device)
    model = import_module(f'model.{args.arch}').densenet(honey=honey).to(device)
elif args.arch == 'vgglayerwise':
    orimodel = import_module(f'model.{args.arch}').VGG(args.cfg, depth = args.depth).to(device)
    model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=honey, depth = args.depth).to(device)

# 根据数据集确定输入图像大小
if args.data_set == 'cifar10':
    input_image_size = 32   # CIFAR10图像是32x32
elif args.data_set == 'imagenet':
    input_image_size = 224  # ImageNet图像是224x224

# 创建随机输入用于计算FLOPs
input = torch.randn(1, 3, input_image_size, input_image_size).to(device)

# 初始化通道计数器
orichannel = 0  # 原始模型总通道数
channel = 0     # 剪枝后模型总通道数

# 使用thop库计算FLOPs和参数量
oriflops, oriparams = profile(orimodel, inputs=(input, ))  # 原始模型
flops, params = profile(model, inputs=(input, ))           # 剪枝后模型

# 统计原始模型的卷积层通道数
for name, module in orimodel.named_modules():
    if isinstance(module, nn.Conv2d):
        orichannel += orimodel.state_dict()[name + '.weight'].size(0)

# 统计剪枝后模型的卷积层通道数
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        channel += model.state_dict()[name + '.weight'].size(0)

# ==================== 输出统计结果 ====================
print('--------------UnPrune Model (原始模型)--------------')
print('Channels: %d'%(orichannel))
print('Params: %.2f M '%(oriparams/1000000))
print('FLOPS: %.2f M '%(oriflops/1000000))

print('--------------Prune Model (剪枝后模型)--------------')
print('Channels:%d'%(channel))
print('Params: %.2f M'%(params/1000000))
print('FLOPS: %.2f M'%(flops/1000000))

print('--------------Compress Rate (压缩率)--------------')
print('Channels Prune Rate: %d/%d (%.2f%%)' % (channel, orichannel, 100. * (orichannel - channel) / orichannel))
print('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % (params/1000000, oriparams/1000000, 100. * (oriparams- params) / oriparams))
print('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops/1000000, oriflops/1000000, 100. * (oriflops- flops) / oriflops))

