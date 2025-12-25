"""
基于人工蜂群算法(ABC)的神经网络剪枝算法 - CIFAR数据集版本

本文件实现了使用人工蜂群算法对神经网络进行自动剪枝的完整流程。
主要功能：
1. 使用ABC算法搜索最优的网络剪枝配置
2. 支持多种网络架构(VGG, ResNet, GoogLeNet, DenseNet)
3. 在CIFAR10/CIFAR100/ImageNet数据集上训练和测试
4. 自动保存最优剪枝模型和训练检查点

REM 使用ABC算法搜索最优剪枝配置并训练
python bee_cifar.py `
    --data_set cifar10 `
    --data_path ./data `
    --arch resnet_cifar `
    --cfg resnet56 `
    --honey_model ./pretrain/resnet56_cifar10.pth `
    --job_dir ./experiments/resnet56_prune `
    --gpus 0 `
    --lr 0.01 `
    --lr_decay_step 50 100 `
    --num_epochs 150 `
    --train_batch_size 128 `
    --calfitness_epoch 2 `
    --max_cycle 10 `
    --max_preserve 9 `
    --food_number 10 `
    --food_limit 5 `
    --random_rule random_pretrain `
    --num_workers 4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model.googlenet import Inception
from utils.options import args
import utils.common as utils

import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from data import cifar10, cifar100, imagenet
from importlib import import_module

# 初始化检查点管理器、设备、日志记录器和损失函数
checkpoint = utils.checkpoint(args)  # 用于保存和加载模型检查点
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'  # 设置计算设备
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))  # 日志记录器
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数

# 不同网络架构的卷积层数量配置
# 这决定了蜂群算法中"食物源"的维度（每个卷积层的保留率）
conv_num_cfg = {
    'vgg16': 13,        # VGG16有13个卷积层
    'resnet56' : 27,    # ResNet56有27个可剪枝的卷积层
    'resnet110' : 54,   # ResNet110有54个可剪枝的卷积层
    'googlenet' : 9,    # GoogLeNet有9个Inception模块
    'densenet':36,      # DenseNet有36个可剪枝的卷积层
    }
food_dimension = conv_num_cfg[args.cfg]  # 食物源维度 = 可剪枝的卷积层数量

# 全局变量声明（将在main中初始化）
loader = None
origin_model = None
oristate_dict = None


# ==================== 人工蜂群算法核心数据结构 ====================

class BeeGroup():
    """
    蜂群个体类 - 代表一个剪枝方案

    在人工蜂群算法中，每个蜜蜂代表一个候选解（剪枝配置）

    属性:
        code: 编码列表，长度为卷积层数量，每个值表示该层保留的通道数等级(1-10)
        fitness: 适应度值，通常为验证集准确率
        rfitness: 相对适应度，用于计算观察蜂选择概率
        trail: 试验次数，记录该食物源未被改进的次数
    """
    def __init__(self):
        super(BeeGroup, self).__init__()
        self.code = []      # 剪枝编码: 大小为卷积层数量，值范围{1,2,3,...,max_preserve}
        self.fitness = 0    # 适应度（准确率）
        self.rfitness = 0   # 相对适应度
        self.trail = 0      # 未改进次数计数器

# 初始化全局变量
best_honey = BeeGroup()      # 全局最优解
NectraSource = []            # 食物源列表（候选解）
EmployedBee = []             # 雇佣蜂列表
OnLooker = []                # 观察蜂列表
best_honey_state = {}        # 最优模型的状态字典



# ==================== 权重继承函数 ====================

def load_vgg_honey_model(model, random_rule):
    """
    为剪枝后的VGG模型加载预训练权重

    根据剪枝后的网络结构，从原始模型中选择性地继承权重。
    支持两种选择策略：
    1. random_pretrain: 随机选择要保留的通道
    2. l1_pretrain: 根据L1范数选择重要的通道

    参数:
        model: 剪枝后的模型
        random_rule: 权重选择规则 ('random_pretrain' 或 'l1_pretrain')
    """
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None  # 记录上一层选择的通道索引，用于处理层间依赖

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            oriweight = oristate_dict[name + '.weight']  # 原始权重
            curweight = state_dict[name + '.weight']     # 当前（剪枝后）权重
            orifilter_num = oriweight.size(0)            # 原始通道数
            currentfilter_num = curweight.size(0)        # 剪枝后通道数

            # 如果通道数不同，需要选择性继承权重
            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
                select_num = currentfilter_num

                # 选择要保留的通道索引
                if random_rule == 'random_pretrain':
                    # 随机选择
                    select_index = random.sample(range(0, orifilter_num-1), select_num)
                    select_index.sort()
                else:
                    # 基于L1范数选择（保留L1范数最大的通道）
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()

                # 继承选定的权重
                if last_select_index is not None:
                    # 需要同时考虑输入和输出通道的选择
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    # 只需要选择输出通道
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index
            else:
                # 通道数相同，直接复制权重
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_dense_honey_model(model, random_rule):
    """
    为剪枝后的DenseNet模型加载预训练权重

    DenseNet的特殊之处在于每层都与之前所有层连接，需要特殊处理

    参数:
        model: 剪枝后的DenseNet模型
        random_rule: 权重选择规则 ('random_pretrain' 或 'l1_pretrain')
    """
    global oristate_dict
    

    state_dict = model.state_dict()

    conv_weight = []
    conv_trans_weight = []
    bn_weight = []
    bn_bias = []

    for i in range(3):
        for j in range(12):
            conv1_weight_name = 'dense%d.%d.conv1.weight' % (i + 1, j)
            conv_weight.append(conv1_weight_name)

            bn1_weight_name = 'dense%d.%d.bn1.weight' % (i + 1, j)
            bn_weight.append(bn1_weight_name)

            bn1_bias_name = 'dense%d.%d.bn1.bias' %(i+1,j)
            bn_bias.append(bn1_bias_name)

    for i in range(2):
        conv1_weight_name = 'trans%d.conv1.weight' % (i + 1)
        conv_weight.append(conv1_weight_name)
        conv_trans_weight.append(conv1_weight_name)

        bn_weight_name = 'trans%d.bn1.weight' % (i + 1)
        bn_weight.append(bn_weight_name)

        bn_bias_name = 'trans%d.bn1.bias' % (i + 1)
        bn_bias.append(bn_bias_name)
    
    bn_weight.append('bn.weight')
    bn_bias.append('bn.bias')


    # 处理卷积层权重：选择输入通道
    for k in range(len(conv_weight)):
        conv_weight_name = conv_weight[k]
        oriweight = oristate_dict[conv_weight_name]  # 原始权重
        curweight = state_dict[conv_weight_name]     # 剪枝后权重
        orifilter_num = oriweight.size(1)            # 原始输入通道数
        currentfilter_num = curweight.size(1)        # 剪枝后输入通道数
        select_num = currentfilter_num

        # 如果输入通道数不同，需要选择保留哪些通道
        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                # 随机选择通道
                select_index = random.sample(range(0, orifilter_num-1), select_num)
                select_index.sort()
            else:
                # 基于L1范数选择重要通道
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            # 复制选定的输入通道权重（保留所有输出通道）
            for i in range(curweight.size(0)):
                for index_j, j in enumerate(select_index):
                    state_dict[conv_weight_name][i][index_j] = \
                            oristate_dict[conv_weight_name][i][j]


    # 处理BN层权重：选择输出通道对应的BN参数
    for k in range(len(bn_weight)):
        bn_weight_name = bn_weight[k]
        bn_bias_name = bn_bias[k]
        bn_bias.append(bn_bias_name)
        bn_weight.append(bn_weight_name)
        oriweight = oristate_dict[bn_weight_name]  # 原始BN权重
        curweight = state_dict[bn_weight_name]     # 剪枝后BN权重

        orifilter_num = oriweight.size(0)          # 原始通道数
        currentfilter_num = curweight.size(0)      # 剪枝后通道数
        select_num = currentfilter_num

        # 如果通道数不同，需要选择对应的BN参数
        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                # 随机选择
                select_index = random.sample(range(0, orifilter_num-1), select_num)
                select_index.sort()
            else:
                # 基于L1范数选择
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            # 复制选定通道的BN权重和偏置
            for index_j, j in enumerate(select_index):
                state_dict[bn_weight_name][index_j] = \
                        oristate_dict[bn_weight_name][j]
                state_dict[bn_bias_name][index_j] = \
                        oristate_dict[bn_bias_name][j]

    # 处理全连接层：选择输入特征
    oriweight = oristate_dict['fc.weight']
    curweight = state_dict['fc.weight']
    orifilter_num = oriweight.size(1)       # 原始输入特征数
    currentfilter_num = curweight.size(1)   # 剪枝后输入特征数
    select_num = currentfilter_num

    if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
        if random_rule == 'random_pretrain':
            select_index = random.sample(range(0, orifilter_num-1), select_num)
            select_index.sort()
        else:
            l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
            select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
            select_index.sort()

        # 复制选定的输入特征权重
        for i in range(curweight.size(0)):
            for index_j, j in enumerate(select_index):
                state_dict['fc.weight'][i][index_j] = \
                        oristate_dict['fc.weight'][i][j]

    # 复制未被剪枝的其他层权重
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            # 如果该卷积层不在剪枝列表中，直接复制原始权重
            if conv_name not in conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.BatchNorm2d):
            bn_weight_name = name + '.weight'
            bn_bias_name = name + '.bias'
            # 如果该BN层不在剪枝列表中，直接复制原始参数
            if bn_weight_name not in bn_weight and bn_bias_name not in bn_bias:
                state_dict[bn_weight_name] = oristate_dict[bn_weight_name]
                state_dict[bn_bias_name] = oristate_dict[bn_bias_name]

    model.load_state_dict(state_dict)




def load_google_honey_model(model, random_rule):
    """
    为剪枝后的GoogLeNet模型加载预训练权重

    GoogLeNet使用Inception模块，需要特殊处理多分支结构

    参数:
        model: 剪枝后的GoogLeNet模型
        random_rule: 权重选择规则
    """
    global oristate_dict
    state_dict = model.state_dict()
        
    last_select_index = None
    all_honey_conv_name = []
    all_honey_bn_name = []

    # 遍历模型中的所有Inception模块
    for name, module in model.named_modules():
        if isinstance(module, Inception):
            # 定义需要剪枝的层索引
            honey_filter_channel_index = ['.branch5x5.3']  # 需要剪枝输入和输出通道的层
            honey_channel_index = ['.branch3x3.3', '.branch5x5.6']  # 只需要剪枝输入通道的层
            honey_filter_index = ['.branch3x3.0', '.branch5x5.0']  # 只需要剪枝输出通道的层
            honey_bn_index = ['.branch3x3.1', '.branch5x5.1', '.branch5x5.4']  # 需要剪枝的BN层

            # 收集所有需要剪枝的BN层名称
            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            # 处理需要剪枝输入和输出通道的层（branch5x5.3）
            for weight_index in honey_filter_channel_index:
                last_select_index = None
                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                # 第一步：处理输入通道维度
                orifilter_num = oriweight.size(1)       # 原始输入通道数
                currentfilter_num = curweight.size(1)   # 剪枝后输入通道数

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        # 随机选择输入通道
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        # 基于L1范数选择输入通道
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                # 第二步：处理输出通道维度
                orifilter_num = oriweight.size(0)       # 原始输出通道数
                currentfilter_num = curweight.size(0)   # 剪枝后输出通道数

                select_index_1 = copy.deepcopy(select_index)  # 保存输入通道的选择索引

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        # 随机选择输出通道
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        # 基于L1范数选择输出通道
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                # 同时应用输入和输出通道的选择，复制权重
                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                            state_dict[conv_name][index_i][index_j] = \
                                oristate_dict[conv_name][i][j]



            # 处理只需要剪枝输入通道的层（branch3x3.3, branch5x5.6）
            for weight_index in honey_channel_index:
                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]
                orifilter_num = oriweight.size(1)       # 原始输入通道数
                currentfilter_num = curweight.size(1)   # 剪枝后输入通道数

                # 只处理输入通道维度，输出通道保持不变
                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    # 对所有输出通道，复制选定的输入通道权重
                    for i in range(state_dict[conv_name].size(0)):
                        for index_j, j in enumerate(select_index):
                            state_dict[conv_name][i][index_j] = \
                                oristate_dict[conv_name][i][j]

            # 处理只需要剪枝输出通道的层（branch3x3.0, branch5x5.0）
            for weight_index in honey_filter_index:
                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(0)       # 原始输出通道数
                currentfilter_num = curweight.size(0)   # 剪枝后输出通道数

                # 只处理输出通道维度，输入通道保持不变
                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    # 复制选定的输出通道权重
                    for index_i, i in enumerate(select_index):
                            state_dict[conv_name][index_i] = \
                                oristate_dict[conv_name][i]


    # 复制未被剪枝的其他层权重到新网络
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 如果卷积层不在剪枝列表中，直接复制原始权重和偏置
            if name not in all_honey_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):
            # 如果BN层不在剪枝列表中，直接复制所有BN参数
            if name not in all_honey_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            # 全连接层直接复制
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_resnet_honey_model(model, random_rule):
    """
    为剪枝后的ResNet模型加载预训练权重

    ResNet使用残差连接，需要特别注意shortcut连接的处理

    参数:
        model: 剪枝后的ResNet模型
        random_rule: 权重选择规则
    """
    cfg = {
           'resnet56': [9,9,9],     # ResNet56: 3个stage，每个stage 9个block
           'resnet110': [18,18,18],  # ResNet110: 3个stage，每个stage 18个block
           }

    global oristate_dict
    state_dict = model.state_dict()
        
    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []  # 记录所有被剪枝的卷积层

    # 遍历ResNet的所有层和块
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'  # layer1, layer2, layer3
        for k in range(num):  # 遍历每个stage中的block
            for l in range(2):  # 每个ResNet block有2个卷积层
                conv_name = layer_name + str(k) + '.conv' + str(l+1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]  # 原始权重
                curweight = state_dict[conv_weight_name]     # 剪枝后权重
                orifilter_num = oriweight.size(0)            # 原始输出通道数
                currentfilter_num = curweight.size(0)        # 剪枝后输出通道数

                # 如果输出通道数不同，需要选择保留哪些通道
                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        # 随机选择输出通道
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        # 基于L1范数选择输出通道
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    # 根据上一层的输出通道选择来决定当前层的输入通道
                    if last_select_index is not None:
                        # 上一层有剪枝，需要同时选择输入和输出通道
                        logger.info('last_select_index'.format(last_select_index))
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        # 上一层没有剪枝，只需要选择输出通道
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index  # 记录当前层的选择，供下一层使用

                # 当前层没有剪枝，但上一层有剪枝
                elif last_select_index != None:
                    # 只需要根据上一层的输出选择当前层的输入通道
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None  # 当前层没有剪枝，清空选择索引

                # 当前层和上一层都没有剪枝
                else:
                    state_dict[conv_weight_name] = oriweight  # 直接复制权重
                    last_select_index = None

    # 复制未被剪枝的其他层权重
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            # 如果卷积层不在剪枝列表中（如第一层卷积、shortcut卷积），直接复制
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            # 全连接层直接复制权重和偏置
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


# ==================== 训练和测试函数 ====================

def train(model, optimizer, trainLoader, args, epoch):
    """
    训练函数 - 在训练集上训练模型一个epoch

    参数:
        model: 要训练的模型
        optimizer: 优化器
        trainLoader: 训练数据加载器
        args: 命令行参数
        epoch: 当前epoch数
    """

    model.train()  # 设置模型为训练模式（启用dropout和batch normalization）
    losses = utils.AverageMeter()  # 用于跟踪损失的平均值
    accurary = utils.AverageMeter()  # 用于跟踪准确率的平均值
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10  # 每10%的batch打印一次信息
    start_time = time.time()

    # 遍历训练数据集的所有batch
    for batch, (inputs, targets) in enumerate(trainLoader):
        # 将数据移动到指定设备（CPU或GPU）
        inputs, targets = inputs.to(device), targets.to(device)

        # 清零梯度（PyTorch会累积梯度，每次迭代前需要清零）
        optimizer.zero_grad()

        # 前向传播：计算模型输出
        output = model(inputs)

        # 计算损失函数
        loss = loss_func(output, targets)

        # 反向传播：计算梯度
        loss.backward()

        # 更新损失统计
        losses.update(loss.item(), inputs.size(0))

        # 更新模型参数（梯度下降）
        optimizer.step()

        # 计算准确率
        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        # 每隔一定batch数打印训练信息
        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time

def test(model, testLoader):
    """
    测试函数 - 在测试集上评估模型性能

    参数:
        model: 要评估的模型
        testLoader: 测试数据加载器

    返回:
        accurary.avg: 平均准确率
    """
    global best_acc
    model.eval()  # 设置模型为评估模式（禁用dropout和batch normalization的训练行为）

    losses = utils.AverageMeter()  # 用于跟踪损失的平均值
    accurary = utils.AverageMeter()  # 用于跟踪准确率的平均值

    start_time = time.time()

    # 禁用梯度计算以节省内存和加快计算速度
    with torch.no_grad():
        # 遍历测试数据集的所有batch
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            # 将数据移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播：计算模型输出
            outputs = model(inputs)

            # 计算损失
            loss = loss_func(outputs, targets)

            # 更新损失统计
            losses.update(loss.item(), inputs.size(0))

            # 计算准确率并更新统计
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg

# ==================== ABC算法核心函数 ====================

def calculationFitness(honey, train_loader, args):
    """
    计算一个食物源（剪枝方案）的适应度

    流程:
    1. 根据honey编码构建剪枝后的模型
    2. 加载预训练权重
    3. 在训练集上训练若干epoch
    4. 在测试集上评估准确率作为适应度

    参数:
        honey: 蜜蜂编码（剪枝配置列表）
        train_loader: 训练数据加载器
        args: 命令行参数

    返回:
        fit_accurary.avg: 该剪枝方案的测试准确率（适应度）
    """
    global best_honey
    global best_honey_state

    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').BeeVGG(args.cfg,honeysource=honey).to(device)
        load_vgg_honey_model(model, args.random_rule)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=honey).to(device)
        load_resnet_honey_model(model, args.random_rule)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(honey=honey).to(device)
        load_google_honey_model(model, args.random_rule)
    elif args.arch == 'densenet':
        model = import_module(f'model.{args.arch}').densenet(honey=honey).to(device)
        load_dense_honey_model(model, args.random_rule)

    fit_accurary = utils.AverageMeter()  # 记录测试集准确率（适应度）
    train_accurary = utils.AverageMeter()  # 记录训练集准确率

    # 如果使用多GPU，启用数据并行
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    # 创建优化器：使用SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 训练阶段：在训练集上训练calfitness_epoch个epoch
    model.train()
    for epoch in range(args.calfitness_epoch):
        # 遍历训练数据
        for batch, (inputs, targets) in enumerate(train_loader):
            # 将数据移到GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            output = model(inputs)

            # 计算损失
            loss = loss_func(output, targets)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 记录训练准确率
            prec1 = utils.accuracy(output, targets)
            train_accurary.update(prec1[0], inputs.size(0))

    # 测试阶段：在测试集上评估模型性能
    model.eval()
    with torch.no_grad():
        # 遍历测试数据
        for batch_idx, (inputs, targets) in enumerate(loader.testLoader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算准确率
            predicted = utils.accuracy(outputs, targets)
            fit_accurary.update(predicted[0], inputs.size(0))


    # 如果当前剪枝方案的适应度超过历史最优，更新最优解
    if fit_accurary.avg > best_honey.fitness:
        # 保存最优模型的权重（多GPU情况下需要访问module属性）
        best_honey_state = copy.deepcopy(model.module.state_dict() if len(args.gpus) > 1 else model.state_dict())
        # 保存最优剪枝编码
        best_honey.code = copy.deepcopy(honey)
        # 更新最优适应度
        best_honey.fitness = fit_accurary.avg

    return fit_accurary.avg  # 返回当前方案的适应度


def initilize():
    """
    初始化人工蜂群算法

    1. 创建food_number个食物源（随机剪枝方案）
    2. 为每个食物源计算初始适应度
    3. 初始化雇佣蜂和观察蜂
    4. 记录初始最优解
    """
    print('==> Initilizing Honey_model..')
    print(f'==> Total food sources to initialize: {args.food_number}')
    global best_honey, NectraSource, EmployedBee, OnLooker

    # 创建food_number个食物源及其对应的蜜蜂
    for i in range(args.food_number):
        print(f'\n==> Initializing food source [{i+1}/{args.food_number}]')
        init_start_time = time.time()

        # 创建蜜蜂群体对象
        NectraSource.append(copy.deepcopy(BeeGroup()))  # 食物源
        EmployedBee.append(copy.deepcopy(BeeGroup()))   # 雇佣蜂
        OnLooker.append(copy.deepcopy(BeeGroup()))      # 观察蜂

        # 为第i个食物源随机生成剪枝编码（每层的保留等级：1-9）
        for j in range(food_dimension):
            NectraSource[i].code.append(copy.deepcopy(random.randint(1, args.max_preserve)))

        print(f'    Generated code: {NectraSource[i].code}')
        print(f'    Calculating fitness (training for {args.calfitness_epoch} epochs)...')

        # 初始化食物源的属性
        NectraSource[i].fitness = calculationFitness(NectraSource[i].code, loader.trainLoader, args)  # 计算适应度
        NectraSource[i].rfitness = 0  # 相对适应度（用于概率计算）
        NectraSource[i].trail = 0     # 未改进次数计数器

        init_end_time = time.time()
        print(f'    Fitness: {float(NectraSource[i].fitness):.2f}% | Time: {init_end_time - init_start_time:.2f}s')

        # 初始化雇佣蜂（复制食物源的信息）
        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        EmployedBee[i].fitness = NectraSource[i].fitness
        EmployedBee[i].rfitness = NectraSource[i].rfitness
        EmployedBee[i].trail = NectraSource[i].trail

        # 初始化观察蜂（复制食物源的信息）
        OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
        OnLooker[i].fitness = NectraSource[i].fitness
        OnLooker[i].rfitness = NectraSource[i].rfitness
        OnLooker[i].trail = NectraSource[i].trail

    # 初始化全局最优解（选择适应度最高的食物源）
    best_index = 0
    for i in range(1, args.food_number):
        if NectraSource[i].fitness > NectraSource[best_index].fitness:
            best_index = i

    best_honey.code = copy.deepcopy(NectraSource[best_index].code)
    best_honey.fitness = NectraSource[best_index].fitness
    best_honey.rfitness = NectraSource[best_index].rfitness
    best_honey.trail = NectraSource[best_index].trail

    print(f'\n==> Initialization complete!')
    print(f'==> Initial best fitness: {float(best_honey.fitness):.2f}% (food source {best_index+1})')

def sendEmployedBees():
    """
    派遣雇佣蜂阶段

    每个雇佣蜂负责一个食物源:
    1. 随机选择另一个食物源进行比较
    2. 生成新的候选解（基于当前解和随机解的差异）
    3. 如果新解更好，则更新食物源；否则增加trail计数
    """
    global NectraSource, EmployedBee
    print(f'\n==> Sending Employed Bees (Total: {args.food_number})')

    # 遍历每个食物源，派遣对应的雇佣蜂
    for i in range(args.food_number):
        print(f'    Employed Bee [{i+1}/{args.food_number}] searching...', end=' ', flush=True)
        bee_start_time = time.time()
        # 随机选择一个不同的食物源k进行比较
        while 1:
            k = random.randint(0, args.food_number-1)
            if k != i:  # 确保k不等于i
                break

        # 雇佣蜂从当前食物源出发
        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)

        # 随机选择honeychange_num个维度进行变异
        param2change = np.random.randint(0, food_dimension-1, args.honeychange_num)
        # 生成随机扰动因子R（范围：-1到1）
        R = np.random.uniform(-1, 1, args.honeychange_num)

        # 对选中的维度进行变异：Vi = Xi + R*(Xi - Xk)
        for j in range(args.honeychange_num):
            EmployedBee[i].code[param2change[j]] = int(
                NectraSource[i].code[param2change[j]] +
                R[j] * (NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]])
            )
            # 边界检查：确保编码值在[1, max_preserve]范围内
            if EmployedBee[i].code[param2change[j]] < 1:
                EmployedBee[i].code[param2change[j]] = 1
            if EmployedBee[i].code[param2change[j]] > args.max_preserve:
                EmployedBee[i].code[param2change[j]] = args.max_preserve

        # 计算新解的适应度
        EmployedBee[i].fitness = calculationFitness(EmployedBee[i].code, loader.trainLoader, args)

        bee_end_time = time.time()
        # 贪婪选择：如果新解更好，则更新食物源
        if EmployedBee[i].fitness > NectraSource[i].fitness:
            improvement = float(EmployedBee[i].fitness - NectraSource[i].fitness)
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)
            NectraSource[i].trail = 0  # 重置未改进计数器
            NectraSource[i].fitness = EmployedBee[i].fitness
            print(f'Improved! {float(EmployedBee[i].fitness):.2f}% (↑{improvement:.2f}%) [{bee_end_time - bee_start_time:.1f}s]')
        else:
            # 新解不如原解，增加未改进计数器
            NectraSource[i].trail = NectraSource[i].trail + 1
            print(f'No improvement. Keep {float(NectraSource[i].fitness):.2f}% (trail={NectraSource[i].trail}) [{bee_end_time - bee_start_time:.1f}s]')

def calculateProbabilities():
    """
    计算每个食物源的选择概率

    根据适应度计算相对适应度（rfitness），用于观察蜂阶段的轮盘赌选择
    适应度越高的食物源，被观察蜂选中的概率越大
    """
    global NectraSource

    # 找到当前所有食物源中的最大适应度
    maxfit = NectraSource[0].fitness
    for i in range(1, args.food_number):
        if NectraSource[i].fitness > maxfit:
            maxfit = NectraSource[i].fitness

    # 计算每个食物源的相对适应度（归一化到[0.1, 1.0]区间）
    # 公式：rfitness = 0.9 * (fitness / maxfit) + 0.1
    # 这样即使适应度最低的食物源也有0.1的概率被选中
    for i in range(args.food_number):
        NectraSource[i].rfitness = (0.9 * (NectraSource[i].fitness / maxfit)) + 0.1

def sendOnlookerBees():
    """
    派遣观察蜂阶段

    观察蜂根据食物源的适应度概率性地选择食物源:
    1. 使用轮盘赌选择策略（基于rfitness）
    2. 对选中的食物源进行邻域搜索
    3. 如果找到更好的解则更新，否则增加trail计数
    """
    global NectraSource, EmployedBee, OnLooker
    print(f'\n==> Sending Onlooker Bees (Total: {args.food_number})')
    i = 0  # 当前考察的食物源索引
    t = 0  # 已派出的观察蜂数量

    # 使用轮盘赌选择策略派遣food_number只观察蜂
    while t < args.food_number:
        # 生成随机数，用于轮盘赌选择
        R_choosed = random.uniform(0, 1)

        # 如果随机数小于当前食物源的相对适应度，则选中该食物源
        if R_choosed < NectraSource[i].rfitness:
            t += 1  # 观察蜂计数+1
            print(f'    Onlooker Bee [{t}/{args.food_number}] selected source {i+1}...', end=' ', flush=True)
            onlooker_start_time = time.time()

            # 随机选择另一个食物源k进行比较
            while 1:
                k = random.randint(0, args.food_number-1)
                if k != i:
                    break

            # 观察蜂从选中的食物源出发
            OnLooker[i].code = copy.deepcopy(NectraSource[i].code)

            # 随机选择honeychange_num个维度进行变异
            param2change = np.random.randint(0, food_dimension-1, args.honeychange_num)
            # 生成随机扰动因子
            R = np.random.uniform(-1, 1, args.honeychange_num)

            # 对选中的维度进行变异（与雇佣蜂相同的变异策略）
            for j in range(args.honeychange_num):
                OnLooker[i].code[param2change[j]] = int(
                    NectraSource[i].code[param2change[j]] +
                    R[j] * (NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]])
                )
                # 边界检查
                if OnLooker[i].code[param2change[j]] < 1:
                    OnLooker[i].code[param2change[j]] = 1
                if OnLooker[i].code[param2change[j]] > args.max_preserve:
                    OnLooker[i].code[param2change[j]] = args.max_preserve

            # 计算新解的适应度
            OnLooker[i].fitness = calculationFitness(OnLooker[i].code, loader.trainLoader, args)

            onlooker_end_time = time.time()
            # 贪婪选择：如果新解更好，则更新食物源
            if OnLooker[i].fitness > NectraSource[i].fitness:
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)
                NectraSource[i].trail = 0  # 重置未改进计数器
                NectraSource[i].fitness = OnLooker[i].fitness
                print(f'Improved! {float(OnLooker[i].fitness):.2f}% [{onlooker_end_time - onlooker_start_time:.1f}s]')
            else:
                # 新解不如原解，增加未改进计数器
                NectraSource[i].trail = NectraSource[i].trail + 1
                print(f'No improvement. [{onlooker_end_time - onlooker_start_time:.1f}s]')

        # 移动到下一个食物源
        i += 1
        if i == args.food_number:
            i = 0  # 循环到第一个食物源

def sendScoutBees():
    """
    派遣侦察蜂阶段

    如果某个食物源长时间未改进（trail >= food_limit），则放弃该食物源:
    1. 找到trail最大的食物源
    2. 随机生成新的食物源代替它
    3. 重新计算适应度
    """
    global NectraSource, EmployedBee, OnLooker

    # 找到trail计数最大的食物源（最久未改进的食物源）
    maxtrailindex = 0
    for i in range(args.food_number):
        if NectraSource[i].trail > NectraSource[maxtrailindex].trail:
            maxtrailindex = i

    # 如果该食物源的trail超过限制，则放弃它并重新初始化
    if NectraSource[maxtrailindex].trail >= args.food_limit:
        print(f'\n==> Sending Scout Bee: Abandoning food source {maxtrailindex+1} (trail={NectraSource[maxtrailindex].trail})')
        scout_start_time = time.time()

        # 为每个维度随机生成新的编码值
        for j in range(food_dimension):
            R = random.uniform(0, 1)
            NectraSource[maxtrailindex].code[j] = int(R * args.max_preserve)
            # 确保编码值至少为1
            if NectraSource[maxtrailindex].code[j] == 0:
                NectraSource[maxtrailindex].code[j] += 1

        print(f'    New random code: {NectraSource[maxtrailindex].code}')
        print(f'    Calculating fitness...')

        # 重置trail计数器
        NectraSource[maxtrailindex].trail = 0
        # 计算新食物源的适应度
        NectraSource[maxtrailindex].fitness = calculationFitness(
            NectraSource[maxtrailindex].code, loader.trainLoader, args
        )

        scout_end_time = time.time()
        print(f'    New fitness: {float(NectraSource[maxtrailindex].fitness):.2f}% [{scout_end_time - scout_start_time:.1f}s]')
    else:
        print(f'\n==> Scout Bee: No food source to abandon (max trail={NectraSource[maxtrailindex].trail}/{args.food_limit})')
 
def memorizeBestSource():
    """
    记忆最优食物源

    遍历所有食物源，更新全局最优解
    """
    global best_honey, NectraSource

    old_best_fitness = best_honey.fitness
    # 遍历所有食物源，寻找适应度最高的
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            # 更新全局最优解
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.fitness = NectraSource[i].fitness

    if best_honey.fitness > old_best_fitness:
        improvement = float(best_honey.fitness - old_best_fitness)
        print(f'\n*** NEW BEST SOLUTION FOUND! Fitness: {float(best_honey.fitness):.2f}% (↑{improvement:.2f}%) ***')
        print(f'*** Best Code: {best_honey.code} ***')


def main():
    """
    主函数 - 完整的剪枝和训练流程

    流程:
    1. 如果没有resume，执行ABC算法搜索最优剪枝配置
    2. 根据最优配置构建剪枝模型
    3. 训练剪枝模型直到收敛
    4. 保存最优模型和检查点
    """
    # 声明全局变量
    global loader, origin_model, oristate_dict

    # 加载数据集
    import datetime
    startup_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f'==> [{startup_time}] Loading Data..')
    if args.data_set == 'cifar10':
        loader = cifar10.Data(args)
    elif args.data_set == 'cifar100':
        loader = cifar100.Data(args)
    else:
        loader = imagenet.Data(args)

    # 加载原始预训练模型
    print(f'==> [{startup_time}] Loading Model..')
    if args.arch == 'vgg_cifar':
        origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    elif args.arch == 'resnet_cifar':
        origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    elif args.arch == 'googlenet':
        origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
    elif args.arch == 'densenet':
        origin_model = import_module(f'model.{args.arch}').densenet().to(device)

    # 检查预训练模型路径是否存在
    if args.honey_model is None or not os.path.exists(args.honey_model):
        raise ('Honey_model path should be exist!')

    # 加载预训练模型权重
    ckpt = torch.load(args.honey_model, map_location=device)
    origin_model.load_state_dict(ckpt['state_dict'])
    oristate_dict = origin_model.state_dict()  # 保存原始模型的状态字典，用于后续权重继承

    start_epoch = 0  # 训练起始epoch
    best_acc = 0.0   # 记录最优准确率
    code = []        # 剪枝编码

    # 情况1：从头开始训练（没有resume检查点）
    if args.resume == None:
        # 首先测试原始模型的性能作为baseline
        test(origin_model, loader.testLoader)

        # 如果没有指定best_honey，需要运行ABC算法搜索最优剪枝配置
        if args.best_honey == None:
            start_time = time.time()
            bee_start_time = time.time()

            print('==> Start BeePruning..')

            # 初始化蜂群和食物源
            initilize()

            # 早停机制相关变量
            no_improvement_cycles = 0  # 记录连续无改进的周期数
            early_stop_patience = 3    # 连续3个周期无改进则早停
            last_best_fitness = best_honey.fitness

            # ABC算法主循环：执行max_cycle个搜索周期
            for cycle in range(args.max_cycle):
                current_time = time.time()
                print('\n' + '='*80)
                print(f'SEARCH CYCLE [{cycle+1}/{args.max_cycle}]')
                print(f'Current Best Fitness: {float(best_honey.fitness):.2f}%')
                print(f'Current Best Code: {best_honey.code}')
                if cycle > 0:
                    print(f'Cycle Time: {current_time - start_time:.2f}s')
                print('='*80)

                logger.info(
                    'Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime {:.2f}s\n'
                    .format(cycle, best_honey.code, float(best_honey.fitness), (current_time - start_time) if cycle > 0 else 0)
                )
                start_time = time.time()

                # 第一阶段：派遣雇佣蜂
                sendEmployedBees()

                # 第二阶段：计算选择概率
                print('\n==> Calculating Probabilities...')
                calculateProbabilities()

                # 第三阶段：派遣观察蜂
                sendOnlookerBees()

                # 第四阶段：派遣侦察蜂（替换停滞的食物源）
                sendScoutBees()

                # 第五阶段：记忆最优解
                memorizeBestSource()

                # 早停机制：检查是否有改进
                if best_honey.fitness > last_best_fitness:
                    # 有改进，重置计数器
                    no_improvement_cycles = 0
                    last_best_fitness = best_honey.fitness
                else:
                    # 无改进，增加计数器
                    no_improvement_cycles += 1
                    print(f'\n>>> No improvement for {no_improvement_cycles} consecutive cycle(s)')

                    if no_improvement_cycles >= early_stop_patience:
                        print(f'\n>>> Early stopping triggered! No improvement for {early_stop_patience} cycles.')
                        print(f'>>> Final Best Fitness: {float(best_honey.fitness):.2f}%')
                        print(f'>>> Stopping at cycle {cycle+1}/{args.max_cycle}')
                        break

            print('==> BeePruning Complete!')
            bee_end_time = time.time()
            logger.info(
                'Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime Used{:.2f}s\n'
                .format(best_honey.code, float(best_honey.fitness), (bee_end_time - bee_start_time))
            )
        else:
            # 如果用户提供了best_honey参数，直接使用该剪枝配置
            best_honey.code = args.best_honey

        # 根据最优剪枝配置构建剪枝后的模型
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=best_honey.code).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=best_honey.code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(honey=best_honey.code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(honey=best_honey.code).to(device)

        # 加载模型权重
        if args.best_honey_s:
            # 如果指定了best_honey_s路径，从该检查点加载权重
            bestckpt = torch.load(args.best_honey_s)
            model.load_state_dict(bestckpt)
        else:
            # 否则使用ABC算法搜索过程中保存的最优权重
            model.load_state_dict(best_honey_state)

        # 保存剪枝后的模型
        checkpoint.save_honey_model(model.state_dict())

        print(args.random_rule + ' Done!')

        # 如果使用多GPU，启用数据并行
        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)

        # 如果是通过ABC搜索得到的配置，需要创建优化器
        if args.best_honey == None:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

        # 记录剪枝编码
        code = best_honey.code

    else:
        # 情况2：从检查点恢复训练
        resumeckpt = torch.load(args.resume)
        state_dict = resumeckpt['state_dict']  # 模型权重
        code = resumeckpt['honey_code']        # 剪枝编码

        # 根据剪枝编码构建模型
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=code).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(honey=code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(honey=code).to(device)

        # 创建优化器和学习率调度器
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

        # 从检查点恢复模型、优化器、调度器状态
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(resumeckpt['optimizer'])
        scheduler.load_state_dict(resumeckpt['scheduler'])
        start_epoch = resumeckpt['epoch']  # 恢复训练起始epoch

        # 如果使用多GPU，启用数据并行
        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)


    # 如果只是测试模式，直接测试后退出
    if args.test_only:
        test(model, loader.testLoader)
    else:
        # 训练模式：从start_epoch开始训练到num_epochs
        best_epoch = start_epoch  # 记录最佳epoch
        for epoch in range(start_epoch, args.num_epochs):
            # 训练一个epoch
            train(model, optimizer, loader.trainLoader, args, epoch)

            # 学习率衰减
            scheduler.step()

            # 在测试集上评估
            test_acc = test(model, loader.testLoader)

            # 判断是否是最优模型
            is_best = best_acc < test_acc
            if is_best:
                best_epoch = epoch
                logger.info('*** New best model found at epoch {} with accuracy {:.2f}% ***'.format(epoch, float(test_acc)))
            best_acc = max(best_acc, test_acc)

            # 获取模型权重（多GPU情况下需要访问module属性）
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

            # 构建检查点字典
            state = {
                'state_dict': model_state_dict,      # 模型权重
                'best_acc': best_acc,                # 最优准确率
                'optimizer': optimizer.state_dict(),  # 优化器状态
                'scheduler': scheduler.state_dict(),  # 学习率调度器状态
                'epoch': epoch + 1,                  # 下一个epoch编号
                'honey_code': code                   # 剪枝编码
            }

            # 保存检查点
            checkpoint.save_model(state, epoch + 1, is_best)

        # 训练结束，打印最优准确率和对应的epoch
        logger.info('Best accurary: {:.3f} at epoch {}'.format(float(best_acc), best_epoch))

if __name__ == '__main__':
    main()
