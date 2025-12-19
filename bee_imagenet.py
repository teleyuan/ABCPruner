"""
基于人工蜂群算法(ABC)的神经网络剪枝算法 - ImageNet数据集版本

与bee_cifar.py类似，但针对ImageNet数据集进行了优化：
1. 使用DALI加速数据加载
2. 支持更大的模型（如ResNet50/101/152）
3. 使用动态学习率调整策略
4. Top-1和Top-5准确率评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils

import os
import copy
import time
import math
import sys
import numpy as np
import heapq
import random
from data import imagenet_dali
from importlib import import_module

# ImageNet数据集支持的网络架构配置
conv_num_cfg = {
    'vgg16' : 13,       # VGG16
	'resnet18' : 8,     # ResNet18有8个可剪枝的卷积块
	'resnet34' : 16,    # ResNet34有16个可剪枝的卷积块
	'resnet50' : 16,    # ResNet50有16个bottleneck块
	'resnet101' : 33,   # ResNet101有33个bottleneck块
	'resnet152' : 50    # ResNet152有50个bottleneck块
}

food_dimension = conv_num_cfg[args.cfg]  # ABC算法的搜索空间维度（对应网络中可剪枝的层数）

# ==================== 初始化设备和工具 ====================
# 设置计算设备（优先使用指定的GPU，否则使用CPU）
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
# 创建检查点管理器（用于保存模型和配置）
checkpoint = utils.checkpoint(args)
# 创建日志记录器（同时输出到文件和控制台）
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
# 定义损失函数（交叉熵损失）
loss_func = nn.CrossEntropyLoss()

# 数据加载器
print('==> Preparing data..')
def get_data_set(type='train'):
    """
    使用NVIDIA DALI加速ImageNet数据加载

    参数:
        type: 'train' 或 'test'

    返回:
        DALI数据迭代器
    """
    if type == 'train':
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
    else:
        return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
# 创建训练集和测试集的DALI数据加载器
trainLoader = get_data_set('train')
testLoader = get_data_set('test')


# 如果不是从头训练模式，需要加载预训练模型（用于BeePruning）
if args.from_scratch == False:

    # 加载原始未剪枝的预训练模型
    print('==> Loading Model..')
    if args.arch == 'vgg':
        origin_model = import_module(f'model.{args.arch}').VGG(num_classes=1000).to(device)
    elif args.arch == 'resnet':
        origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    elif args.arch == 'googlenet':
        pass
    elif args.arch == 'densenet':
        pass

    # 验证预训练模型路径是否存在
    if args.honey_model is None or not os.path.exists(args.honey_model):
        raise ('Honey_model path should be exist!')

    # 加载预训练权重
    ckpt = torch.load(args.honey_model, map_location=device)
    '''
    调试代码：打印模型权重的维度信息
    print("model's state_dict:")
    for param_tensor in ckpt:
        print(param_tensor,'\t',ckpt[param_tensor].size())

    print("origin_model's state_dict:")
    for param_tensor in origin_model.state_dict():
        print(param_tensor,'\t',origin_model.state_dict()[param_tensor].size())
    '''
    # 将预训练权重加载到原始模型
    origin_model.load_state_dict(ckpt)
    # 保存原始模型的状态字典（全局变量，供权重继承函数使用）
    oristate_dict = origin_model.state_dict()

def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """
    动态调整学习率

    策略:
    1. 每30个epoch学习率衰减0.1倍
    2. 在epoch 80之后额外衰减一次
    3. 前5个epoch使用warmup策略逐渐增加学习率

    参数:
        optimizer: 优化器
        epoch: 当前epoch
        step: 当前batch步数
        len_epoch: 每个epoch的batch数
        args: 命令行参数
    """
    factor = epoch // 30  # 每30个epoch衰减一次

    if epoch >= 80:
        factor = factor + 1  # 80 epoch后额外衰减

    lr = args.lr * (0.1 ** factor)

    # Warmup: 前5个epoch逐渐增加学习率
    if epoch < 5 and args.warm_up:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    # 更新所有参数组的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ABC算法实现基于: https://www.cnblogs.com/ybl20000418/p/11366576.html
# ==================== 人工蜂群算法核心数据结构 ====================

class BeeGroup():
    """
    蜂群个体类 - 代表一个剪枝方案

    与bee_cifar.py中的定义相同，用于ImageNet数据集
    """
    def __init__(self):
        super(BeeGroup, self).__init__()
        self.code = []      # 剪枝编码列表
        self.fitness = 0    # 适应度（Top-1准确率）
        self.rfitness = 0   # 相对适应度
        self.trail = 0      # 未改进计数器

# 初始化全局变量
best_honey = BeeGroup()      # 全局最优解
NectraSource = []            # 食物源列表
EmployedBee = []             # 雇佣蜂列表
OnLooker = []                # 观察蜂列表
best_honey_state = {}        # 最优模型状态

# ==================== 权重继承函数 ====================
# 这些函数与bee_cifar.py中的对应函数功能相同

def load_vgg_honey_model(model, random_rule):
    """为剪枝后的VGG模型加载预训练权重"""
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None  # 上一层选择的通道索引（用于处理层间依赖）

    # 遍历模型中的所有卷积层
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 获取原始权重和当前剪枝后的权重
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)       # 原始输出通道数
            currentfilter_num = curweight.size(0)   # 剪枝后输出通道数

            # 如果输出通道数不同，需要选择保留哪些通道
            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
                select_num = currentfilter_num

                # 根据规则选择通道
                if random_rule == 'random_pretrain':
                    # 随机选择输出通道
                    select_index = random.sample(range(0, orifilter_num-1), select_num)
                    select_index.sort()
                else:
                    # 基于L1范数选择输出通道
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()

                # 复制权重到剪枝后的模型
                if last_select_index is not None:
                    # 上一层有剪枝，需要同时选择输入和输出通道
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    # 上一层没有剪枝，只需选择输出通道
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index  # 记录当前层的选择供下一层使用

            else:
                # 当前层没有剪枝，直接复制权重
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_resnet_honey_model(model, random_rule):
    """
    为剪枝后的ResNet模型加载预训练权重（支持ResNet18-152）

    ResNet使用残差连接，需要特别注意shortcut连接的处理
    """
    # ResNet各个版本的配置：每个stage的block数量
    cfg = {'resnet18': [2, 2, 2, 2],     # ResNet18: 4个stage
           'resnet34': [3, 4, 6, 3],      # ResNet34配置
           'resnet50': [3, 4, 6, 3],      # ResNet50配置（使用bottleneck）
           'resnet101': [3, 4, 23, 3],    # ResNet101配置
           'resnet152': [3, 8, 36, 3]}    # ResNet152配置

    global oristate_dict
    state_dict = model.state_dict()

    current_cfg = cfg[args.cfg]
    last_select_index = None  # 上一层选择的通道索引（用于处理层间依赖）

    all_honey_conv_weight = []  # 记录所有被剪枝的卷积层

    # 遍历ResNet的所有层和块
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'  # layer1, layer2, layer3, layer4

        # 遍历每个stage中的block
        for k in range(num):
            # 确定每个block中的卷积层数量
            if args.cfg == 'resnet18' or args.cfg == 'resnet34':
                iter = 2  # BasicBlock有2个卷积层
            else:
                iter = 3  # Bottleneck有3个卷积层

            # 遍历block中的每个卷积层
            for l in range(iter):
                conv_name = layer_name + str(k) + '.conv' + str(l+1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)

                # 获取原始权重和剪枝后权重
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)       # 原始输出通道数
                currentfilter_num = curweight.size(0)   # 剪枝后输出通道数

                # 如果输出通道数不同，需要选择保留哪些通道
                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num

                    # 根据规则选择输出通道
                    if random_rule == 'random_pretrain':
                        # 随机选择输出通道
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        # 基于L1范数选择输出通道
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    # 复制权重到剪枝后的模型
                    if last_select_index is not None:
                        # 上一层有剪枝，需要同时选择输入和输出通道
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        # 上一层没有剪枝，只需选择输出通道
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index  # 记录当前层的选择供下一层使用

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

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):
    """
    训练函数 - 针对ImageNet优化

    特点:
    1. 支持Top-1和Top-5准确率计算
    2. 使用动态学习率调整
    3. DALI数据加载器
    """

    model.train()  # 设置模型为训练模式（启用dropout和batch normalization）
    losses = utils.AverageMeter()        # 用于跟踪损失的平均值
    accuracy = utils.AverageMeter()      # 用于跟踪Top-1准确率的平均值
    top5_accuracy = utils.AverageMeter() # 用于跟踪Top-5准确率的平均值
    print_freq = trainLoader._size // args.train_batch_size // 10  # 每10%的batch打印一次信息
    start_time = time.time()

    # 遍历训练数据集的所有batch（DALI格式）
    for batch, batch_data in enumerate(trainLoader):
        # 提取输入和标签（DALI返回的数据格式）
        inputs = batch_data[0]['data'].to(device)
        targets = batch_data[0]['label'].squeeze().long().to(device)

        # 计算每个epoch的batch数量
        train_loader_len = int(math.ceil(trainLoader._size / args.train_batch_size))

        # 动态调整学习率（包含warmup策略）
        adjust_learning_rate(optimizer, epoch, batch, train_loader_len, args)

        # 前向传播：计算模型输出
        output = model(inputs)

        # 计算损失函数
        loss = loss_func(output, targets)

        # 清零梯度（PyTorch会累积梯度，每次迭代前需要清零）
        optimizer.zero_grad()

        # 反向传播：计算梯度
        loss.backward()

        # 更新损失统计
        losses.update(loss.item(), inputs.size(0))

        # 更新模型参数（梯度下降）
        optimizer.step()

        # 计算Top-1和Top-5准确率
        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))      # Top-1
        top5_accuracy.update(prec1[1], inputs.size(0))  # Top-5

        # 每隔一定batch数打印训练信息
        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Top1 {:.2f}%\t'
                'Top5 {:.2f}%\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, trainLoader._size,
                    float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                )
            )
            start_time = current_time
            

# ==================== 测试函数 ====================
def test(model, testLoader, topk=(1,)):
    """
    测试函数 - 在ImageNet验证集上评估模型

    参数:
        model: 要评估的模型
        testLoader: DALI测试数据加载器
        topk: 计算的Top-K准确率（默认Top-1和Top-5）

    返回:
        top5_accuracy.avg: Top-5准确率
        accuracy.avg: Top-1准确率
    """
    model.eval()  # 设置为评估模式

    losses = utils.AverageMeter()        # 记录损失
    accuracy = utils.AverageMeter()      # 记录Top-1准确率
    top5_accuracy = utils.AverageMeter() # 记录Top-5准确率

    start_time = time.time()

    # 禁用梯度计算
    with torch.no_grad():
        # 遍历测试数据（DALI格式）
        for batch_idx, batch_data in enumerate(testLoader):
            # DALI返回的数据格式：batch_data[0]['data']和batch_data[0]['label']
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)
            targets = targets.cuda(non_blocking=True)  # 异步传输到GPU

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = loss_func(outputs, targets)

            # 更新统计信息
            losses.update(loss.item(), inputs.size(0))

            # 计算Top-1和Top-5准确率
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))      # Top-1
            top5_accuracy.update(predicted[1], inputs.size(0))  # Top-5

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        )

    return top5_accuracy.avg, accuracy.avg

# ==================== 适应度计算函数 ====================
def calculationFitness(honey, args):
    """
    计算一个食物源（剪枝方案）的适应度

    流程（与bee_cifar.py类似但使用DALI和ImageNet）:
    1. 根据honey编码构建剪枝后的模型
    2. 加载预训练权重
    3. 在训练集上训练若干epoch
    4. 在测试集上评估Top-5准确率作为适应度

    参数:
        honey: 蜜蜂编码（剪枝配置列表）
        args: 命令行参数

    返回:
        fit_accurary.avg: 该剪枝方案的Top-5准确率（适应度）
    """
    global best_honey
    global best_honey_state

    # 根据剪枝编码构建模型
    if args.arch == 'vgg':
        model = import_module(f'model.{args.arch}').BeeVGG(honeysource=honey, num_classes=1000).to(device)
        load_vgg_honey_model(model, args.random_rule)
    elif args.arch == 'resnet':
        model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=honey).to(device)
        load_resnet_honey_model(model, args.random_rule)
    elif args.arch == 'googlenet':
        pass
    elif args.arch == 'densenet':
        pass

    # 如果使用多GPU，启用数据并行
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 训练阶段：在训练集上训练calfitness_epoch个epoch
    model.train()
    for epoch in range(args.calfitness_epoch):
        # 遍历训练数据（DALI格式）
        for batch, batch_data in enumerate(trainLoader):
            # 提取输入和标签
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)

            # 计算每个epoch的batch数量
            train_loader_len = int(math.ceil(trainLoader._size / args.train_batch_size))

            # 动态调整学习率（包含warmup）
            adjust_learning_rate(optimizer, epoch, batch, train_loader_len, args)

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

        # DALI需要在每个epoch结束时重置迭代器
        trainLoader.reset()

    # 测试阶段：在测试集上评估模型性能
    fit_accurary = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        # 遍历测试数据
        for batch_idx, batch_data in enumerate(testLoader):
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算Top-1和Top-5准确率
            predicted = utils.accuracy(outputs, targets, topk=(1, 5))
            # 使用Top-5准确率作为适应度（ImageNet更常用Top-5）
            fit_accurary.update(predicted[1], inputs.size(0))

    # 重置测试数据加载器
    testLoader.reset()

    # 避免适应度为0导致的除零问题
    if fit_accurary.avg == 0:
        fit_accurary.avg = 0.01

    # 如果当前剪枝方案的适应度超过历史最优，更新最优解
    if fit_accurary.avg > best_honey.fitness:
        # 保存最优模型权重
        best_honey_state = copy.deepcopy(model.module.state_dict() if len(args.gpus) > 1 else model.state_dict())
        # 保存最优剪枝编码
        best_honey.code = copy.deepcopy(honey)
        # 更新最优适应度
        best_honey.fitness = fit_accurary.avg

    return fit_accurary.avg



# ==================== ABC算法初始化 ====================
def initialize():
    """
    初始化人工蜂群算法

    功能与bee_cifar.py中的initilize()函数相同：
    1. 创建food_number个食物源（随机剪枝方案）
    2. 为每个食物源计算初始适应度
    3. 初始化雇佣蜂和观察蜂
    4. 记录初始最优解
    """
    print('==> Initializing Honey_model..')
    global best_honey, NectraSource, EmployedBee, OnLooker

    # 创建food_number个食物源及其对应的蜜蜂
    for i in range(args.food_number):
        # 创建蜜蜂群体对象
        NectraSource.append(copy.deepcopy(BeeGroup()))  # 食物源
        EmployedBee.append(copy.deepcopy(BeeGroup()))   # 雇佣蜂
        OnLooker.append(copy.deepcopy(BeeGroup()))      # 观察蜂

        # 为第i个食物源随机生成剪枝编码（每层的保留等级：1-max_preserve）
        for j in range(food_dimension):
            NectraSource[i].code.append(copy.deepcopy(random.randint(1, args.max_preserve)))

        # 初始化食物源的属性
        NectraSource[i].fitness = calculationFitness(NectraSource[i].code, args)  # 计算适应度
        NectraSource[i].rfitness = 0  # 相对适应度（用于概率计算）
        NectraSource[i].trail = 0     # 未改进次数计数器

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

    # 初始化全局最优解（使用第一个食物源的信息）
    best_honey.code = copy.deepcopy(NectraSource[0].code)
    best_honey.fitness = NectraSource[0].fitness
    best_honey.rfitness = NectraSource[0].rfitness
    best_honey.trail = NectraSource[0].trail

# ==================== 派遣雇佣蜂 ====================
def sendEmployedBees():
    """
    派遣雇佣蜂阶段

    每个雇佣蜂负责一个食物源:
    1. 随机选择另一个食物源进行比较
    2. 生成新的候选解（基于当前解和随机解的差异）
    3. 如果新解更好，则更新食物源；否则增加trail计数
    """
    global NectraSource, EmployedBee

    # 遍历每个食物源，派遣对应的雇佣蜂
    for i in range(args.food_number):
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
        EmployedBee[i].fitness = calculationFitness(EmployedBee[i].code, args)

        # 贪婪选择：如果新解更好，则更新食物源
        if EmployedBee[i].fitness > NectraSource[i].fitness:
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)
            NectraSource[i].trail = 0  # 重置未改进计数器
            NectraSource[i].fitness = EmployedBee[i].fitness
        else:
            # 新解不如原解，增加未改进计数器
            NectraSource[i].trail = NectraSource[i].trail + 1

# ==================== 计算选择概率 ====================
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

# ==================== 派遣观察蜂 ====================
def sendOnlookerBees():
    """
    派遣观察蜂阶段

    观察蜂根据食物源的适应度概率性地选择食物源:
    1. 使用轮盘赌选择策略（基于rfitness）
    2. 对选中的食物源进行邻域搜索
    3. 如果找到更好的解则更新，否则增加trail计数
    """
    global NectraSource, EmployedBee, OnLooker
    i = 0  # 当前考察的食物源索引
    t = 0  # 已派出的观察蜂数量

    # 使用轮盘赌选择策略派遣food_number只观察蜂
    while t < args.food_number:
        # 生成随机数，用于轮盘赌选择
        R_choosed = random.uniform(0, 1)

        # 如果随机数小于当前食物源的相对适应度，则选中该食物源
        if R_choosed < NectraSource[i].rfitness:
            t += 1  # 观察蜂计数+1

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
            OnLooker[i].fitness = calculationFitness(OnLooker[i].code, args)

            # 贪婪选择：如果新解更好，则更新食物源
            if OnLooker[i].fitness > NectraSource[i].fitness:
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)
                NectraSource[i].trail = 0  # 重置未改进计数器
                NectraSource[i].fitness = OnLooker[i].fitness
            else:
                # 新解不如原解，增加未改进计数器
                NectraSource[i].trail = NectraSource[i].trail + 1

        # 移动到下一个食物源
        i += 1
        if i == args.food_number:
            i = 0  # 循环到第一个食物源

# ==================== 派遣侦察蜂 ====================
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
        # 为每个维度随机生成新的编码值
        for j in range(food_dimension):
            R = random.uniform(0, 1)
            NectraSource[maxtrailindex].code[j] = int(R * args.max_preserve)
            # 确保编码值至少为1
            if NectraSource[maxtrailindex].code[j] == 0:
                NectraSource[maxtrailindex].code[j] += 1

        # 重置trail计数器
        NectraSource[maxtrailindex].trail = 0
        # 计算新食物源的适应度
        NectraSource[maxtrailindex].fitness = calculationFitness(
            NectraSource[maxtrailindex].code, args
        )

# ==================== 记忆最优解 ====================
def memorizeBestSource():
    """
    记忆最优食物源

    遍历所有食物源，更新全局最优解
    """
    global best_honey, NectraSource

    # 遍历所有食物源，寻找适应度最高的
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            # 更新全局最优解
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.fitness = NectraSource[i].fitness


# ==================== 主函数 ====================
def main():
    """
    主函数 - 完整的ImageNet剪枝和训练流程

    支持三种模式:
    1. from_scratch: 从头训练未剪枝的模型
    2. BeePruning: 使用ABC算法搜索最优剪枝配置并训练
    3. resume: 从检查点恢复训练
    """
    start_epoch = 0       # 训练起始epoch
    best_acc = 0.0        # 最优Top-5准确率
    best_acc_top1 = 0.0   # 最优Top-1准确率
    code = []             # 剪枝编码

    # ==================== 模式1：从头训练（不使用剪枝）====================
    if args.from_scratch:
        print('==> Building Model..')

        # 根据架构创建完整模型（未剪枝）
        if args.arch == 'vgg':
            model = import_module(f'model.{args.arch}').VGG(num_classes=1000).to(device)
        elif args.arch == 'resnet':
            model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)

        # 如果使用多GPU，启用数据并行
        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)

        # 创建优化器
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # 如果提供了resume检查点，从检查点恢复
        if args.resume:
            print('=> Resuming from ckpt {}'.format(args.resume))
            ckpt = torch.load(args.resume, map_location=device)
            best_acc = ckpt['best_acc']       # 恢复最优准确率
            start_epoch = ckpt['epoch']        # 恢复epoch计数

            # 恢复模型和优化器状态
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print('=> Continue from epoch {}...'.format(start_epoch))


    # ==================== 模式2：BeePruning（使用ABC算法剪枝）====================
    else:
        # 子模式2.1：从头开始（没有resume检查点）
        if args.resume == None:
            # 首先测试原始模型的性能作为baseline
            test(origin_model, testLoader, topk=(1, 5))
            testLoader.reset()  # DALI需要重置迭代器

            # 如果没有指定best_honey，需要运行ABC算法搜索最优剪枝配置
            if args.best_honey == None:
                start_time = time.time()
                bee_start_time = time.time()

                print('==> Start BeePruning..')

                # 初始化蜂群和食物源
                initialize()

                # ABC算法主循环：执行max_cycle个搜索周期
                for cycle in range(args.max_cycle):
                    current_time = time.time()
                    logger.info(
                        'Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime {:.2f}s\n'
                        .format(cycle, best_honey.code, float(best_honey.fitness), (current_time - start_time))
                    )
                    start_time = time.time()

                    # 第一阶段：派遣雇佣蜂
                    sendEmployedBees()

                    # 第二阶段：计算选择概率
                    calculateProbabilities()

                    # 第三阶段：派遣观察蜂
                    sendOnlookerBees()

                    # 第四阶段：派遣侦察蜂（替换停滞的食物源）
                    sendScoutBees()

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
            if args.arch == 'vgg':
                model = import_module(f'model.{args.arch}').BeeVGG(honeysource=best_honey.code, num_classes=1000).to(device)
            elif args.arch == 'resnet':
                model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=best_honey.code).to(device)
            elif args.arch == 'googlenet':
                pass
            elif args.arch == 'densenet':
                pass

            # 记录剪枝编码
            code = best_honey.code

            # 加载模型权重
            if args.best_honey_s:
                # 如果指定了best_honey_s路径，从该检查点加载权重
                bestckpt = torch.load(args.best_honey_s)
                model.load_state_dict(bestckpt['state_dict'])
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
                code = best_honey.code
                start_epoch = args.calfitness_epoch  # 从适应度计算的epoch开始继续训练

        # 子模式2.2：从检查点恢复训练
        else:
            # 加载检查点
            resumeckpt = torch.load(args.resume)
            state_dict = resumeckpt['state_dict']  # 模型权重

            # 获取剪枝编码（优先使用best_honey_past参数）
            if args.best_honey_past == None:
                code = resumeckpt['honey_code']  # 从检查点恢复剪枝编码
            else:
                code = args.best_honey_past      # 使用用户指定的剪枝编码

            # 根据剪枝编码构建模型
            print('==> Building model..')
            if args.arch == 'vgg':
                model = import_module(f'model.{args.arch}').BeeVGG(honeysource=code, num_classes=1000).to(device)
            elif args.arch == 'resnet':
                model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=code).to(device)
            elif args.arch == 'googlenet':
                pass
            elif args.arch == 'densenet':
                pass

            # 创建优化器
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            # 从检查点恢复模型、优化器状态
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(resumeckpt['optimizer'])
            start_epoch = resumeckpt['epoch']  # 恢复训练起始epoch

            # 如果使用多GPU，启用数据并行
            if len(args.gpus) != 1:
                model = nn.DataParallel(model, device_ids=args.gpus)


    # ==================== 测试或训练模式 ====================
    # 如果只是测试模式，直接测试后退出
    if args.test_only:
        test(model, testLoader, topk=(1, 5))

    else:
        # 训练模式：从start_epoch开始训练到num_epochs
        for epoch in range(start_epoch, args.num_epochs):
            # 训练一个epoch
            train(model, optimizer, trainLoader, args, epoch, topk=(1, 5))

            # 在测试集上评估（返回Top-5和Top-1准确率）
            test_acc, test_acc_top1 = test(model, testLoader, topk=(1, 5))

            # 判断是否是最优模型（基于Top-5准确率）
            is_best = best_acc < test_acc
            # 更新最优准确率
            best_acc_top1 = max(best_acc_top1, test_acc_top1)
            best_acc = max(best_acc, test_acc)

            # 获取模型权重（多GPU情况下需要访问module属性）
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

            # 构建检查点字典
            state = {
                'state_dict': model_state_dict,      # 模型权重
                'best_acc': best_acc,                # 最优Top-5准确率
                'optimizer': optimizer.state_dict(),  # 优化器状态
                'epoch': epoch + 1,                  # 下一个epoch编号
                'honey_code': code                   # 剪枝编码
            }

            # 保存检查点
            checkpoint.save_model(state, epoch + 1, is_best)

            # DALI需要在每个epoch结束时重置迭代器
            trainLoader.reset()
            testLoader.reset()

        # 训练结束，打印最优准确率
        logger.info('Best accurary(top5): {:.3f} (top1): {:.3f}'.format(float(best_acc), float(best_acc_top1)))


if __name__ == '__main__':
    main()
