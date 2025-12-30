import torch
import torch.nn as nn
import random
import heapq
import copy
from model.googlenet import Inception


# ==================== 权重继承函数 ====================
def load_vgg_honey_model(model, args, oristate_dict):
    """
    为剪枝后的VGG模型加载预训练权重

    根据剪枝后的网络结构，从原始模型中选择性地继承权重。
    支持两种选择策略：
    1. random_pretrain: 随机选择要保留的通道
    2. l1_pretrain: 根据L1范数选择重要的通道

    参数:
        model: 剪枝后的模型
        args: 命令行参数对象
        oristate_dict: 原始模型的状态字典
    """
    state_dict = model.state_dict()
    last_select_index = None  # 记录上一层选择的通道索引，用于处理层间依赖

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            oriweight = oristate_dict[name + '.weight']  # 原始权重
            curweight = state_dict[name + '.weight']     # 当前（剪枝后）权重
            orifilter_num = oriweight.size(0)            # 原始通道数
            currentfilter_num = curweight.size(0)        # 剪枝后通道数

            # 如果通道数不同，需要选择性继承权重
            if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
                select_num = currentfilter_num

                # 选择要保留的通道索引
                if args.random_rule == 'random_pretrain':
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

def load_dense_honey_model(model, args, oristate_dict):
    """
    为剪枝后的DenseNet模型加载预训练权重

    DenseNet的特殊之处在于每层都与之前所有层连接，需要特殊处理

    参数:
        model: 剪枝后的DenseNet模型
        args: 命令行参数对象
        oristate_dict: 原始模型的状态字典
    """
    

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
        if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
            if args.random_rule == 'random_pretrain':
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
        if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
            if args.random_rule == 'random_pretrain':
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

    if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
        if args.random_rule == 'random_pretrain':
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

def load_google_honey_model(model, args, oristate_dict):
    """
    为剪枝后的GoogLeNet模型加载预训练权重

    GoogLeNet使用Inception模块，需要特殊处理多分支结构

    参数:
        model: 剪枝后的GoogLeNet模型
        args: 命令行参数对象
        oristate_dict: 原始模型的状态字典
    """
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

                if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if args.random_rule == 'random_pretrain':
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

                if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if args.random_rule == 'random_pretrain':
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
                if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if args.random_rule == 'random_pretrain':
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
                if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if args.random_rule == 'random_pretrain':
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

def load_resnet_honey_model(model, args, oristate_dict):
    """
    为剪枝后的ResNet模型加载预训练权重

    ResNet使用残差连接，需要特别注意shortcut连接的处理

    参数:
        model: 剪枝后的ResNet模型
        args: 命令行参数对象
        oristate_dict: 原始模型的状态字典
    """
    cfg = {
           'resnet56': [9,9,9],     # ResNet56: 3个stage，每个stage 9个block
           'resnet110': [18,18,18],  # ResNet110: 3个stage，每个stage 18个block
           }

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
                if orifilter_num != currentfilter_num and (args.random_rule == 'random_pretrain' or args.random_rule == 'l1_pretrain'):
                    select_num = currentfilter_num
                    if args.random_rule == 'random_pretrain':
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
                        #logger.info('last_select_index'.format(last_select_index))
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
