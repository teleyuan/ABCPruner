"""
DenseNet模型实现 - CIFAR数据集版本

Dense Convolutional Network (DenseNet)特点：
1. 密集连接：每层都与之前所有层连接
2. 特征重用：通过concatenation共享特征
3. 缓解梯度消失：短路径连接促进梯度传播
4. 参数高效：通过特征共享减少参数量

支持蜜蜂编码进行通道剪枝
针对CIFAR10/100的32x32小图像优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

norm_mean, norm_var = 0.0, 1.0  # 归一化参数（未使用）

class DenseBasicBlock(nn.Module):
    """
    DenseNet基本块

    结构: BN -> ReLU -> Conv3x3
    特点: 输出通道数固定为growthRate，通过concatenation累积特征

    与ResNet的区别：
    - ResNet使用加法 (out = x + F(x))
    - DenseNet使用拼接 (out = [x, F(x)])

    参数:
        inplanes: 输入通道数（用于BN）
        filters: 实际参与卷积的通道数（可能经过剪枝）
        index: 当前块的索引（未使用）
        expansion: 扩展系数（默认1，未使用）
        growthRate: 增长率，每个块输出的通道数（可根据蜜蜂编码剪枝）
        dropRate: Dropout比率（默认0，不使用Dropout）
    """
    def __init__(self, inplanes, filters, index, expansion=1, growthRate=12, dropRate=0):
        super(DenseBasicBlock, self).__init__()
        planes = expansion * growthRate

        # BN层：对输入特征进行批归一化
        self.bn1 = nn.BatchNorm2d(inplanes)
        # 3x3卷积：输入filters通道，输出growthRate通道
        self.conv1 = nn.Conv2d(filters, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        """
        前向传播

        输入: [B, inplanes, H, W]
        输出: [B, inplanes+growthRate, H, W] (通过concatenation)
        """
        # BN -> ReLU -> Conv
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        # 可选的Dropout
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        # DenseNet核心：将新特征与输入拼接（而非相加）
        out = torch.cat((x, out), 1)  # 在通道维度拼接
        return out

class Transition(nn.Module):
    """
    Transition层 - 连接相邻DenseBlock的过渡层

    功能:
    1. 降维：通过1x1卷积压缩通道数（通常压缩到一半）
    2. 下采样：通过平均池化降低空间分辨率

    参数:
        inplanes: 输入通道数
        outplanes: 输出通道数（通常为inplanes的一半）
        filters: 实际参与卷积的通道数
        index: 当前transition的索引（未使用）
    """
    def __init__(self, inplanes, outplanes, filters, index):
        super(Transition, self).__init__()
        # BN层
        self.bn1 = nn.BatchNorm2d(inplanes)
        # 1x1卷积：降维
        self.conv1 = nn.Conv2d(filters, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播

        输入: [B, inplanes, H, W]
        输出: [B, outplanes, H/2, W/2]
        """
        # BN -> ReLU -> Conv -> AvgPool
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)  # 2x2平均池化，下采样2倍
        return out


class DenseNet(nn.Module):
    """
    DenseNet主网络 - CIFAR版本

    网络结构：
    - 初始3x3卷积（16通道）
    - 3个DenseBlock，每个包含n个DenseBasicBlock
    - 2个Transition层（连接DenseBlock）
    - 全局平均池化 + 全连接

    DenseNet-40配置（默认）：
    - depth = 40: 总层数
    - n = (40-4)//3 = 12: 每个DenseBlock包含12个block
    - growthRate = 12: 每个block输出12个通道
    - compressionRate = 2: Transition层压缩率（通道数减半）

    参数:
        depth: 网络深度（总层数），默认40
        block: 使用的块类型（DenseBasicBlock）
        dropRate: Dropout比率
        num_classes: 分类数量
        growthRate: 增长率k，每个block增加的通道数
        compressionRate: 压缩率，Transition层的压缩比例
        filters: 每层的实际卷积通道数配置（考虑剪枝）
        honey: 蜜蜂编码列表，控制每个block的growthRate
        indexes: 每层的通道索引（用于剪枝）
    """

    def __init__(self, depth=40, block=DenseBasicBlock,
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2, filters=None, honey=None, indexes=None):
        super(DenseNet, self).__init__()

        # 验证网络深度：DenseNet要求depth满足 depth = 3n + 4
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        # 计算每个DenseBlock的block数量
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6

        # 初始化蜜蜂编码
        if honey is None:
            self.honey = [10] * 36  # DenseNet-40有36个可剪枝位置
        else:
            self.honey = honey

        # 如果没有提供filters配置，根据蜜蜂编码自动计算
        if filters == None:
            filters = []
            start = growthRate * 2  # 初始通道数：24 (12*2)
            index = 0

            # 为3个DenseBlock计算filters
            for i in range(3):
                index -= 1
                filter = 0
                # 每个DenseBlock包含n+1层（n个block + 1个transition）
                for j in range(n+1):
                    if j != 0:
                        # 累加根据蜜蜂编码剪枝后的growthRate
                        filter += int(growthRate * self.honey[index] / 10)
                    # 记录当前层的总通道数
                    filters.append([start + filter])
                    index += 1
                # Transition层压缩通道数
                start = (start + int(growthRate * self.honey[index-1] / 10) * n) // compressionRate

            # 展平filters列表
            filters = [item for sub_list in filters for item in sub_list]

            # 生成每层的通道索引
            indexes = []
            for f in filters:
                indexes.append(np.arange(f))

        self.growthRate = growthRate
        self.currentindex = 0  # 当前处理的block索引
        self.dropRate = dropRate
        self.inplanes = growthRate * 2  # 初始通道数：24

        # 初始3x3卷积：3 -> 24通道
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)

        # 构建3个DenseBlock和2个Transition
        # DenseBlock1: 32x32, 24 -> 24+12*n 通道
        self.dense1 = self._make_denseblock(block, n, filters[0:n], indexes[0:n])
        # Transition1: 32x32 -> 16x16, 通道数减半
        self.trans1 = self._make_transition(Transition, filters[n+1], filters[n], indexes[n])

        # DenseBlock2: 16x16
        self.dense2 = self._make_denseblock(block, n, filters[n+1:2*n+1], indexes[n+1:2*n+1])
        # Transition2: 16x16 -> 8x8
        self.trans2 = self._make_transition(Transition, filters[2*n+2], filters[2*n+1], indexes[2*n+1])

        # DenseBlock3: 8x8
        self.dense3 = self._make_denseblock(block, n, filters[2*n+2:3*n+2], indexes[2*n+2:3*n+2])

        # 最后的BN和ReLU
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 全局平均池化：8x8 -> 1x1
        self.avgpool = nn.AvgPool2d(8)
        # 全连接层：分类
        self.fc = nn.Linear(self.inplanes, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, filters, indexes):
        """
        构建一个DenseBlock

        DenseBlock特点：
        - 每个block都与之前所有block连接
        - 通道数不断增加：start -> start+k -> start+2k -> ... -> start+n*k

        参数:
            block: DenseBasicBlock类
            blocks: block数量
            filters: 每个block的实际通道数配置
            indexes: 每个block的通道索引

        返回:
            nn.Sequential: DenseBlock序列
        """
        layers = []
        # 验证参数长度
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        assert blocks == len(indexes), 'Length of the indexes parameter is not right.'

        for i in range(blocks):
            # 根据蜜蜂编码计算当前block的growthRate（可能被剪枝）
            self.growthRate = int(12 * self.honey[self.currentindex] / 10)
            self.currentindex += 1

            # 更新输入通道数（DenseNet的核心：通道数不断累积）
            self.inplanes = filters[i]

            # 添加DenseBasicBlock
            layers.append(block(self.inplanes, filters=filters[i], index=indexes[i],
                               growthRate=self.growthRate, dropRate=self.dropRate))

        # 更新输出通道数（加上最后一个block的输出）
        self.inplanes += self.growthRate

        return nn.Sequential(*layers)


    def _make_transition(self, transition, compressionRate, filters, index):
        """
        构建Transition层

        参数:
            transition: Transition类
            compressionRate: 输出通道数（压缩后）
            filters: 输入通道数配置
            index: 通道索引

        返回:
            Transition层实例
        """
        inplanes = self.inplanes      # 输入通道数
        outplanes = compressionRate   # 输出通道数
        self.inplanes = outplanes     # 更新当前通道数
        return transition(inplanes, outplanes, filters, index)


    def forward(self, x):
        """
        前向传播

        输入: [B, 3, 32, 32]
        输出: [B, num_classes]
        """
        # 初始卷积：32x32x3 -> 32x32x24
        x = self.conv1(x)

        # 第一个DenseBlock：32x32
        x = self.dense1(x)
        # Transition1：32x32 -> 16x16，通道数减半
        x = self.trans1(x)

        # 第二个DenseBlock：16x16
        x = self.dense2(x)
        # Transition2：16x16 -> 8x8，通道数减半
        x = self.trans2(x)

        # 第三个DenseBlock：8x8
        x = self.dense3(x)

        # 最后的BN和ReLU
        x = self.bn(x)
        x = self.relu(x)

        # 全局平均池化：8x8 -> 1x1
        x = self.avgpool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 分类
        x = self.fc(x)

        return x

def densenet(honey=None, **kwargs):
    """
    DenseNet工厂函数

    默认创建DenseNet-40：
    - depth = 40
    - growthRate = 12
    - compressionRate = 1 (不压缩，用于CIFAR)

    参数:
        honey: 蜜蜂编码列表（长度36），如果为None则使用默认配置
        **kwargs: 其他参数

    返回:
        DenseNet模型实例
    """
    return DenseNet(depth=40, block=DenseBasicBlock, compressionRate=1, honey=honey, **kwargs)

# 测试代码
def test():
    honey = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]
    model = densenet(honey=None)
    # 打印模型参数
    print('Model.state_dict:')
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

#test()
