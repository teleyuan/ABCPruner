"""
ResNet模型实现 - CIFAR数据集版本

支持ResNet56和ResNet110
针对CIFAR10/100的32x32小图像优化
使用零填充shortcut而非1x1卷积，减少参数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


norm_mean, norm_var = 0.0, 1.0  # 归一化参数（未使用）

def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3卷积层的便捷函数

    参数:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 卷积步长

    返回:
        3x3卷积层（padding=1，无bias）
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class LambdaLayer(nn.Module):
    """
    Lambda层：包装任意函数为PyTorch模块

    用于实现CIFAR-ResNet的零填充shortcut
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd  # 保存lambda函数

    def forward(self, x):
        return self.lambd(x)  # 执行lambda函数

class ResBasicBlock(nn.Module):
    """
    CIFAR版ResNet基本块

    与标准ResNet不同：
    1. 使用零填充shortcut而非1x1卷积（节省参数）
    2. 针对32x32小图像设计
    3. 支持蜜蜂编码进行通道剪枝

    参数:
        inplanes: 输入通道数
        planes: 输出通道数（基准）
        honey: 蜜蜂编码列表
        index: 当前块在honey中的索引
        stride: 卷积步长（用于下采样）
    """
    expansion = 1  # 输出通道扩展系数

    def __init__(self, inplanes, planes, honey, index, stride=1):
        super(ResBasicBlock, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        # 根据蜜蜂编码计算中间层通道数（剪枝）
        middle_planes = int(planes * honey[index] / 10)

        # 第一个3x3卷积：可能包含下采样
        self.conv1 = conv3x3(inplanes, middle_planes, stride)
        self.bn1 = nn.BatchNorm2d(middle_planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个3x3卷积：恢复到标准通道数
        self.conv2 = conv3x3(middle_planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.stride = stride

        # Shortcut连接
        self.shortcut = nn.Sequential()
        # CIFAR-ResNet特色：使用零填充代替1x1卷积
        if stride != 1 or inplanes != planes:
            # 使用零填充增加通道数：[B, C, H, W] -> [B, 2C, H/2, W/2]
            # ::2表示每隔一个像素采样（实现下采样）
            # planes//4表示在通道维度两侧各填充planes//4个零通道
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x):
        # 主路径：conv1 -> bn1 -> relu -> conv2 -> bn2
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：主路径 + shortcut
        out += self.shortcut(x)
        out = self.relu(out)

        return out





class ResNet(nn.Module):
    """
    CIFAR版ResNet主网络类

    与标准ResNet的区别：
    1. 初始层使用3x3卷积（无池化），保持32x32分辨率
    2. 只有3个stage（不是4个）
    3. 使用零填充shortcut
    4. 全局平均池化后直接连接全连接层

    网络结构：
    - 初始3x3卷积（16通道）
    - Stage 1: n个残差块，16通道
    - Stage 2: n个残差块，32通道，下采样
    - Stage 3: n个残差块，64通道，下采样
    - 全局平均池化 + 全连接层

    参数:
        block: 使用的块类型（ResBasicBlock）
        num_layers: 网络总层数（如56, 110）
        num_classes: 分类数量（CIFAR-10为10，CIFAR-100为100）
        honey: 蜜蜂编码列表
    """
    def __init__(self, block, num_layers, num_classes=10, honey=None):
        super(ResNet, self).__init__()
        # CIFAR-ResNet要求层数满足6n+2的形式
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        # 计算每个stage的块数量
        n = (num_layers - 2) // 6

        self.honey = honey
        self.current_conv = 0  # 当前处理的卷积层索引
        self.inplanes = 16     # 初始通道数（CIFAR用16而非64）

        # 初始卷积层：3x3，stride=1（保持分辨率32x32）
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 构建3个stage
        # Stage 1: 16通道，stride=1（保持32x32）
        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1)
        # Stage 2: 32通道，stride=2（下采样到16x16）
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2)
        # Stage 3: 64通道，stride=2（下采样到8x8）
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2)

        # 自适应全局平均池化：任意大小 -> 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # 权重初始化
        self.initialize()

    def initialize(self):
        """
        初始化网络权重

        卷积层使用Kaiming初始化，BN层权重初始化为1，偏置为0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化：适合ReLU激活
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        """
        构建一个stage（包含多个残差块）

        参数:
            block: 块类型（ResBasicBlock）
            planes: 输出通道数
            blocks: 块的数量
            stride: 第一个块的步长

        返回:
            nn.Sequential: 包含所有块的序列模块
        """
        layers = []
        # 第一个块：使用指定stride进行下采样（如果需要）
        layers.append(block(self.inplanes, planes,
                            honey=self.honey, index=self.current_conv, stride=stride))
        self.current_conv += 1  # 索引递增
        # 更新输入通道数
        self.inplanes = planes * block.expansion

        # 后续块：stride=1，不下采样
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                honey=self.honey, index=self.current_conv))
            self.current_conv += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 3个stage的残差块
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        # 全局平均池化：8x8 -> 1x1
        x = self.avgpool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 分类
        x = self.fc(x)

        return x

def resnet56(num_classes=10, honey=None):
    """
    创建ResNet56模型

    ResNet56 = 1个初始卷积 + 3*(9个block*2层卷积) + 1个全连接 = 56层
    """
    return ResNet(ResBasicBlock, 56, num_classes=num_classes, honey=honey)

def resnet110(num_classes=10, honey=None):
    """
    创建ResNet110模型

    ResNet110 = 1个初始卷积 + 3*(18个block*2层卷积) + 1个全连接 = 110层
    """
    return ResNet(ResBasicBlock, 110, num_classes=num_classes, honey=honey)


# 网络配置：记录每个CIFAR-ResNet的可剪枝卷积层数量
conv_num_cfg = {
    'resnet56' : 27,   # ResNet56: 3个stage * 9个block = 27个可剪枝位置
    'resnet110' : 54,  # ResNet110: 3个stage * 18个block = 54个可剪枝位置
    }

def resnet(cfg, honey=None, num_classes=10):
    """
    CIFAR-ResNet工厂函数

    参数:
        cfg: 网络配置名称（'resnet56' 或 'resnet110'）
        honey: 蜜蜂编码，如果为None则使用默认配置（所有层保留100%通道）
        num_classes: 分类数量（CIFAR-10为10，CIFAR-100为100）

    返回:
        ResNet模型实例
    """
    # 如果没有提供蜜蜂编码，使用默认值（所有层保留100%）
    if honey == None:
        honey = conv_num_cfg[cfg] * [10]  # 10表示保留100%通道

    # 根据配置创建对应的ResNet模型
    if cfg == 'resnet56':
        return resnet56(num_classes=num_classes, honey=honey)
    elif cfg == 'resnet110':
        return resnet110(num_classes=num_classes, honey=honey)


'''
# 测试代码
def test():
    honey = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]  # 示例蜜蜂编码
    model = resnet('resnet56', honey)
    print(model)

test()
'''
