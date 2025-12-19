"""
ResNet模型实现 - ImageNet版本

支持ResNet18/34/50/101/152等多种配置
用于ImageNet数据集（224x224输入）
支持蜜蜂编码进行通道剪枝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 网络配置：记录每个ResNet变体的可剪枝卷积层数量
conv_num_cfg = {
    'resnet18' : 8,     # ResNet18: 8个残差块（每块1个可剪枝卷积）
    'resnet34' : 16,    # ResNet34: 16个残差块
    'resnet50' : 16,    # ResNet50: 16个bottleneck块（每块1个可剪枝位置）
    'resnet101' : 33,   # ResNet101: 33个bottleneck块
    'resnet152' : 50    # ResNet152: 50个bottleneck块
}

class BasicBlock(nn.Module):
    """
    ResNet基本块 - 用于ResNet18和ResNet34

    结构: 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> (+shortcut) -> ReLU

    参数:
        in_planes: 输入通道数
        planes: 输出通道数（基准）
        honey: 蜜蜂编码列表
        index: 当前块在honey编码中的索引
        stride: 卷积步长（用于下采样）
    """
    expansion = 1  # 输出通道扩展系数（BasicBlock不扩展）

    def __init__(self, in_planes, planes, honey, index, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积：根据蜜蜂编码调整输出通道数
        # planes * honey[index] / 10 实现通道剪枝（honey值范围1-10）
        self.conv1 = nn.Conv2d(in_planes, int(planes * honey[index] / 10),
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes * honey[index] / 10))

        # 第二个卷积：恢复到标准输出通道数
        self.conv2 = nn.Conv2d(int(planes * honey[index] / 10), planes,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut连接（残差连接）
        self.downsample = nn.Sequential()
        # 如果输入输出维度不同，需要用1x1卷积调整shortcut
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # 主路径：conv1 -> bn1 -> relu -> conv2 -> bn2
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差连接：主路径 + shortcut
        out += self.downsample(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    """
    ResNet Bottleneck块 - 用于ResNet50/101/152

    结构: 1x1 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv -> BN -> (+shortcut) -> ReLU
    使用1x1卷积降维和升维，减少计算量

    参数:
        in_planes: 输入通道数
        planes: 中间通道数（基准）
        honey: 蜜蜂编码列表
        index: 当前块在honey编码中的索引
        stride: 3x3卷积的步长
    """
    expansion = 4  # 输出通道扩展系数（最终输出是planes的4倍）

    def __init__(self, in_planes, planes, honey, index, stride=1):
        super(Bottleneck, self).__init__()
        # 根据蜜蜂编码计算剪枝后的通道数
        pr_channels = int(planes * honey[index] / 10)

        # 第一个1x1卷积：降维到pr_channels
        self.conv1 = nn.Conv2d(in_planes, pr_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pr_channels)

        # 3x3卷积：在低维空间进行特征提取
        self.conv2 = nn.Conv2d(pr_channels, pr_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(pr_channels)

        # 第二个1x1卷积：升维到planes*4
        self.conv3 = nn.Conv2d(pr_channels, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # Shortcut连接
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # 主路径：1x1降维 -> 3x3卷积 -> 1x1升维
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 残差连接
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet主网络类

    标准ResNet结构用于ImageNet:
    - 初始7x7卷积 + maxpool
    - 4个stage，每个stage包含多个残差块
    - 全局平均池化 + 全连接层

    参数:
        block: 使用的块类型（BasicBlock或Bottleneck）
        num_blocks: 每个stage的块数量列表，如[3,4,6,3]
        num_classes: 分类数量（ImageNet默认1000）
        honey: 蜜蜂编码列表，控制每个块的通道剪枝率
    """
    def __init__(self, block, num_blocks, num_classes=10, honey=None):
        super(ResNet, self).__init__()
        self.in_planes = 64  # 初始通道数
        self.honey = honey   # 蜜蜂编码
        self.current_conv = 0  # 当前处理的卷积层索引

        # 初始卷积层：7x7卷积，stride=2，输出64通道
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 最大池化层：进一步下采样
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建4个stage
        # Stage 1: 输出通道64，不下采样
        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        # Stage 2: 输出通道128，下采样2倍
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        # Stage 3: 输出通道256，下采样2倍
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        # Stage 4: 输出通道512，下采样2倍
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)

        # 全局平均池化：7x7 -> 1x1
        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        # 全连接层：分类
        self.fc = nn.Linear(512*block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化（He初始化）：适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layers(self, block, planes, num_blocks, stride):
        """
        构建一个stage（包含多个残差块）

        参数:
            block: 块类型
            planes: 输出通道数（基准）
            num_blocks: 块的数量
            stride: 第一个块的步长（用于下采样）

        返回:
            nn.Sequential: 包含所有块的序列模块
        """
        # 第一个块使用指定的stride进行下采样，其余块stride=1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # 创建残差块，传入当前的蜜蜂编码索引
            layers.append(block(self.in_planes, planes,
                self.honey, self.current_conv, stride))
            self.current_conv += 1  # 索引递增
            # 更新输入通道数为当前块的输出通道数
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        # 初始卷积和池化
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        # 4个stage的残差块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 全局平均池化
        out = self.avgpool(out)
        # 展平
        out = out.view(out.size(0), -1)
        # 分类
        out = self.fc(out)
        return out

def resnet(cfg, honey=None, num_classes=1000):
    """
    ResNet工厂函数

    参数:
        cfg: 网络配置名称（'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'）
        honey: 蜜蜂编码，如果为None则使用默认配置（所有层保留100%通道）
        num_classes: 分类数量

    返回:
        ResNet模型实例
    """
    # 如果没有提供蜜蜂编码，使用默认值（所有层保留100%）
    if honey == None:
        honey = conv_num_cfg[cfg] * [10]  # 10表示保留100%通道

    # 根据配置创建对应的ResNet模型
    if cfg == 'resnet18':
        return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet34':
        return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet50':
        return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet101':
        return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet152':
        return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, honey=honey)


# ==================== 快捷创建函数 ====================

def ResNet18():
    """创建标准ResNet18"""
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    """创建标准ResNet34"""
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    """创建标准ResNet50"""
    return ResNet(Bottleneck, [3,4,6,3], num_classes=1000, honey=None)

def ResNet101():
    """创建标准ResNet101"""
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    """创建标准ResNet152"""
    return ResNet(Bottleneck, [3,8,36,3])

'''
# 测试代码
def test():
    honey = [5,6,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6]  # 示例蜜蜂编码
    model = resnet('resnet50', honey)
    #print(model)

test()
'''
