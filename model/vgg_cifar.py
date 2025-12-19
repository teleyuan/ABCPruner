"""
VGG网络模型实现 - CIFAR数据集版本

支持VGG11/13/16/19多种配置
针对CIFAR10/100的32x32小图像优化
包含标准VGG和支持蜜蜂编码剪枝的BeeVGG两个版本
"""

import torch.nn as nn

# VGG网络配置字典
# 每个配置定义了网络中每层的通道数
# 'M'表示MaxPooling层，数字表示卷积层的输出通道数
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """
    标准VGG网络 - CIFAR版本

    与ImageNet版VGG的区别：
    - 输入图像较小（32x32）
    - 全连接层简化为单层（512 -> num_classes）
    - 使用AvgPool代替多层全连接

    参数:
        vgg_name: VGG配置名称（'vgg11', 'vgg13', 'vgg16', 'vgg19'）
        num_classes: 分类数量（CIFAR-10为10，CIFAR-100为100）
    """
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        # 构建卷积特征提取层
        self.features = self._make_layers(cfg[vgg_name])
        # 简化的分类器：直接从512维特征映射到类别
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        """前向传播"""
        out = self.features(x)             # 卷积特征提取
        out = out.view(out.size(0), -1)    # 展平
        out = self.classifier(out)         # 分类
        return out

    def _make_layers(self, cfg):
        """
        根据配置构建卷积层

        参数:
            cfg: 网络配置列表

        返回:
            nn.Sequential: 卷积层序列
        """
        layers = []
        in_channels = 3  # 输入通道数（RGB）
        for x in cfg:
            if x == 'M':
                # 添加最大池化层：2x2，stride=2
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 添加卷积块：Conv -> BN -> ReLU
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # CIFAR版特色：添加1x1平均池化（将特征图降到1x1）
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# honeysource: 1维向量，值限制在1-10（表示保留通道的比例）
class BeeVGG(nn.Module):
    """
    支持蜜蜂编码剪枝的VGG网络 - CIFAR版本

    每个卷积层的通道数根据honeysource进行剪枝：
    - honeysource[i] = 10: 保留100%通道
    - honeysource[i] = 5: 保留50%通道
    - honeysource[i] = 1: 保留10%通道

    参数:
        vgg_name: VGG配置名称
        honeysource: 蜜蜂编码列表，长度等于卷积层数量
    """
    def __init__(self, vgg_name, honeysource):
        super(BeeVGG, self).__init__()
        self.honeysource = honeysource
        # 构建剪枝后的卷积层
        self.features = self._make_layers(cfg[vgg_name])
        # 分类器：输入维度根据最后一层的剪枝情况调整
        # 最后一层实际通道数 = 512 * honeysource[-1] / 10
        self.classifier = nn.Linear(int(512 * honeysource[len(honeysource)-1] / 10), 10)

    def forward(self, x):
        """前向传播"""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        """
        根据配置和蜜蜂编码构建剪枝后的卷积层

        处理逻辑：
        1. 遍历网络配置
        2. 遇到'M'添加MaxPooling
        3. 遇到数字，根据对应位置的honeysource值计算剪枝后的通道数

        参数:
            cfg: 网络配置列表

        返回:
            nn.Sequential: 剪枝后的卷积层序列
        """
        layers = []
        in_channels = 3  # 输入通道数
        index = 0        # 卷积层索引（未使用，可以删除）
        Mlayers = 0      # MaxPooling层计数

        for x_index, x in enumerate(cfg):
            if x == 'M':
                # 添加最大池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                # 计算剪枝后的通道数
                # x_index - Mlayers 得到实际的卷积层索引
                x = int(x * self.honeysource[x_index - Mlayers] / 10)
                # 确保至少保留1个通道
                if x == 0:
                    x = 1
                # 添加剪枝后的卷积块
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # 添加平均池化层
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
