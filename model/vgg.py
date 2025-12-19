"""
VGG网络模型实现 - ImageNet版本

VGG16网络用于ImageNet数据集（224x224输入）
包含标准VGG和支持蜜蜂编码剪枝的BeeVGG两个版本
"""

from collections import OrderedDict
import torch.nn as nn


# VGG16配置：定义每层的通道数
# 'M'表示MaxPooling层
# 数字表示卷积层的输出通道数
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    """
    标准VGG16网络 - ImageNet版本

    网络结构：
    - 13个卷积层（5组，通道数：64->128->256->512->512）
    - 5个最大池化层
    - 3个全连接层（4096->4096->num_classes）

    输入: 224x224x3
    输出: num_classes维向量

    参数:
        num_classes: 分类数量（ImageNet默认1000）
    """
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        # 卷积特征提取层
        self.features = self._make_layers(cfg)
        # 分类器（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 第一个全连接层：展平后的特征 -> 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),                  # Dropout正则化，防止过拟合
            nn.Linear(4096, 4096),         # 第二个全连接层：4096 -> 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),  # 输出层：4096 -> num_classes
        )

    def forward(self, x):
        """前向传播"""
        out = self.features(x)             # 卷积特征提取：224x224 -> 7x7
        out = out.view(out.size(0), -1)    # 展平：[B, 512, 7, 7] -> [B, 512*7*7]
        out = self.classifier(out)         # 全连接分类
        return out

    def _make_layers(self, cfg):
        """
        根据配置构建卷积层

        参数:
            cfg: 网络配置列表，如[64, 64, 'M', 128, ...]

        返回:
            nn.Sequential: 卷积层序列
        """
        layers = []
        in_channels = 3  # 输入图像的通道数（RGB）
        for x in cfg:
            if x == 'M':
                # 添加最大池化层：2x2，stride=2（下采样2倍）
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 添加卷积块：Conv -> BN -> ReLU
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),  # 3x3卷积，padding=1保持尺寸
                           nn.BatchNorm2d(x),                                     # 批归一化
                           nn.ReLU(inplace=True)]                                 # ReLU激活
                in_channels = x  # 更新输入通道数为当前层输出通道数
        return nn.Sequential(*layers)


class BeeVGG(nn.Module):
    """
    支持蜜蜂编码剪枝的VGG16网络 - ImageNet版本

    与标准VGG的区别：
    - 每个卷积层的通道数根据蜜蜂编码进行剪枝
    - honeysource[i]表示第i层保留的通道比例（1-10对应10%-100%）

    参数:
        honeysource: 蜜蜂编码列表，长度为13（对应13个卷积层）
        num_classes: 分类数量
    """
    def __init__(self, honeysource, num_classes=10):
        super(BeeVGG, self).__init__()
        self.honeysource = honeysource
        # 构建剪枝后的卷积层
        self.features = self._make_layers(cfg)
        # 分类器：第一层输入维度需要根据最后一个卷积层的剪枝情况调整
        # 最后一层的实际通道数 = 512 * honeysource[-1] / 10
        self.classifier = nn.Sequential(
            nn.Linear(int(512 * honeysource[len(honeysource)-1] / 10) * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """前向传播"""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        """
        根据配置和蜜蜂编码构建剪枝后的卷积层

        参数:
            cfg: 网络配置列表

        返回:
            nn.Sequential: 剪枝后的卷积层序列
        """
        layers = []
        in_channels = 3  # 输入通道数
        index = 0        # 当前处理的卷积层索引（用于访问honeysource）
        Mlayers = 0      # 已遇到的MaxPooling层数量（用于计算实际卷积层索引）

        for x_index, x in enumerate(cfg):
            if x == 'M':
                # 添加最大池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1  # MaxPooling计数+1
            else:
                # 根据蜜蜂编码计算剪枝后的通道数
                # x_index - Mlayers 得到实际的卷积层索引（去除MaxPooling的影响）
                x = int(x * self.honeysource[x_index - Mlayers] / 10)
                # 确保至少保留1个通道（避免层完全消失）
                if x == 0:
                    x = 1
                # 添加剪枝后的卷积块
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x  # 更新输入通道数
        return nn.Sequential(*layers)

'''
# 测试代码
def test():
    honey = [5,6,1,2,3,4,5,6,7,3,1,1,1]  # 示例蜜蜂编码（13个值对应13个卷积层）
    model = BeeVGG(honey)
    print(model)

test()
'''
