"""
GoogLeNet (Inception v1) 模型实现 - CIFAR数据集版本

使用Inception模块实现多尺度特征提取
支持蜜蜂编码进行通道剪枝
针对CIFAR10/100的32x32小图像优化
"""

import torch
import torch.nn as nn

norm_mean, norm_var = 0.0, 1.0  # 归一化参数（未使用）

# 卷积层配置（未使用，可能用于调试）
cov_cfg = [(22*i+1) for i in range(1+2+5+2)]


class Inception(nn.Module):
    """
    Inception模块 - GoogLeNet的核心组件

    Inception模块特点：
    1. 并行使用多种尺寸的卷积核（1x1, 3x3, 5x5）
    2. 通过1x1卷积降维，减少计算量
    3. 包含池化分支，增加特征多样性

    结构：
        分支1: 1x1 conv
        分支2: 1x1 conv -> 3x3 conv
        分支3: 1x1 conv -> 5x5 conv (实现为两个3x3)
        分支4: 3x3 maxpool -> 1x1 conv
        输出: 将四个分支在通道维度concat

    参数:
        in_planes: 输入通道数
        n1x1: 分支1的1x1卷积输出通道数
        n3x3red: 分支2的降维1x1卷积输出通道数
        n3x3: 分支2的3x3卷积输出通道数
        n5x5red: 分支3的降维1x1卷积输出通道数（可剪枝）
        n5x5: 分支3的5x5卷积输出通道数（可剪枝）
        pool_planes: 分支4的1x1卷积输出通道数
        honey_rate: 蜜蜂编码值（1-10），控制分支3的剪枝率
        tmp_name: 模块名称（用于调试）
    """
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, honey_rate, tmp_name):
        super(Inception, self).__init__()
        self.honey_rate = honey_rate  # 剪枝率
        self.tmp_name = tmp_name      # 模块名称

        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # ==================== 分支1: 1x1卷积 ====================
        if self.n1x1:
            conv1x1 = nn.Conv2d(in_planes, n1x1, kernel_size=1)
            conv1x1.tmp_name = self.tmp_name  # 保存名称用于调试

            self.branch1x1 = nn.Sequential(
                conv1x1,
                nn.BatchNorm2d(n1x1),
                nn.ReLU(True),
            )

        # ==================== 分支2: 1x1降维 -> 3x3卷积 ====================
        if self.n3x3:
            # 第一步：1x1降维（通道数根据蜜蜂编码剪枝）
            conv3x3_1 = nn.Conv2d(in_planes, int(n3x3red * self.honey_rate / 10), kernel_size=1)
            # 第二步：3x3卷积（输出固定通道数n3x3）
            conv3x3_2 = nn.Conv2d(int(n3x3red * self.honey_rate / 10), n3x3, kernel_size=3, padding=1)
            conv3x3_1.tmp_name = self.tmp_name
            conv3x3_2.tmp_name = self.tmp_name

            self.branch3x3 = nn.Sequential(
                conv3x3_1,
                nn.BatchNorm2d(int(n3x3red * self.honey_rate / 10)),
                nn.ReLU(True),
                conv3x3_2,
                nn.BatchNorm2d(n3x3),
                nn.ReLU(True),
            )


        # ==================== 分支3: 1x1降维 -> 3x3 -> 3x3 (模拟5x5) ====================
        # 使用两个3x3卷积代替一个5x5卷积，减少参数量
        if self.n5x5 > 0:
            # 第一步：1x1降维（通道数根据蜜蜂编码剪枝）
            conv5x5_1 = nn.Conv2d(in_planes, int(n5x5red * self.honey_rate / 10), kernel_size=1)
            # 第二步：第一个3x3卷积（中间层通道数也根据蜜蜂编码剪枝）
            conv5x5_2 = nn.Conv2d(int(n5x5red * self.honey_rate / 10),
                                  int(n5x5 * self.honey_rate / 10), kernel_size=3, padding=1)
            # 第三步：第二个3x3卷积（输出固定通道数n5x5）
            conv5x5_3 = nn.Conv2d(int(n5x5 * self.honey_rate / 10), n5x5, kernel_size=3, padding=1)
            conv5x5_1.tmp_name = self.tmp_name
            conv5x5_2.tmp_name = self.tmp_name
            conv5x5_3.tmp_name = self.tmp_name

            self.branch5x5 = nn.Sequential(
                conv5x5_1,
                nn.BatchNorm2d(int(n5x5red * self.honey_rate / 10)),
                nn.ReLU(True),
                conv5x5_2,
                nn.BatchNorm2d(int(n5x5 * self.honey_rate / 10)),
                nn.ReLU(True),
                conv5x5_3,
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
            )

        # ==================== 分支4: 3x3 maxpool -> 1x1卷积 ====================
        if self.pool_planes > 0:
            conv_pool = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
            conv_pool.tmp_name = self.tmp_name

            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),  # 保持尺寸不变的maxpool
                conv_pool,
                nn.BatchNorm2d(pool_planes),
                nn.ReLU(True),
            )

    def forward(self, x):
        """
        前向传播：并行执行四个分支，然后在通道维度拼接

        输入: [B, in_planes, H, W]
        输出: [B, n1x1+n3x3+n5x5+pool_planes, H, W]
        """
        out = []
        # 分支1: 1x1卷积
        y1 = self.branch1x1(x)
        out.append(y1)

        # 分支2: 1x1 -> 3x3
        y2 = self.branch3x3(x)
        out.append(y2)

        # 分支3: 1x1 -> 3x3 -> 3x3
        y3 = self.branch5x5(x)
        out.append(y3)

        # 分支4: maxpool -> 1x1
        y4 = self.branch_pool(x)
        out.append(y4)

        # 在通道维度拼接所有分支的输出
        return torch.cat(out, 1)


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) 主网络

    网络结构（CIFAR版本）：
    - 初始卷积：3x3, 192通道
    - 9个Inception模块（分3组）
      * 组1: 2个模块（32x32）
      * 组2: 5个模块（16x16）
      * 组3: 2个模块（8x8）
    - 全局平均池化 + 全连接

    与ImageNet版GoogLeNet的区别：
    - 去掉了初始的7x7卷积和maxpool（针对小图像优化）
    - 去掉了辅助分类器（简化训练）

    参数:
        block: Inception模块类
        filters: 每个Inception模块的通道配置（如果为None则使用默认配置）
        honey: 蜜蜂编码列表，长度为9（对应9个Inception模块）
    """
    def __init__(self, block=Inception, filters=None, honey=None):
        super(GoogLeNet, self).__init__()
        self.covcfg = cov_cfg

        # 如果没有提供蜜蜂编码，使用默认值（所有模块保留100%通道）
        if honey is None:
            self.honey = [10] * 9  # 9个Inception模块
        else:
            self.honey = honey

        # 初始卷积层：3x3, 192通道
        conv_pre = nn.Conv2d(3, 192, kernel_size=3, padding=1)
        conv_pre.tmp_name = 'pre_layer'
        self.pre_layers = nn.Sequential(
            conv_pre,
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        # 默认的filters配置
        # 每个列表[n1x1, n3x3, n5x5, pool_planes]定义一个Inception模块的输出通道
        if filters is None:
            filters = [
                [64, 128, 32, 32],    # inception_a3: 总输出256通道
                [128, 192, 96, 64],   # inception_b3: 总输出480通道
                [192, 208, 48, 64],   # inception_a4: 总输出512通道
                [160, 224, 64, 64],   # inception_b4: 总输出512通道
                [128, 256, 64, 64],   # inception_c4: 总输出512通道
                [112, 288, 64, 64],   # inception_d4: 总输出528通道
                [256, 320, 128, 128], # inception_e4: 总输出832通道
                [256, 320, 128, 128], # inception_a5: 总输出832通道
                [384, 384, 128, 128]  # inception_b5: 总输出1024通道
            ]

        self.filters = filters

        # ==================== 第一组Inception模块（32x32分辨率）====================
        # Inception a3: 输入192, 输出256 (64+128+32+32)
        self.inception_a3 = block(192, filters[0][0], 96, filters[0][1], 16, filters[0][2], filters[0][3],
                                   self.honey[0], 'a3')
        # Inception b3: 输入256, 输出480 (128+192+96+64)
        self.inception_b3 = block(sum(filters[0]), filters[1][0], 128, filters[1][1], 32, filters[1][2],
                                   filters[1][3], self.honey[1], 'a4')

        # 第一次下采样：32x32 -> 16x16
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # ==================== 第二组Inception模块（16x16分辨率）====================
        # Inception a4: 输入480, 输出512
        self.inception_a4 = block(sum(filters[1]), filters[2][0], 96, filters[2][1], 16, filters[2][2],
                                   filters[2][3], self.honey[2], 'a4')
        # Inception b4: 输入512, 输出512
        self.inception_b4 = block(sum(filters[2]), filters[3][0], 112, filters[3][1], 24, filters[3][2],
                                   filters[3][3], self.honey[3], 'b4')
        # Inception c4: 输入512, 输出512
        self.inception_c4 = block(sum(filters[3]), filters[4][0], 128, filters[4][1], 24, filters[4][2],
                                   filters[4][3], self.honey[4], 'c4')
        # Inception d4: 输入512, 输出528
        self.inception_d4 = block(sum(filters[4]), filters[5][0], 144, filters[5][1], 32, filters[5][2],
                                   filters[5][3], self.honey[5], 'd4')
        # Inception e4: 输入528, 输出832
        self.inception_e4 = block(sum(filters[5]), filters[6][0], 160, filters[6][1], 32, filters[6][2],
                                   filters[6][3], self.honey[6], 'e4')

        # 第二次下采样：16x16 -> 8x8
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # ==================== 第三组Inception模块（8x8分辨率）====================
        # Inception a5: 输入832, 输出832
        self.inception_a5 = block(sum(filters[6]), filters[7][0], 160, filters[7][1], 32, filters[7][2],
                                   filters[7][3], self.honey[7], 'a5')
        # Inception b5: 输入832, 输出1024
        self.inception_b5 = block(sum(filters[7]), filters[8][0], 192, filters[8][1], 48, filters[8][2],
                                   filters[8][3], self.honey[8], 'b5')

        # 全局平均池化：8x8 -> 1x1
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # 全连接层：1024 -> 10 (CIFAR-10)
        self.linear = nn.Linear(sum(filters[-1]), 10)

    def forward(self, x):
        """前向传播"""
        # 初始卷积：32x32x3 -> 32x32x192
        out = self.pre_layers(x)

        # 第一组Inception: 32x32
        out = self.inception_a3(out)  # 32x32x192 -> 32x32x256
        out = self.inception_b3(out)  # 32x32x256 -> 32x32x480
        out = self.maxpool1(out)      # 32x32x480 -> 16x16x480

        # 第二组Inception: 16x16
        out = self.inception_a4(out)  # 16x16x480 -> 16x16x512
        out = self.inception_b4(out)  # 16x16x512 -> 16x16x512
        out = self.inception_c4(out)  # 16x16x512 -> 16x16x512
        out = self.inception_d4(out)  # 16x16x512 -> 16x16x528
        out = self.inception_e4(out)  # 16x16x528 -> 16x16x832
        out = self.maxpool2(out)      # 16x16x832 -> 8x8x832

        # 第三组Inception: 8x8
        out = self.inception_a5(out)  # 8x8x832 -> 8x8x832
        out = self.inception_b5(out)  # 8x8x832 -> 8x8x1024

        # 全局平均池化和分类
        out = self.avgpool(out)       # 8x8x1024 -> 1x1x1024
        out = out.view(out.size(0), -1)  # 展平: [B, 1024]
        out = self.linear(out)        # 分类: [B, 1024] -> [B, 10]

        return out

def googlenet(honey=None):
    """
    GoogLeNet工厂函数

    参数:
        honey: 蜜蜂编码列表（长度为9），如果为None则使用默认配置

    返回:
        GoogLeNet模型实例
    """
    return GoogLeNet(block=Inception, honey=honey)

# 测试代码
def test():
    honey = [1,2,3,4,5,6,7,8,9]  # 示例蜜蜂编码
    model = googlenet(honey=honey)
    print(model)

#test()
