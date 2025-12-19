"""
CIFAR-10数据集加载器

CIFAR-10数据集包含60000张32x32彩色图像，分为10个类别，
每个类别6000张图像。训练集50000张，测试集10000张。
"""

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Data:
    """
    CIFAR-10数据集封装类

    功能:
    1. 自动下载CIFAR-10数据集
    2. 应用数据增强(训练集)
    3. 数据标准化
    4. 创建DataLoader

    需要的args参数:
        gpus: GPU设备列表
        data_path: 数据集存储路径
        train_batch_size: 训练批次大小
        eval_batch_size: 评估批次大小
    """
    def __init__(self, args):
        # 如果使用GPU，启用pin_memory以加速数据传输
        if args.gpus is not None:
            pin_memory = True

        # 训练集数据增强
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪，padding=4增加多样性
            transforms.RandomHorizontalFlip(),     # 随机水平翻转
            transforms.ToTensor(),                  # 转换为Tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化(CIFAR10的均值和标准差)
        ])

        # 测试集数据预处理(不进行数据增强)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 加载训练集
        trainset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
        self.trainLoader = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=2, pin_memory=pin_memory
        )

        # 加载测试集
        testset = CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
        self.testLoader = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory)