"""
CIFAR-100数据集加载器

CIFAR-100数据集包含60000张32x32彩色图像，分为100个类别，
每个类别600张图像。训练集50000张，测试集10000张。
100个类别被分组为20个超类。
"""

from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Data:
    """
    CIFAR-100数据集封装类

    功能与CIFAR-10类似，但类别数更多(100类)
    """
    def __init__(self, args):
        # 启用pin_memory以加速GPU数据传输
        pin_memory = True

        # 训练集数据增强
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),     # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 使用CIFAR标准化参数
        ])

        # 测试集数据预处理
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 加载CIFAR-100训练集
        trainset = CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
        self.trainLoader = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=2, pin_memory=pin_memory
        )

        # 加载CIFAR-100测试集
        testset = CIFAR100(root=args.data_path, train=False, download=False, transform=transform_test)
        self.testLoader = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory)
