"""
ImageNet数据集加载器 - 标准PyTorch版本

ImageNet (ILSVRC2012) 数据集：
- 训练集：约128万张图像，1000个类别
- 验证集：5万张图像
- 图像尺寸：可变，通常resize到224x224

使用torchvision的ImageFolder加载器，适合标准训练
对于大规模训练建议使用imagenet_dali.py（NVIDIA DALI加速版本）
"""

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch


class Data:
    """
    ImageNet数据集封装类 - 标准PyTorch DataLoader实现

    功能:
    1. 加载ImageNet训练集和验证集
    2. 应用数据增强（训练集）
    3. 数据标准化（使用ImageNet统计值）
    4. 创建DataLoader

    数据集目录结构要求:
        data_path/
        ├── ILSVRC2012_img_train/    # 训练集（1000个子文件夹，每个代表一个类别）
        │   ├── n01440764/
        │   ├── n01443537/
        │   └── ...
        └── val/                      # 验证集（1000个子文件夹）
            ├── n01440764/
            ├── n01443537/
            └── ...

    需要的args参数:
        gpus: GPU设备列表（用于决定是否启用pin_memory）
        data_path: ImageNet数据集根目录
        train_batch_size: 训练批次大小
        eval_batch_size: 评估批次大小
    """
    def __init__(self, args):
        # 根据是否使用GPU决定是否启用pin_memory
        # pin_memory可以加速CPU到GPU的数据传输
        pin_memory = False
        if args.gpus is not None:
            pin_memory = True

        # 图像缩放尺寸
        # 如果使用Inception模型需要299x299，其他模型使用224x224
        # scale_size = 299 if args.student_model.startswith('inception') else 224
        scale_size = 224  # 默认使用224x224

        # 训练集和验证集目录路径
        traindir = os.path.join(args.data_path, 'ILSVRC2012_img_train')
        valdir = os.path.join(args.data_path, 'val')

        # ImageNet标准归一化参数
        # 这些均值和标准差是在整个ImageNet数据集上统计得到的
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # RGB三个通道的均值
            std=[0.229, 0.224, 0.225])   # RGB三个通道的标准差

        # ==================== 训练集数据预处理 ====================
        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                # 随机裁剪并resize到224x224
                # 裁剪区域占原图的比例随机（增加多样性）
                transforms.RandomResizedCrop(224),
                # 随机水平翻转（50%概率）
                transforms.RandomHorizontalFlip(),
                # Resize到指定尺寸（通常与crop尺寸相同）
                transforms.Resize(scale_size),
                # 转换为Tensor：[0,255] uint8 -> [0,1] float32
                transforms.ToTensor(),
                # 标准化：(x - mean) / std
                normalize,
            ]))

        # 创建训练集DataLoader
        self.loader_train = DataLoader(
            trainset,
            batch_size=args.train_batch_size,  # 批次大小（通常256或更大）
            shuffle=True,                       # 随机打乱（训练集需要打乱）
            num_workers=args.num_workers,                      # 数据加载的并行worker数量
            pin_memory=pin_memory)              # 是否将数据加载到CUDA固定内存

        # ==================== 验证集数据预处理 ====================
        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                # 先resize到256x256（保持长宽比，短边为256）
                transforms.Resize(256),
                # 中心裁剪到224x224
                transforms.CenterCrop(224),
                # Resize到指定尺寸
                transforms.Resize(scale_size),
                # 转换为Tensor
                transforms.ToTensor(),
                # 标准化
                normalize,
            ]))

        # 创建验证集DataLoader
        self.loader_test = DataLoader(
            testset,
            batch_size=args.eval_batch_size,  # 评估批次大小（通常可以更大）
            shuffle=False,                     # 验证集不需要打乱
            num_workers=args.num_workers,                     # 并行worker数量
            pin_memory=True)                   # 启用pin_memory加速
