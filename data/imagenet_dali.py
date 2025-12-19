"""
ImageNet数据集加载器 - NVIDIA DALI加速版本

NVIDIA DALI (Data Loading Library) 特点：
1. GPU加速的数据预处理（解码、resize、crop等在GPU上完成）
2. 高度优化的数据管道，减少CPU瓶颈
3. 比标准PyTorch DataLoader快2-3倍
4. 适合大规模训练（如ImageNet）

使用场景：
- 当数据加载成为训练瓶颈时
- GPU利用率不高，CPU负载高时
- 需要加速ImageNet训练时

前置条件：
- 需要安装NVIDIA DALI：pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
- 需要CUDA支持
"""

import time
import torch.utils.data
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision.datasets as datasets
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator


class HybridTrainPipe(Pipeline):
    """
    DALI训练数据管道 - 混合CPU/GPU处理

    数据流程：
    1. CPU: 读取JPEG文件
    2. GPU: 解码JPEG
    3. GPU: 随机裁剪resize
    4. GPU: 归一化和随机水平翻转

    参数:
        batch_size: 批次大小
        num_threads: CPU线程数
        device_id: GPU设备ID
        data_dir: 训练数据目录
        crop: 裁剪后的图像尺寸（通常224）
        dali_cpu: 是否使用CPU模式（默认False，使用GPU）
        local_rank: 分布式训练的本地rank
        world_size: 分布式训练的总进程数
    """
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        # 初始化Pipeline，设置随机种子（seed = 12 + device_id 确保不同GPU使用不同随机序列）
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        dali_device = "gpu"  # 使用GPU进行数据预处理

        # 文件读取器：从磁盘读取JPEG文件
        # shard_id和num_shards用于分布式训练时数据分片
        self.input = ops.FileReader(
            file_root=data_dir,       # 数据根目录
            shard_id=local_rank,      # 当前进程的分片ID
            num_shards=world_size,    # 总分片数
            random_shuffle=True)      # 随机打乱文件顺序

        # 图像解码器：将JPEG解码为RGB图像
        # device="mixed" 表示在CPU读取后立即传输到GPU解码
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        # 随机裁剪并resize：模拟RandomResizedCrop
        # random_area: 裁剪区域占原图的比例范围 [0.08, 1.25]
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])

        # 归一化和镜像翻转操作
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,        # 输出float32类型
            output_layout=types.NCHW,        # 输出格式：[N, C, H, W]
            image_type=types.RGB,            # RGB图像
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],  # ImageNet均值（注意：DALI使用[0,255]范围）
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])   # ImageNet标准差

        # 随机翻转：生成0/1随机数，控制是否镜像
        self.coin = ops.CoinFlip(probability=0.5)  # 50%概率翻转

        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        """
        定义DALI数据处理图（计算图）

        这个方法定义了数据处理的流程，DALI会自动优化这个图的执行
        """
        # 生成随机翻转标志
        rng = self.coin()

        # 读取JPEG文件和标签
        self.jpegs, self.labels = self.input(name="Reader")

        # 解码JPEG -> RGB图像
        images = self.decode(self.jpegs)

        # 随机裁剪并resize
        images = self.res(images)

        # 归一化并随机水平翻转
        output = self.cmnp(images, mirror=rng)

        # 返回处理后的图像和标签
        return [output, self.labels]


class HybridValPipe(Pipeline):
    """
    DALI验证数据管道 - 混合CPU/GPU处理

    与训练管道的区别：
    1. 不使用随机裁剪，使用固定的resize和中心裁剪
    2. 不使用随机翻转
    3. 不打乱数据顺序

    数据流程：
    1. CPU: 读取JPEG文件
    2. GPU: 解码JPEG
    3. GPU: Resize（短边到256）
    4. GPU: 中心裁剪到224x224
    5. GPU: 归一化

    参数:
        batch_size: 批次大小
        num_threads: CPU线程数
        device_id: GPU设备ID
        data_dir: 验证数据目录
        crop: 裁剪后的图像尺寸（通常224）
        size: resize的目标尺寸（通常256）
        local_rank: 分布式训练的本地rank
        world_size: 分布式训练的总进程数
    """
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        # 文件读取器（不打乱顺序）
        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=local_rank,
            num_shards=world_size,
            random_shuffle=False)  # 验证集不打乱

        # 图像解码器
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        # Resize操作：保持长宽比，短边resize到size（256）
        self.res = ops.Resize(
            device="gpu",
            resize_shorter=size,           # 短边目标尺寸
            interp_type=types.INTERP_TRIANGULAR)  # 三角插值（双线性插值的变体）

        # 中心裁剪和归一化（不翻转）
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),              # 中心裁剪到(crop, crop)
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        """定义验证数据处理图"""
        # 读取JPEG和标签
        self.jpegs, self.labels = self.input(name="Reader")

        # 解码
        images = self.decode(self.jpegs)

        # Resize（短边到256）
        images = self.res(images)

        # 中心裁剪并归一化（不翻转）
        output = self.cmnp(images)

        return [output, self.labels]



def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                           world_size=1, local_rank=0):
    """
    创建DALI ImageNet数据迭代器

    参数:
        type: 'train' 或 'val'，指定训练集或验证集
        image_dir: ImageNet数据集根目录
        batch_size: 批次大小
        num_threads: CPU线程数（通常4）
        device_id: GPU设备ID
        num_gpus: GPU数量
        crop: 裁剪尺寸（通常224）
        val_size: 验证集resize尺寸（通常256）
        world_size: 分布式训练的总进程数
        local_rank: 当前进程的rank

    返回:
        DALIClassificationIterator: DALI数据迭代器

    使用示例:
        >>> train_loader = get_imagenet_iter_dali('train', '/data/imagenet', 256, 4, 0, 1, 224)
        >>> for batch_data in train_loader:
        >>>     images = batch_data[0]['data']
        >>>     labels = batch_data[0]['label']
    """
    if type == 'train':
        # 创建训练管道
        pip_train = HybridTrainPipe(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank,
            data_dir=image_dir + '/ILSVRC2012_img_train',  # 训练数据目录
            crop=crop,
            world_size=world_size,
            local_rank=local_rank)

        # 构建管道（编译和优化计算图）
        pip_train.build()

        # 创建DALI迭代器
        dali_iter_train = DALIClassificationIterator(
            pip_train,
            size=pip_train.epoch_size("Reader") // world_size)  # 每个进程的数据量

        return dali_iter_train

    elif type == 'val':
        # 创建验证管道
        pip_val = HybridValPipe(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank,
            data_dir=image_dir + '/val',  # 验证数据目录
            crop=crop,
            size=val_size,
            world_size=world_size,
            local_rank=local_rank)

        # 构建管道
        pip_val.build()

        # 创建DALI迭代器
        dali_iter_val = DALIClassificationIterator(
            pip_val,
            size=pip_val.epoch_size("Reader") // world_size)

        return dali_iter_val


def get_imagenet_iter_torch(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                            world_size=1, local_rank=0):
    """
    创建标准PyTorch ImageNet数据迭代器（用于对比）

    这是标准的PyTorch DataLoader实现，可以与DALI版本进行性能对比

    参数:
        同get_imagenet_iter_dali

    返回:
        torch.utils.data.DataLoader: PyTorch数据加载器
    """
    if type == 'train':
        # 训练集数据增强
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),  # 随机裁剪
            transforms.RandomHorizontalFlip(),                        # 随机翻转
            transforms.ToTensor(),                                    # 转Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],         # 归一化
                               std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/train', transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_threads,
            pin_memory=True)
    else:
        # 验证集数据预处理
        transform = transforms.Compose([
            transforms.Resize(val_size),              # Resize短边到256
            transforms.CenterCrop(crop),              # 中心裁剪到224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/val', transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_threads,
            pin_memory=True)
    return dataloader


# ==================== 性能测试代码 ====================
if __name__ == '__main__':
    # 测试DALI数据加载速度
    print('Testing DALI data loader...')
    train_loader = get_imagenet_iter_dali(
        type='train',
        image_dir='/userhome/memory_data/imagenet',
        batch_size=256,
        num_threads=4,
        crop=224,
        device_id=0,
        num_gpus=1)

    print('Start iterating with DALI...')
    start = time.time()
    for i, data in enumerate(train_loader):
        # DALI返回的数据格式：batch_data[0]['data']和batch_data[0]['label']
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    end = time.time()
    print('DALI iterate time: %fs' % (end - start))

    # 测试标准PyTorch数据加载速度
    print('\nTesting PyTorch data loader...')
    train_loader = get_imagenet_iter_torch(
        type='train',
        image_dir='/userhome/data/imagenet',
        batch_size=256,
        num_threads=4,
        crop=224,
        device_id=0,
        num_gpus=1)

    print('Start iterating with PyTorch...')
    start = time.time()
    for i, data in enumerate(train_loader):
        # PyTorch返回的数据格式：(images, labels)
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    print('PyTorch iterate time: %fs' % (end - start))

    print('\nNote: DALI is usually 2-3x faster than PyTorch DataLoader for ImageNet')
