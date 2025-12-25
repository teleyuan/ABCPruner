# Channel Pruning via Automatic Structure Search ([论文链接](https://arxiv.org/abs/2001.08565))

[![访问量](https://visitor-badge.glitch.me/badge?page_id=lmbxmu.abcpruner)](https://github.com/lmbxmu/ABCPruner)

**ABCPruner 的 PyTorch 实现 (IJCAI 2020)**

基于人工蜂群算法(ABC)的神经网络自动剪枝框架，可自动搜索最优的网络通道剪枝配置。

<div align=center><img src="https://raw.githubusercontent.com/zyxxmu/Images/master/ABCPruner/ABCPruner_framework.png"/></div>

---

## 📋 目录

- [项目简介](#项目简介)
- [实验结果](#实验结果)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [参数详解](#参数详解)
- [引用](#引用)
- [联系方式](#联系方式)

---

## 📖 项目简介

ABCPruner 是一个基于人工蜂群算法的神经网络剪枝工具，能够：

- ✅ **自动搜索**：使用ABC算法自动搜索最优剪枝配置，无需手动设计剪枝方案
- ✅ **多架构支持**：支持VGG、ResNet、GoogLeNet、DenseNet等主流网络架构
- ✅ **多数据集**：支持CIFAR-10、CIFAR-100、ImageNet数据集
- ✅ **高压缩率**：在保持精度的同时，可达到50%-70%的参数和FLOPs压缩率
- ✅ **易于使用**：提供完整的训练、测试和评估工具

---

## 🎯 实验结果

我们提供了论文中所有剪枝模型的下载链接、训练日志和配置文件。

*（括号中的百分比表示剪枝压缩率）*

### CIFAR-10 数据集

| 原始模型 | 参数量 | FLOPs | 通道数 | 准确率 | 剪枝模型下载 |
| ---------- | ------------- | -------------- | ------------ | -------- | ------------------------------------------------------------ |
| VGG16      | 1.67M(88.68%) | 82.81M(73.68%) | 1639(61.20%) | 93.08%   | [ABCPruner-80%](https://drive.google.com/drive/folders/19p0dqM4g_9ypQ_hgYIUkt7SUJI1w_u-T?usp=sharing) |
| ResNet56   | 0.39M(54.20%) | 58.54M(54.13%) | 1482(27.07%) | 93.23%   | [ABCPruner-70%](https://drive.google.com/drive/folders/1o3K_y7YFLRu7MSIEHV7kecHKIKm1fUqC?usp=sharing) |
| ResNet110  | 0.56M(67.41%) | 89.87M(65.04%) | 2701(33.28%) | 93.58%   | [ABCPruner-60%](https://drive.google.com/drive/folders/1WWVqLvLHgUmBpP3huYU_dpbFk5wPMmTV?usp=sharing) |
| GoogLeNet  | 2.46M(60.14%) | 513.19M(66.56) | 6150(22.19%) | 94.84%   | [ABCPruner-30%](https://drive.google.com/drive/folders/1vlOAwI_FrQeJU0ntsPQJyQt-mk26OTOc?usp=sharing) |

### ImageNet 数据集

| 原始模型 | 参数量 | FLOPs | 通道数 | Top-1 | Top-5 | 剪枝模型下载 |
| ---------- | -------------- | ---------------- | ------------- | -------- | -------- | ------------------------------------------------------------ |
| ResNet18   | 6.6M(43.55%)   | 1005.71M(44.88%) | 3894(18.88%)  | 67.28%   | 87.28%   | [ABCPruner-70%](https://drive.google.com/drive/folders/1ydTZ0VZTs5RKoVqRKX3oOo2zT27-ROGM?usp=sharing) |
| ResNet18   | 9.5M(18.72%)   | 968.13M(46.94%)  | 4220(12%)     | 67.80%   | 88.00%   | [ABCPruner-100%](https://drive.google.com/drive/folders/1vp65RN9hzveqpgsJWJ5kgHvo40tHTsY6?usp=sharing) |
| ResNet34   | 10.52M(51.76%) | 1509.76M(58.97%) | 5376(25.09%)  | 70.45%   | 89.688%  | [ABCPruner-50%](https://drive.google.com/drive/folders/1Nl1YVgwODzPmAalDgDp-qwhAhkkdRLR4?usp=sharing) |
| ResNet34   | 10.12M(53.58%) | 2170.77M(41%)    | 6655(21.82%)  | 70.98%   | 90.053%  | [ABCPruner-90%](https://drive.google.com/drive/folders/18g5spNsvL5fSHnIR9hvjk2vX53L2nD9A?usp=sharing) |
| ResNet50   | 7.35M(71.24%)  | 944.85M(68.68%)  | 20576(25.53%) | 70.289%  | 89.631%  | [ABCPruner-30%](https://drive.google.com/drive/folders/19qR4g5MRFCbmM7DMzLxUNRJifrhX-xgm?usp=sharing) |
| ResNet50   | 9.1M(64.38%)   | 1295.4M(68.68%)  | 21426(19.33%) | 72.582%  | 90.19%   | [ABCPruner-50%](https://drive.google.com/drive/folders/1LNUG0He2Idux7leL28i4pOYoWP31txsr?usp=sharing) |
| ResNet50   | 11.24M(56.01%) | 1794.45M(56.61%) | 22348(15.86%) | 73.516%  | 91.512%  | [ABCPruner-70%](https://drive.google.com/drive/folders/1GJ70Kcsf-ixc9sTIeTqmFlLDUE1zHiJK?usp=sharing) |
| ResNet50   | 11.75M(54.02%) | 1890.6M(54.29%)  | 22518(15.22%) | 73.864%  | 91.687%  | [ABCPruner-80%](https://drive.google.com/drive/folders/1Sbq1yv1BZHvx9ai57-_MO-v6pQeBjf2S?usp=sharing) |
| ResNet50   | 18.02M(29.5%)  | 2555.55M(38.21%) | 24040(9.5%)   | 74.843%  | 92.272%  | [ABCPruner-100%](https://drive.google.com/drive/folders/1Htt_wvgC1syCJQ-qjbgAEFdAOasGmVox?usp=sharing) |
| ResNet101  | 12.94M(70.94%) | 1975.61M(74.89%) | 41316(21.56%) | 74.683%  | 92.08%   | [ABCPruner-50%](https://drive.google.com/drive/folders/1ACxsGeW8YmCCFOG44cCq8t_mtMoqUvzt?usp=sharing) |
| ResNet101  | 17.72M(60.21%) | 3164.91M(59.78%) | 43168(17.19%) | 75.823%  | 92.736%  | [ABCPruner-80%](https://drive.google.com/drive/folders/1RJPjBsB1pKJE0NL8qGD718YfGtTfLO4z?usp=sharing) |
| ResNet152  | 15.62M(74.06%) | 2719.47M(76.57%) | 58750(22.4%)  | 76.004%  | 92.901%  | [ABCPruner-50%](https://drive.google.com/drive/folders/1p5aU800DylH-piwekTAxSM61aLM2lW3X?usp=sharing) |
| ResNet152  | 24.07M(60.01%) | 4309.52M(62.87%) | 62368(17.62%) | 77.115%  | 93.481%  | [ABCPruner-70%](https://drive.google.com/drive/folders/1Z0JofwEKpPsmXrgpQqKpmAmhngb11BTw?usp=sharing) |

---

## 🔧 环境要求

### 基础环境

```bash
Python >= 3.6
PyTorch >= 1.0.1
CUDA = 10.0.0 (如果使用GPU)
```

### 依赖库安装

```cmd
REM 基础依赖
pip install torch torchvision

REM FLOPs和参数量计算
pip install thop

REM ImageNet加速（可选，用于加速ImageNet数据加载）
pip install nvidia-dali-cuda100
```

### 预训练模型下载

#### CIFAR-10 预训练模型

| [VGG16](https://drive.google.com/open?id=1sAax46mnA01qK6S_J5jFr19Qnwbl1gpm) | [ResNet56](https://drive.google.com/open?id=1pt-LgK3kI_4ViXIQWuOP0qmmQa3p2qW5) | [ResNet110](https://drive.google.com/open?id=1Uqg8_J-q2hcsmYTAlRtknCSrkXDqYDMD) | [GoogLeNet](https://drive.google.com/open?id=1YNno621EuTQTVY2cElf8YEue9J4W5BEd) |

#### ImageNet 预训练模型

| [ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) | [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth) |

---

## 🚀 快速开始

### 1. CIFAR-10 上剪枝 ResNet56

```cmd
REM 使用ABC算法搜索最优剪枝配置并训练
python bee_cifar.py `
    --data_set cifar10 `
    --data_path ./data `
    --arch resnet_cifar `
    --cfg resnet56 `
    --honey_model ./pretrain/resnet56_cifar10.pth `
    --job_dir ./experiments/resnet56_prune `
    --gpus 0 `
    --lr 0.01 `
    --lr_decay_step 50 100 `
    --num_epochs 150 `
    --train_batch_size 128 `
    --calfitness_epoch 2 `
    --max_cycle 10 `
    --max_preserve 9 `
    --food_number 10 `
    --food_limit 5 `
    --random_rule random_pretrain
```

### 2. ImageNet 上剪枝 ResNet18

```cmd
REM 使用ABC算法搜索最优剪枝配置
python bee_imagenet.py `
    --data_path D:\data\ImageNet2012 `
    --honey_model ./pretrain/resnet18.pth `
    --job_dir ./experiments/resnet18_imagenet `
    --arch resnet `
    --cfg resnet18 `
    --gpus 0 `
    --lr 0.01 `
    --lr_decay_step 75 112 `
    --num_epochs 150 `
    --calfitness_epoch 2 `
    --max_cycle 50 `
    --max_preserve 9 `
    --food_number 10 `
    --food_limit 5 `
    --random_rule random_pretrain `
    --warm_up
```

### 3. 计算模型 FLOPs 和参数量

```cmd
REM 比较原始模型和剪枝模型的FLOPs、参数量
python get_flops_params.py `
    --data_set cifar10 `
    --arch resnet_cifar `
    --cfg resnet56 `
    --honey "5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5"
```

**输出示例：**
```
--------------UnPrune Model (原始模型)--------------
Channels: 1482
Params: 0.85 M
FLOPS: 125.49 M

--------------Prune Model (剪枝后模型)--------------
Channels: 1081
Params: 0.39 M
FLOPS: 58.54 M

--------------Compress Rate (压缩率)--------------
Channels Prune Rate: 1081/1482 (27.07%)
Params Compress Rate: 0.39 M/0.85 M(54.20%)
FLOPS Compress Rate: 58.54 M/125.49 M(53.38%)
```

### 4. 测试已剪枝的模型

```cmd
REM 直接测试已剪枝并训练好的模型
python bee_imagenet.py `
    --data_path D:\data\ImageNet2012 `
    --job_dir ./experiments/resnet18_test `
    --arch resnet `
    --cfg resnet18 `
    --honey_model ./pretrain/resnet18.pth `
    --best_honey 5 5 5 5 5 5 5 5 `
    --best_honey_s ./pruned/resnet18_pruned.pth `
    --test_only `
    --gpus 0
```

---

## 📚 详细使用说明

### 运行模式说明

ABCPruner 支持三种主要运行模式：

#### 模式1: 完整的剪枝流程（推荐新手）

```cmd
REM 从预训练模型开始，自动搜索最优剪枝配置，然后训练
python bee_cifar.py `
    --data_set cifar10 `
    --arch resnet_cifar `
    --cfg resnet56 `
    --honey_model ./pretrain/resnet56.pth `
    --job_dir ./experiments/resnet56 `
    --gpus 0
```

**流程**：预训练模型 → ABC搜索 → 剪枝 → 微调训练 → 保存最优模型

#### 模式2: 使用已知剪枝配置（跳过搜索）

```cmd
REM 如果已经有最优的剪枝配置（honey code），直接使用
python bee_cifar.py `
    --data_set cifar10 `
    --arch resnet_cifar `
    --cfg resnet56 `
    --honey_model ./pretrain/resnet56.pth `
    --job_dir ./experiments/resnet56_finetune `
    --best_honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 `
    --gpus 0
```

**流程**：预训练模型 → 直接剪枝 → 微调训练

#### 模式3: 从检查点恢复训练

```cmd
REM 从之前保存的检查点继续训练
python bee_cifar.py `
    --data_set cifar10 `
    --arch resnet_cifar `
    --cfg resnet56 `
    --resume ./experiments/resnet56/checkpoint/model_100.pt `
    --job_dir ./experiments/resnet56_resume `
    --gpus 0
```

**流程**：加载检查点 → 继续训练

---

## 🎛️ 参数详解

### 1. 数据配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_set` | str | `'cifar10'` | **数据集选择**。可选：`cifar10`、`cifar100`、`imagenet` |
| `--data_path` | str | `'/home/lmb/cvpr_vgg2/data'` | **数据集根目录路径**。CIFAR数据集会自动下载到此路径，ImageNet需手动准备 |

**详细说明：**
- `data_set`: 指定训练使用的数据集
  - `cifar10`: 10类、50,000训练图像、10,000测试图像，分辨率32×32
  - `cifar100`: 100类、50,000训练图像、10,000测试图像，分辨率32×32
  - `imagenet`: 1000类、约130万训练图像、50,000验证图像，分辨率224×224
- `data_path`: 数据集存储位置，程序会在该路径下查找或下载数据

**使用示例：**
```cmd
REM CIFAR-10数据集（自动下载）
--data_set cifar10 --data_path ./data/cifar10

REM ImageNet数据集（需手动准备）
--data_set imagenet --data_path D:\data\ImageNet2012
```

---

### 2. 模型架构参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--arch` | str | `'vgg_cifar'` | **网络架构类型**。决定使用哪个模型文件 |
| `--cfg` | str | `'vgg16'` | **具体网络配置**。指定模型的深度和结构 |

**架构对应关系：**

| `--arch` 值 | 对应的模型文件 | 适用数据集 | 可选的 `--cfg` 值 |
|------------|--------------|-----------|----------------|
| `vgg_cifar` | `model/vgg_cifar.py` | CIFAR-10/100 | `vgg16`, `vgg19` |
| `resnet_cifar` | `model/resnet_cifar.py` | CIFAR-10/100 | `resnet18`, `resnet34`, `resnet50`, `resnet56`, `resnet110` |
| `vgg` | `model/vgg.py` | ImageNet | `vgg16`, `vgg19` |
| `resnet` | `model/resnet.py` | ImageNet | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| `googlenet` | `model/googlenet.py` | CIFAR-10/100 | `googlenet` |
| `densenet` | `model/densenet.py` | CIFAR-10/100, ImageNet | `densenet121`, `densenet169`, `densenet201` |

**使用示例：**
```cmd
REM CIFAR-10上的ResNet56
--arch resnet_cifar --cfg resnet56

REM ImageNet上的ResNet50
--arch resnet --cfg resnet50

REM CIFAR-10上的VGG16
--arch vgg_cifar --cfg vgg16
```

---

### 3. 预训练模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--honey_model` | str | `None` | **[必需] 预训练模型路径**。作为剪枝的基础模型 |

**详细说明：**
- 必须是与`--arch`和`--cfg`匹配的预训练模型权重文件
- 模型应该在目标数据集上已经训练至收敛
- 剪枝算法会基于该模型的权重进行通道选择和初始化

**使用示例：**
```cmd
REM CIFAR-10预训练模型
--honey_model ./pretrain/resnet56_cifar10.pth

REM ImageNet预训练模型（PyTorch官方）
--honey_model ./pretrain/resnet50-19c8e357.pth
```

---

### 4. 输出配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--job_dir` | str | `'experiments/'` | **实验输出目录**。保存模型、日志、配置文件 |
| `--reset` | bool flag | `False` | **重置输出目录**。删除job_dir中的已有内容（谨慎使用）|

**详细说明：**
- `job_dir`: 所有实验输出都保存在此目录下
  - `checkpoint/`: 保存模型检查点（.pt文件）
  - `run/`: TensorBoard日志文件
  - `logger.log`: 训练日志，包含详细的训练信息和最优honey code
  - `config.txt`: 保存本次实验的所有参数配置
- `reset`: 如果设置，会在训练前清空job_dir目录（小心使用，会删除已有结果）

**目录结构示例：**
```
experiments/resnet56_prune/
├── checkpoint/
│   ├── model_1.pt              # 第1个epoch的模型
│   ├── model_50.pt             # 第50个epoch的模型
│   ├── model_best.pt           # 微调阶段的最优模型
│   └── bestmodel_after_bee.pt  # ABC搜索后的最优剪枝模型
├── run/
│   └── events.out.tfevents.*   # TensorBoard日志
├── logger.log                  # 训练日志
└── config.txt                  # 参数配置
```

---

### 5. GPU 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gpus` | int list | `[0]` | **使用的GPU设备ID**。支持单卡或多卡训练 |

**详细说明：**
- 单GPU训练: `--gpus 0` (使用GPU 0)
- 多GPU训练: `--gpus 0 1 2 3` (使用4个GPU，自动启用DataParallel)
- 程序会自动检测可用GPU并进行分配

**使用示例：**
```cmd
REM 使用单个GPU
--gpus 0

REM 使用多个GPU（自动数据并行）
--gpus 0 1 2 3
```

---

### 6. 训练超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lr` | float | `0.1` | **初始学习率**。控制参数更新的步长 |
| `--lr_decay_step` | int list | `[30]` | **学习率衰减节点**。在指定epoch处将学习率乘以0.1 |
| `--num_epochs` | int | `150` | **微调训练总轮数**。ABC搜索后的fine-tuning阶段epoch数 |
| `--train_batch_size` | int | `256` | **训练batch size**。每次迭代处理的样本数 |
| `--eval_batch_size` | int | `256` | **测试batch size**。评估时的batch size |
| `--momentum` | float | `0.9` | **SGD动量系数**。加速收敛并减少震荡 |
| `--weight_decay` | float | `1e-4` | **权重衰减（L2正则化）系数**。防止过拟合 |
| `--num_workers` | int | `4` | **数据加载线程数**。并行加载数据的worker进程数量 |

**详细说明：**
- `lr`: 初始学习率
  - CIFAR数据集推荐: 0.01 - 0.1
  - ImageNet数据集推荐: 0.01 - 0.1（使用warm_up时从0.01开始）
- `lr_decay_step`: 学习率衰减策略，使用MultiStepLR
  - 例如 `--lr_decay_step 50 100` 表示在第50和100个epoch时学习率×0.1
- `num_epochs`: 微调训练轮数
  - CIFAR: 通常150-200
  - ImageNet: 通常90-150
- `train_batch_size` 和 `eval_batch_size`: 根据GPU显存调整
  - 单卡GTX 1080Ti (11GB): 128-256 (CIFAR), 64-128 (ImageNet)
  - 多卡可按比例增大
- `num_workers`: 数据加载并行进程数
  - 值越大，数据加载越快，但占用更多CPU和内存
  - 推荐值: 2-8（根据CPU核心数调整）
  - 如果遇到"too many open files"错误，减小此值
  - Windows系统建议使用较小的值（2-4）

**推荐配置：**
```cmd
REM CIFAR-10/100 标准配置
--lr 0.01 --lr_decay_step 50 100 --num_epochs 150 --train_batch_size 128 --num_workers 4

REM ImageNet 标准配置
--lr 0.01 --lr_decay_step 30 60 --num_epochs 90 --train_batch_size 256 --num_workers 4 --warm_up

REM ImageNet 长训练配置
--lr 0.01 --lr_decay_step 75 112 --num_epochs 150 --train_batch_size 256 --num_workers 4 --warm_up
```

---

### 7. ABC 算法核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--calfitness_epoch` | int | `2` | **适应度评估轮数**。每个候选方案训练的epoch数 |
| `--max_cycle` | int | `10` | **ABC搜索周期数**。算法迭代的最大周期 |
| `--max_preserve` | int | `9` | **最大保留等级**。通道保留数量的上限（1-9等级）|
| `--food_number` | int | `10` | **食物源数量**。候选剪枝方案的数量 |
| `--food_limit` | int | `5` | **食物源更新限制**。连续未改进的最大次数 |
| `--honeychange_num` | int | `2` | **编码变更数量**。每次更新改变的编码维度数 |

**详细说明：**

1. **`calfitness_epoch`** - 适应度计算精度 vs 速度权衡
   - 值越大：适应度评估越准确，但单个周期耗时越长
   - 值越小：搜索速度快，但可能选择次优方案
   - 推荐值：
     - 快速实验: 1
     - 标准实验: 2-3
     - 高精度实验: 5

2. **`max_cycle`** - 搜索充分性
   - ABC算法的主循环次数，每个周期包含三个阶段：
     - 雇佣蜂阶段(Employed Bee): 局部搜索改进
     - 观察蜂阶段(Onlooker Bee): 基于适应度的全局搜索
     - 侦察蜂阶段(Scout Bee): 探索新区域（重置停滞的食物源）
   - 推荐值：
     - CIFAR-10/100: 10-20
     - ImageNet: 30-50

3. **`max_preserve`** - 模型大小控制
   - 编码值1-9对应不同的通道保留比例
   - 值越大，保留的通道越多，模型越大，精度越高
   - 通常设为9，让算法自动搜索每层的最优值（1-9范围内）

4. **`food_number`** - 搜索广度
   - 同时维护的候选方案数量
   - 越大：搜索空间越大，越可能找到最优解，但计算量增加
   - 推荐值：5-20
   - 注意：与GPU显存需求成正比

5. **`food_limit`** - 跳出局部最优
   - 食物源连续未改进达到此限制后，由侦察蜂重新随机初始化
   - 防止算法过早收敛到局部最优
   - 推荐值：3-10

6. **`honeychange_num`** - 搜索步长
   - 每次邻域搜索时随机改变的编码维度数
   - 值越大：探索性越强，但可能错过局部最优
   - 推荐值：1-3

**时间复杂度估算：**
```
总时间 ≈ max_cycle × food_number × calfitness_epoch × 单epoch时间
```

**ABC参数配置建议：**
```cmd
REM 快速测试（1-2小时，CIFAR-10，单GPU）
--calfitness_epoch 1 --max_cycle 5 --food_number 5 --food_limit 3

REM 标准配置（5-10小时，CIFAR-10，单GPU）
--calfitness_epoch 2 --max_cycle 10 --food_number 10 --food_limit 5

REM 高质量配置（20-30小时，CIFAR-10，单GPU）
--calfitness_epoch 5 --max_cycle 20 --food_number 20 --food_limit 10

REM ImageNet配置（需要更多周期和GPU）
--calfitness_epoch 2 --max_cycle 50 --food_number 10 --food_limit 5
```

---

### 8. 剪枝配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--best_honey` | int list | `None` | **最优剪枝编码**。跳过ABC搜索，直接使用指定配置 |
| `--best_honey_s` | str | `None` | **已剪枝模型路径**。直接加载剪枝后的模型权重 |
| `--best_honey_past` | int list | `None` | **历史最优编码**。从检查点恢复时指定之前的最优配置 |
| `--random_rule` | str | `'default'` | **权重继承策略**。控制剪枝后如何初始化模型权重 |

**详细说明：**

1. **`best_honey`** - 直接使用已知最优配置
   - 格式：空格分隔的整数列表，每个值对应一层的保留等级（1-9）
   - 长度：等于网络的可剪枝层数
   - 使用场景：
     - ABC搜索完成后，使用搜索到的最优配置重新训练
     - 复现论文结果
     - 跳过耗时的搜索过程
   - 示例：`--best_honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5`

2. **`best_honey_s`** - 加载已剪枝模型
   - 与`best_honey`配合使用
   - 直接加载经过剪枝和训练的模型权重
   - 使用场景：
     - 测试已训练好的剪枝模型
     - 在剪枝模型基础上继续微调

3. **`random_rule`** - 权重继承策略（重要）
   - `default`: 直接复制前N个通道的权重
     - 简单快速，但可能保留不重要的通道
   - `random_pretrain`: 随机选择N个通道的权重
     - 增加多样性，但无理论依据
   - `l1_pretrain`: **推荐** - 基于L1范数选择最重要的N个通道
     - 选择权重绝对值和最大的通道
     - 理论上更合理，通常效果最好

**使用场景示例：**
```cmd
REM 场景1: 首次运行ABC搜索（不指定best_honey）
python bee_cifar.py \
    --honey_model ./pretrain/resnet56.pth \
    --job_dir ./exp1 \
    --calfitness_epoch 2 --max_cycle 10

REM 查看搜索结果
findstr "Best Honey Source" ./exp1/logger.log
REM 输出示例: Best Honey Source [5, 5, 6, 7, 5, 5, 4, ...]

REM 场景2: 使用搜索到的配置重新训练（跳过搜索）
python bee_cifar.py \
    --honey_model ./pretrain/resnet56.pth \
    --best_honey 5 5 6 7 5 5 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 \
    --random_rule l1_pretrain \
    --job_dir ./exp2

REM 场景3: 测试已剪枝并训练好的模型
python bee_cifar.py \
    --honey_model ./pretrain/resnet56.pth \
    --best_honey 5 5 6 7 5 5 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 \
    --best_honey_s ./exp2/checkpoint/bestmodel_after_bee.pt \
    --test_only
```

---

### 9. 检查点和恢复参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--resume` | str | `None` | **恢复训练检查点路径**。从中断处继续训练 |
| `--refine` | str | `None` | **微调检查点路径**。加载模型继续fine-tune |

**详细说明：**
- `resume`: 完全恢复训练状态，包括：
  - 模型权重
  - 优化器状态
  - 学习率调度器状态
  - 当前epoch数
  - 最优精度
- `refine`: 仅加载模型权重，其他状态重新初始化

**使用示例：**
```cmd
REM 从中断处恢复训练
--resume ./experiments/resnet56/checkpoint/model_100.pt

REM 微调已有模型
--refine ./experiments/resnet56/checkpoint/model_best.pt
```

---

### 10. 特殊功能参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test_only` | bool flag | `False` | **仅测试模式**。不训练，只在测试集上评估模型 |
| `--from_scratch` | bool flag | `False` | **从头训练未剪枝模型**。训练baseline模型 |
| `--bee_from_scratch` | bool flag | `False` | **从头开始ABC搜索**。忽略之前的搜索结果 |
| `--warm_up` | bool flag | `False` | **学习率预热**。前5个epoch线性增长学习率（ImageNet推荐）|
| `--label_smooth` | bool flag | `False` | **标签平滑**。防止过拟合，提升泛化能力 |
| `--split_optimizer` | bool flag | `False` | **分离优化器参数**。BatchNorm层不使用权重衰减 |

**详细说明：**

1. **`test_only`** - 评估模式
   - 跳过所有训练，直接在测试集上评估
   - 需配合`--best_honey`和`--best_honey_s`使用
   - 快速验证模型性能

2. **`from_scratch`** - 训练baseline
   - 训练未剪枝的原始网络
   - 用于对比实验，评估剪枝的效果
   - 不执行ABC算法

3. **`bee_from_scratch`** - 重新搜索
   - 即使存在之前的搜索记录，也重新开始ABC搜索
   - 用于完全重复实验

4. **`warm_up`** - 学习率预热（ImageNet推荐）
   - 前5个epoch学习率从0线性增长到初始lr
   - 稳定大batch size训练
   - ImageNet训练强烈推荐使用

5. **`label_smooth`** - 标签平滑
   - 将hard label（0/1）软化为接近0/1的值
   - 提升模型泛化能力
   - 可能轻微提升精度

6. **`split_optimizer`** - 优化器参数分离
   - BatchNorm的weight和bias不应用权重衰减
   - 理论上更合理，但提升有限

**使用示例：**
```cmd
REM 仅测试模型
python bee_cifar.py --test_only \
    --honey_model ./pretrain/resnet56.pth \
    --best_honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 \
    --best_honey_s ./experiments/pruned_model.pt

REM 训练未剪枝的baseline模型
python bee_cifar.py --from_scratch \
    --honey_model ./pretrain/resnet56.pth \
    --job_dir ./baseline

REM ImageNet训练（使用预热和标签平滑）
python bee_imagenet.py --warm_up --label_smooth \
    --honey_model ./pretrain/resnet50.pth \
    --lr 0.01 --num_epochs 90
```

---

## 💡 常见问题 (FAQ)

### 1. GPU显存不足 (CUDA out of memory)

**解决方案**：
- 减小batch size: `--train_batch_size 64 --eval_batch_size 128`
- 减少食物源数量: `--food_number 5`
- 使用更小的网络: `--cfg resnet18`

### 2. ABC搜索时间过长

**解决方案**：
- 减少搜索周期: `--max_cycle 5`
- 减少适应度计算轮数: `--calfitness_epoch 1`
- 减少食物源数量: `--food_number 5`

### 3. 剪枝后准确率下降过多

**解决方案**：
- 增大max_preserve保留更多通道: `--max_preserve 8` 或 `9`
- 使用L1范数选择重要通道: `--random_rule l1_pretrain`
- 增加微调训练轮数: `--num_epochs 200`

---

## 📖 引用

如果您在研究中使用了 ABCPruner，请引用我们的论文：

```bibtex
@inproceedings{lin2020channel,
  title={Channel Pruning via Automatic Structure Search},
  author={Lin, Mingbao and Ji, Rongrong and Zhang, Yuxin and Zhang, Baochang and Wu, Yongjian and Tian, Yonghong},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={673--679},
  year={2020}
}
```

---

## 📧 联系方式

如有问题，请通过邮件联系：
- 邮箱: lmbxmu@stu.xmu.edu.cn 或 yxzhangxmu@163.com

---

## 📜 许可证

本项目遵循 MIT 许可证。

---

如果本项目对您有帮助，欢迎给我们一个 ⭐ Star！
