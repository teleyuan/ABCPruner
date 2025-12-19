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
python bee_cifar.py ^
    --data_set cifar10 ^
    --data_path ./data ^
    --arch resnet_cifar ^
    --cfg resnet56 ^
    --honey_model ./pretrain/resnet56_cifar10.pth ^
    --job_dir ./experiments/resnet56_prune ^
    --gpus 0 ^
    --lr 0.01 ^
    --lr_decay_step 50 100 ^
    --num_epochs 150 ^
    --train_batch_size 128 ^
    --calfitness_epoch 2 ^
    --max_cycle 10 ^
    --max_preserve 9 ^
    --food_number 10 ^
    --food_limit 5 ^
    --random_rule random_pretrain
```

### 2. ImageNet 上剪枝 ResNet18

```cmd
REM 使用ABC算法搜索最优剪枝配置
python bee_imagenet.py ^
    --data_path D:\data\ImageNet2012 ^
    --honey_model ./pretrain/resnet18.pth ^
    --job_dir ./experiments/resnet18_imagenet ^
    --arch resnet ^
    --cfg resnet18 ^
    --gpus 0 ^
    --lr 0.01 ^
    --lr_decay_step 75 112 ^
    --num_epochs 150 ^
    --calfitness_epoch 2 ^
    --max_cycle 50 ^
    --max_preserve 9 ^
    --food_number 10 ^
    --food_limit 5 ^
    --random_rule random_pretrain ^
    --warm_up
```

### 3. 计算模型 FLOPs 和参数量

```cmd
REM 比较原始模型和剪枝模型的FLOPs、参数量
python get_flops_params.py ^
    --data_set cifar10 ^
    --arch resnet_cifar ^
    --cfg resnet56 ^
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
python bee_imagenet.py ^
    --data_path D:\data\ImageNet2012 ^
    --job_dir ./experiments/resnet18_test ^
    --arch resnet ^
    --cfg resnet18 ^
    --honey_model ./pretrain/resnet18.pth ^
    --best_honey 5 5 5 5 5 5 5 5 ^
    --best_honey_s ./pruned/resnet18_pruned.pth ^
    --test_only ^
    --gpus 0
```

---

## 📚 详细使用说明

### 运行模式说明

ABCPruner 支持三种主要运行模式：

#### 模式1: 完整的剪枝流程（推荐新手）

```cmd
REM 从预训练模型开始，自动搜索最优剪枝配置，然后训练
python bee_cifar.py ^
    --data_set cifar10 ^
    --arch resnet_cifar ^
    --cfg resnet56 ^
    --honey_model ./pretrain/resnet56.pth ^
    --job_dir ./experiments/resnet56 ^
    --gpus 0
```

**流程**：预训练模型 → ABC搜索 → 剪枝 → 微调训练 → 保存最优模型

#### 模式2: 使用已知剪枝配置（跳过搜索）

```cmd
REM 如果已经有最优的剪枝配置（honey code），直接使用
python bee_cifar.py ^
    --data_set cifar10 ^
    --arch resnet_cifar ^
    --cfg resnet56 ^
    --honey_model ./pretrain/resnet56.pth ^
    --job_dir ./experiments/resnet56_finetune ^
    --best_honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 ^
    --gpus 0
```

**流程**：预训练模型 → 直接剪枝 → 微调训练

#### 模式3: 从检查点恢复训练

```cmd
REM 从之前保存的检查点继续训练
python bee_cifar.py ^
    --data_set cifar10 ^
    --arch resnet_cifar ^
    --cfg resnet56 ^
    --resume ./experiments/resnet56/checkpoint/model_100.pt ^
    --job_dir ./experiments/resnet56_resume ^
    --gpus 0
```

**流程**：加载检查点 → 继续训练

---

## 🎛️ 参数详解

### GPU 和数据集配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gpus` | int list | `[0]` | 使用的GPU ID列表。例如：`--gpus 0` 使用单GPU，`--gpus 0 1 2 3` 使用4个GPU |
| `--data_set` | str | `'cifar10'` | 数据集名称。可选：`cifar10`、`cifar100`、`imagenet` |
| `--data_path` | str | `'/home/lmb/cvpr_vgg2/data'` | 数据集存储路径。CIFAR数据集会自动下载到此路径 |

**使用示例：**
```cmd
REM 使用GPU 0
--gpus 0

REM 使用多GPU（0,1,2,3）
--gpus 0 1 2 3

REM 使用CIFAR-10数据集
--data_set cifar10 --data_path ./data/cifar10

REM 使用ImageNet数据集
--data_set imagenet --data_path D:\data\ImageNet2012
```

---

### 网络架构配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--arch` | str | `'vgg_cifar'` | 网络架构类型。可选：`vgg_cifar`（CIFAR的VGG）、`resnet_cifar`（CIFAR的ResNet）、`vgg`（ImageNet的VGG）、`resnet`（ImageNet的ResNet）、`googlenet`、`densenet` |
| `--cfg` | str | `'vgg16'` | 具体的网络配置。可选：`vgg16`、`resnet18`、`resnet34`、`resnet50`、`resnet56`、`resnet110`、`googlenet`、`densenet` |

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

### 训练超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_epochs` | int | `150` | 总训练轮数。CIFAR通常150-200，ImageNet通常90-150 |
| `--train_batch_size` | int | `256` | 训练时的batch size。根据GPU显存调整，单GPU通常128-256 |
| `--eval_batch_size` | int | `256` | 测试时的batch size。可以设置更大以加快测试速度 |
| `--lr` | float | `0.1` | 初始学习率。CIFAR通常0.1，ImageNet通常0.01-0.1 |
| `--lr_decay_step` | int list | `[30]` | 学习率衰减的epoch节点。例如：`--lr_decay_step 50 100` 表示在第50和100个epoch衰减 |
| `--momentum` | float | `0.9` | SGD优化器的动量参数。通常保持0.9 |
| `--weight_decay` | float | `1e-4` | 权重衰减（L2正则化）系数。通常1e-4到5e-4 |

**使用示例：**
```cmd
REM CIFAR-10标准配置
--num_epochs 150 --lr 0.1 --lr_decay_step 50 100 --train_batch_size 128

REM ImageNet标准配置
--num_epochs 90 --lr 0.01 --lr_decay_step 30 60 --train_batch_size 256

REM 学习率预热（ImageNet推荐）
--lr 0.01 --warm_up
```

---

### ABC 算法超参数（核心）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--honey_model` | str | `None` | **必需**。待剪枝的预训练模型路径。这是剪枝的起点 |
| `--calfitness_epoch` | int | `2` | 计算适应度时训练的轮数。越大越准确但越慢。快速测试用1，正式实验用2-5 |
| `--max_cycle` | int | `10` | ABC算法的最大搜索周期数。每个周期包括雇佣蜂、观察蜂、侦察蜂三阶段。CIFAR用10-20，ImageNet用30-50 |
| `--max_preserve` | int | `9` | 每层最大保留通道数等级（1-9）。值越大保留的通道越多，模型越大。通常使用9 |
| `--food_number` | int | `10` | 食物源数量（候选剪枝方案数量）。越大搜索空间越大但越慢。通常5-20 |
| `--food_limit` | int | `5` | 食物源未改进的最大次数限制。超过后由侦察蜂重新初始化。通常3-10 |
| `--honeychange_num` | int | `2` | 每次蜜蜂更新时改变的编码维度数量。通常1-3 |

**ABC算法参数调优建议：**

```cmd
REM 快速测试配置（约1-2小时，CIFAR-10）
--calfitness_epoch 1 --max_cycle 5 --food_number 5

REM 标准配置（约5-10小时，CIFAR-10）
--calfitness_epoch 2 --max_cycle 10 --food_number 10

REM 高质量配置（约20-30小时，CIFAR-10）
--calfitness_epoch 5 --max_cycle 20 --food_number 20

REM ImageNet配置（需要更多周期）
--calfitness_epoch 2 --max_cycle 50 --food_number 10
```

---

### 剪枝配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--best_honey` | int list | `None` | 最优剪枝配置（honey code）。如果提供，将跳过ABC搜索，直接使用该配置。格式：每个数字代表一层的保留等级（1-9）|
| `--best_honey_s` | str | `None` | 已剪枝模型的权重文件路径。与`--best_honey`配合使用，直接加载剪枝好的模型 |
| `--best_honey_past` | int list | `None` | 之前搜索得到的最优配置。用于从检查点恢复时指定剪枝配置 |
| `--random_rule` | str | `'default'` | 权重继承规则。可选：`default`（直接复制）、`random_pretrain`（随机选择通道）、`l1_pretrain`（基于L1范数选择重要通道，推荐）|

**使用示例：**
```cmd
REM 运行ABC搜索（会在日志中输出最优honey code）
python bee_cifar.py ^
    --honey_model ./pretrain/resnet56.pth ^
    --job_dir ./exp1

REM 查看搜索结果
findstr "Best Honey Source" ./exp1/logger.log
REM 输出: Best Honey Source [5, 5, 5, 5, 5, 5, 5, 5, ...]

REM 使用搜索到的配置直接剪枝（跳过搜索）
python bee_cifar.py ^
    --honey_model ./pretrain/resnet56.pth ^
    --best_honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 ^
    --job_dir ./exp2

REM 测试已剪枝的模型
python bee_cifar.py ^
    --honey_model ./pretrain/resnet56.pth ^
    --best_honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 ^
    --best_honey_s ./exp1/checkpoint/bestmodel_after_bee.pt ^
    --test_only
```

---

### 日志和检查点

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--job_dir` | str | `'experiments/'` | 实验结果保存目录。会在此目录下创建`checkpoint`（模型）和`run`（日志）子目录 |
| `--reset` | bool flag | `False` | 是否重置job_dir目录（删除已有内容）。小心使用！|
| `--resume` | str | `None` | 从指定检查点恢复训练。路径格式：`./experiments/job/checkpoint/model_100.pt` |
| `--refine` | str | `None` | 微调模型的检查点路径。用于在已剪枝模型基础上继续训练 |

**目录结构：**
```
job_dir/
├── checkpoint/              # 模型检查点
│   ├── model_1.pt          # 第1个epoch的模型
│   ├── model_best.pt       # 最优模型
│   └── bestmodel_after_bee.pt  # ABC搜索后的最优剪枝模型
├── run/                    # TensorBoard日志
├── logger.log              # 训练日志（包含best honey code）
└── config.txt              # 训练配置参数
```

---

### 特殊模式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test_only` | bool flag | `False` | 仅测试模式，不进行训练 |
| `--from_scratch` | bool flag | `False` | 从头训练未剪枝的模型（baseline） |
| `--bee_from_scratch` | bool flag | `False` | 从头开始执行BeePruning（不加载之前的搜索结果）|
| `--warm_up` | bool flag | `False` | 使用学习率预热策略（ImageNet推荐）|
| `--label_smooth` | bool flag | `False` | 使用标签平滑（Label Smoothing）|
| `--split_optimizer` | bool flag | `False` | 分离需要权重衰减的参数 |

**使用示例：**
```cmd
REM 仅测试模型
--test_only

REM 训练未剪枝的baseline模型
--from_scratch

REM ImageNet训练使用学习率预热
--warm_up

REM 使用标签平滑提升泛化能力
--label_smooth
```

---

## 📊 完整示例

### 示例1: CIFAR-10上完整的剪枝流程

```cmd
REM Step 1: 创建目录
mkdir pretrain
mkdir data
mkdir experiments

REM Step 2: 下载预训练模型（ResNet56）
REM 从Google Drive下载到 pretrain\resnet56_cifar10.pth

REM Step 3: 运行ABC算法搜索最优剪枝配置
python bee_cifar.py ^
    --data_set cifar10 ^
    --data_path ./data ^
    --arch resnet_cifar ^
    --cfg resnet56 ^
    --honey_model ./pretrain/resnet56_cifar10.pth ^
    --job_dir ./experiments/resnet56_abc ^
    --gpus 0 ^
    --num_epochs 150 ^
    --lr 0.01 ^
    --lr_decay_step 50 100 ^
    --train_batch_size 128 ^
    --calfitness_epoch 2 ^
    --max_cycle 10 ^
    --max_preserve 9 ^
    --food_number 10 ^
    --food_limit 5 ^
    --random_rule l1_pretrain

REM Step 4: 查看搜索到的最优配置
findstr "Best Honey Source" ./experiments/resnet56_abc/logger.log
REM 假设输出: Best Honey Source [5, 5, 5, 5, 5, 5, 5, ...]

REM Step 5: 计算剪枝后的FLOPs和参数量
python get_flops_params.py ^
    --data_set cifar10 ^
    --arch resnet_cifar ^
    --cfg resnet56 ^
    --honey "5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5"

REM Step 6: 测试最终模型
python bee_cifar.py ^
    --data_set cifar10 ^
    --data_path ./data ^
    --arch resnet_cifar ^
    --cfg resnet56 ^
    --honey_model ./pretrain/resnet56_cifar10.pth ^
    --best_honey 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 ^
    --best_honey_s ./experiments/resnet56_abc/checkpoint/bestmodel_after_bee.pt ^
    --job_dir ./experiments/resnet56_test ^
    --test_only ^
    --gpus 0
```

### 示例2: ImageNet上快速测试

```cmd
REM 使用较小的搜索周期快速测试
python bee_imagenet.py ^
    --data_path D:\data\ImageNet2012 ^
    --honey_model ./pretrain/resnet18.pth ^
    --job_dir ./experiments/resnet18_quick ^
    --arch resnet ^
    --cfg resnet18 ^
    --gpus 0 ^
    --num_epochs 90 ^
    --lr 0.01 ^
    --lr_decay_step 30 60 ^
    --train_batch_size 256 ^
    --calfitness_epoch 1 ^
    --max_cycle 10 ^
    --food_number 5 ^
    --random_rule random_pretrain ^
    --warm_up
```

---

## 💡 常见问题 (FAQ)

### 1. CUDA out of memory 错误

**原因**：GPU显存不足

**解决方案**：
```cmd
REM 减小batch size
--train_batch_size 64 --eval_batch_size 128

REM 减少食物源数量
--food_number 5

REM 使用更小的网络
--cfg resnet18  REM 而不是resnet50
```

### 2. ABC搜索时间过长

**原因**：搜索周期和适应度计算轮数太多

**解决方案**：
```cmd
REM 减少搜索周期
--max_cycle 5

REM 减少适应度计算轮数
--calfitness_epoch 1

REM 减少食物源数量
--food_number 5
```

### 3. 剪枝后准确率下降过多

**原因**：剪枝过于激进

**解决方案**：
```cmd
REM 增大max_preserve，保留更多通道
--max_preserve 7  REM 或 8、9

REM 使用L1范数选择重要通道
--random_rule l1_pretrain

REM 增加微调训练的轮数
--num_epochs 200
```

### 4. ImageNet数据加载慢

**解决方案**：
```cmd
REM 安装NVIDIA DALI加速库
pip install nvidia-dali-cuda100

REM 增加数据加载线程（在代码中修改num_workers）
```

### 5. 如何选择最优的ABC参数？

**快速实验**（1-2小时）：
```cmd
--calfitness_epoch 1 --max_cycle 5 --food_number 5
```

**标准实验**（5-10小时）：
```cmd
--calfitness_epoch 2 --max_cycle 10 --food_number 10
```

**高质量实验**（20-30小时）：
```cmd
--calfitness_epoch 5 --max_cycle 20 --food_number 20
```

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

如有任何问题，欢迎通过邮件联系作者：
- **邮箱**: lmbxmu@stu.xmu.edu.cn 或 yxzhangxmu@163.com
- **建议**: 请优先使用邮件联系，以确保能及时收到回复

**注意**: 由于GitHub邮件通知可能遗漏，请尽量避免在GitHub上提交issue，直接发邮件联系会得到更快的响应。

---

## 📜 许可证

本项目遵循 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢所有为本项目做出贡献的研究者和开发者！

**Star History**

如果本项目对您有帮助，欢迎给我们一个 ⭐ Star！

---

**Happy Pruning! 🎉**
