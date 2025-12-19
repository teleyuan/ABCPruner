"""
命令行参数配置文件

包含所有BeePruning算法的超参数和配置选项：

参数类别:
- GPU设置: gpus
- 数据集: dataset, data_path
- 日志: job_dir, reset
- 检查点: resume, refine
- 网络: arch, cfg
- 训练: num_epochs, train_batch_size, eval_batch_size, momentum, lr, lr_decay_step, weight_decay
- ABC算法: honey_model, calfitness_epoch, max_cycle, food_number, food_limit, honeychange_num等
"""

import argparse
import ast
import os

parser = argparse.ArgumentParser(description='使用人工蜂群算法进行神经网络剪枝')

# ==================== 训练模式选项 ====================
parser.add_argument(
    '--from_scratch',
    action='store_true',
    help='是否从头训练（不使用预训练模型）'
)

parser.add_argument(
    '--bee_from_scratch',
    action='store_true',
    help='是否从头开始执行BeePruning（不加载之前的搜索结果）'
)

parser.add_argument(
    '--label_smooth',
    action='store_true',
    help='是否使用标签平滑（Label Smoothing）'
)

parser.add_argument(
    '--split_optimizer',
    action='store_true',
    help='是否分离需要权重衰减的参数'
)

parser.add_argument(
    '--warm_up',
    action='store_true',
    help='是否使用学习率预热（Warm Up）'
)
# ==================== GPU和数据集配置 ====================
parser.add_argument(
	'--gpus',
	type=int,
	nargs='+',
	default=None,
	help='使用的GPU ID列表，例如：[0] 或 [0,1,2,3]',
)

parser.add_argument(
	'--data_set',
	type=str,
	default='cifar10',
	help='数据集名称：cifar10, cifar100, imagenet',
)

parser.add_argument(
	'--data_path',
	type=str,
	default='/home/lmb/cvpr_vgg2/data',
	help='数据集存储路径',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored. default:./experiments'
)

parser.add_argument(
    '--reset',
    action='store_true',
    help='reset the directory?'
)

parser.add_argument(
	'--resume',
	type=str,
	default=None,
	help='Load the model from the specified checkpoint.'
)

parser.add_argument(
	'--refine',
	type=str,
	default=None,
	help='Path to the model to be fine-tuned.'
)

## Training
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_cifar',
    help='Architecture of model. default:vgg_cifar'
)

parser.add_argument(
    '--cfg',
    type=str,
    default='vgg16',
    help='Detail architecuture of model. default:vgg16'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=150,
    help='The num of epochs to train. default:150'
)

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size for training. default:256'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=256,
    help='Batch size for validation. default:256'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.1,
    help='Learning rate for train. default:0.1'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+',
    default=[30],
    help='the iterval of learn rate decay. default:[30]. Example: --lr_decay_step 50 100'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='The weight decay of loss. default:1e-4'
)

parser.add_argument(
    '--random_rule',
    type=str,
    default='default',
    help='Weight initialization criterion after random clipping. default:default optional:default,random_pretrain,l1_pretrain'
)

parser.add_argument(
    '--test_only',
    action='store_true',
    help='Test only?')

# ==================== ABC算法（BeePruning）超参数 ====================
parser.add_argument(
    '--honey_model',
    type=str,
    default=None,
    help='待剪枝的预训练模型路径'
)

parser.add_argument(
    '--calfitness_epoch',
    type=int,
    default=2,
    help='计算适应度时的训练轮数（默认:2）。训练轮数越多，适应度评估越准确但耗时更长'
)

parser.add_argument(
    '--max_cycle',
    type=int,
    default=10,
    help='ABC算法的最大搜索周期数（默认:10）。每个周期包括雇佣蜂、观察蜂和侦察蜂三个阶段'
)

parser.add_argument(
    '--max_preserve',
    type=int,
    default=9,
    help='每层最大保留通道数等级（1-9）。值越大保留的通道越多'
)

parser.add_argument(
    '--preserve_type',
    type = str,
    default = 'layerwise',
    help = '剪枝策略类型：layerwise(逐层剪枝) 或 global(全局剪枝)'
)

parser.add_argument(
    '--food_number',
    type=int,
    default=10,
    help='食物源数量（候选剪枝方案数量），默认:10'
)

parser.add_argument(
    '--food_dimension',
    type=int,
    default=13,
    help='食物源维度（可剪枝的卷积层数量）。例如：VGG16有13个可剪枝层'
)

parser.add_argument(
    '--food_limit',
    type=int,
    default=5,
    help='食物源未改进的最大次数限制。超过此限制后将由侦察蜂重新初始化'
)

parser.add_argument(
    '--honeychange_num',
    type=int,
    default=2,
    help='每次蜜蜂更新时改变的编码数量（默认:2）'
)

parser.add_argument(
    '--best_honey',
    type=int,
    nargs='+',
    default=None,
    help='最优剪枝配置。如果提供此参数，将跳过ABC搜索直接使用该配置进行微调'
)

parser.add_argument(
    '--best_honey_s',
    type=str,
    default=None,
    help='最优剪枝模型的权重文件路径'
)

parser.add_argument(
    '--best_honey_past',
    type=int,
    nargs='+',
    default=None,
    help='之前搜索得到的最优配置（用于恢复训练）'
)

args = parser.parse_args()

# 网络架构配置：记录每个网络的可剪枝卷积层数量
netcfg = {
    'vgg16':13,        # VGG16: 13个卷积层
    'resnet18':8,      # ResNet18: 8个可剪枝的卷积块
    'resnet56':27,     # ResNet56: 27个可剪枝的卷积层
    'resnet110':54,    # ResNet110: 54个可剪枝的卷积层
    'resnet34' : 16,   # ResNet34: 16个可剪枝的卷积块
    'resnet50' : 16,   # ResNet50: 16个bottleneck块
    'resnet101' : 33,  # ResNet101: 33个bottleneck块
    'resnet152' : 50,  # ResNet152: 50个bottleneck块
    'googlenet': 9,    # GoogLeNet: 9个Inception模块
    'densenet': 36,    # DenseNet: 36个可剪枝的卷积层
}

# 根据网络配置自动设置食物源维度
args.food_dimension = netcfg[args.cfg]

# 验证检查点文件路径
if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('未找到用于恢复训练的检查点: {}'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('未找到用于微调的检查点: {}'.format(args.refine))

