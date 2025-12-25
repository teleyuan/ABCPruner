"""
简单的训练和测试脚本

功能:
1. 训练神经网络模型
2. 测试模型在测试集上的准确率
3. 保存最佳模型和定期检查点
4. 从检查点恢复训练

使用示例:
    # 训练模型
    python train_test.py --mode train `
        --arch resnet_cifar `
        --cfg resnet56 `
        --data_set cifar10 `
        --data_path ./data `
        --epochs 10 `
        --train_batch_size 64 `
        --lr 0.01 `
        --gpus 0 `
        --save_dir ./models `
        --num_workers 2 `
        --resume ./models/checkpoint_epoch_10.pth

    # 从检查点恢复训练
    python train_test.py --mode train `
        --arch resnet_cifar `
        --cfg resnet56 `
        --data_set cifar10 `
        --data_path ./data `
        --epochs 10 `
        --resume ./models/checkpoint_epoch_10.pth `
        --gpus 0

    # 测试模型
    python train_test.py --mode test `
        --arch resnet_cifar `
        --cfg resnet56 `
        --data_set cifar10 `
        --data_path ./data `
        --model_path ./pretrain/resnet56_cifar10.pth `
        --gpus 0 `
        --eval_batch_size 64 `
        --num_workers 2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from importlib import import_module
from pathlib import Path

# 导入数据加载模块
from data import cifar10, cifar100, imagenet


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='简单的训练和测试脚本')

    # 模式选择
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train 或 test')

    # 模型配置
    parser.add_argument('--arch', type=str, default='resnet_cifar',
                        choices=['resnet_cifar', 'resnet', 'vgg_cifar', 'vgg', 'googlenet', 'densenet'],
                        help='模型架构')
    parser.add_argument('--cfg', type=str, default='resnet56',
                        help='模型配置 (resnet56, resnet110, vgg16等)')

    # 数据集配置
    parser.add_argument('--data_set', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='数据集名称')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='数据集路径')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='训练批次大小')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='测试批次大小')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--lr_decay_step', type=int, nargs='+', default=[50, 100],
                        help='学习率衰减的epoch')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载的worker数量')

    # GPU配置
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='使用的GPU ID列表')

    # 保存/加载路径
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='模型保存目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='测试时使用的模型路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练的路径')

    args = parser.parse_args()
    return args


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    acc = AverageMeter()

    print(f'\n==> Epoch {epoch} 训练中...')
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录指标
        losses.update(loss.item(), inputs.size(0))
        prec1 = accuracy(outputs, targets)[0]
        acc.update(prec1.item(), inputs.size(0))

        # 每50个batch打印一次
        if (batch_idx + 1) % 50 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {losses.avg:.4f} | Acc: {acc.avg:.2f}%')

    elapsed_time = time.time() - start_time
    print(f'==> Epoch {epoch} 完成 | 时间: {elapsed_time:.2f}s | '
          f'平均Loss: {losses.avg:.4f} | 平均Acc: {acc.avg:.2f}%')

    return losses.avg, acc.avg


def test(model, test_loader, criterion, device):
    """在测试集上评估模型"""
    model.eval()
    losses = AverageMeter()
    acc = AverageMeter()

    print('\n==> 测试中...')
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 记录指标
            losses.update(loss.item(), inputs.size(0))
            prec1 = accuracy(outputs, targets)[0]
            acc.update(prec1.item(), inputs.size(0))

    elapsed_time = time.time() - start_time
    print(f'==> 测试完成 | 时间: {elapsed_time:.2f}s | '
          f'Loss: {losses.avg:.4f} | Acc: {acc.avg:.2f}%\n')

    return losses.avg, acc.avg


def train_model(args):
    """训练模型主函数"""
    print('='*60)
    print('开始训练模型')
    print('='*60)

    # 设置设备
    device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载数据集
    print(f'\n==> 加载数据集: {args.data_set}')
    if args.data_set == 'cifar10':
        loader = cifar10.Data(args)
    elif args.data_set == 'cifar100':
        loader = cifar100.Data(args)
    elif args.data_set == 'imagenet':
        loader = imagenet.Data(args)
    else:
        raise ValueError(f'不支持的数据集: {args.data_set}')

    # 创建模型
    print(f'==> 创建模型: {args.arch} - {args.cfg}')
    if args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    elif args.arch == 'resnet':
        model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    elif args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    elif args.arch == 'vgg':
        model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet().to(device)
    elif args.arch == 'densenet':
        model = import_module(f'model.{args.arch}').densenet().to(device)
    else:
        raise ValueError(f'不支持的模型架构: {args.arch}')

    # 多GPU支持
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
        print(f'使用多GPU训练: {args.gpus}')

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.lr_decay_step,
                                               gamma=0.1)

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环初始化
    start_epoch = 1
    best_acc = 0.0
    best_epoch = 0

    # 检查是否需要从检查点恢复训练
    if args.resume:
        if os.path.exists(args.resume):
            print(f'==> 从检查点恢复训练: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)

            # 恢复模型权重
            if len(args.gpus) > 1:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])

            # 恢复优化器状态
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('==> 优化器状态已恢复')

            # 恢复学习率调度器状态（如果有）
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print('==> 学习率调度器状态已恢复')

            # 恢复起始epoch
            start_epoch = checkpoint['epoch']

            # 恢复最佳准确率（如果有）
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                print(f'==> 历史最佳准确率: {best_acc:.2f}%')

            print(f'==> 从 epoch {start_epoch} 继续训练')
        else:
            raise FileNotFoundError(f'检查点文件不存在: {args.resume}')

    print(f'==> 开始训练 (Epoch {start_epoch} → {args.epochs})')
    for epoch in range(start_epoch, args.epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, loader.trainLoader, criterion, optimizer, device, epoch
        )

        # 测试
        test_loss, test_acc = test(model, loader.testLoader, criterion, device)

        # 打印本epoch的完整统计信息
        print(f'\n{"="*60}')
        print(f'Epoch [{epoch}/{args.epochs}] Summary:')
        print(f'  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'  Test  - Loss: {test_loss:.4f}  | Acc: {test_acc:.2f}%')
        print(f'{"="*60}')

        # 学习率衰减
        scheduler.step()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            model_state = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

            save_path = save_dir / f'best_{args.arch}_{args.cfg}_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'cfg': args.cfg,
                'state_dict': model_state,
                'best_acc': best_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'optimizer': optimizer.state_dict(),
            }, save_path)
            print(f'\n保存最佳模型 (Epoch {epoch}, Test Acc: {best_acc:.2f}%) 到 {save_path}')

        # 定期保存检查点
        if epoch % 5 == 0:
            model_state = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'cfg': args.cfg,
                'state_dict': model_state,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, checkpoint_path)
            print(f'保存检查点到 {checkpoint_path}')

    print('\n' + '='*60)
    print(f'训练完成!')
    print(f'最佳准确率: {best_acc:.2f}% (Epoch {best_epoch})')
    print(f'最佳模型保存在: {save_dir / f"best_{args.arch}_{args.cfg}_epoch{best_epoch}.pth"}')
    print('='*60)


def test_model(args):
    """测试模型主函数"""
    print('='*60)
    print('测试模型')
    print('='*60)

    # 检查模型路径
    if args.model_path is None:
        raise ValueError('测试模式需要指定 --model_path')

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'模型文件不存在: {args.model_path}')

    # 设置设备
    device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载数据集
    print(f'\n==> 加载数据集: {args.data_set}')
    if args.data_set == 'cifar10':
        loader = cifar10.Data(args)
    elif args.data_set == 'cifar100':
        loader = cifar100.Data(args)
    elif args.data_set == 'imagenet':
        loader = imagenet.Data(args)
    else:
        raise ValueError(f'不支持的数据集: {args.data_set}')

    # 加载模型
    print(f'==> 加载模型: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)

    # 获取模型配置
    if 'arch' in checkpoint and 'cfg' in checkpoint:
        arch = checkpoint['arch']
        cfg = checkpoint['cfg']
        print(f'模型架构: {arch} - {cfg}')
    else:
        arch = args.arch
        cfg = args.cfg
        print(f'使用命令行指定的架构: {arch} - {cfg}')

    # 创建模型
    if arch == 'resnet_cifar':
        model = import_module(f'model.{arch}').resnet(cfg).to(device)
    elif arch == 'resnet':
        model = import_module(f'model.{arch}').resnet(cfg).to(device)
    elif arch == 'vgg_cifar':
        model = import_module(f'model.{arch}').VGG(cfg).to(device)
    elif arch == 'vgg':
        model = import_module(f'model.{arch}').VGG(cfg).to(device)
    elif arch == 'googlenet':
        model = import_module(f'model.{arch}').googlenet().to(device)
    elif arch == 'densenet':
        model = import_module(f'model.{arch}').densenet().to(device)
    else:
        raise ValueError(f'不支持的模型架构: {arch}')

    # 加载权重
    model.load_state_dict(checkpoint['state_dict'])

    if 'best_acc' in checkpoint:
        print(f'模型训练时的最佳准确率: {checkpoint["best_acc"]:.2f}%')
    if 'epoch' in checkpoint:
        print(f'模型训练的epoch: {checkpoint["epoch"]}')

    # 多GPU支持
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 测试
    test_loss, test_acc = test(model, loader.testLoader, criterion, device)

    print('='*60)
    print(f'测试结果: 准确率 = {test_acc:.2f}%')
    print('='*60)


def main():
    """主函数"""
    args = get_args()

    print(f'\n当前配置:')
    for arg, value in vars(args).items():
        print(f'  {arg}: {value}')
    print()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    else:
        raise ValueError(f'不支持的模式: {args.mode}')


if __name__ == '__main__':
    main()
