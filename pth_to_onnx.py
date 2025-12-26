"""
PyTorch模型(.pth)转ONNX格式转换工具

功能:
1. 支持多种网络架构(VGG, ResNet, GoogLeNet, DenseNet)
2. 支持剪枝后的模型(带Honey编码)
3. 自动检测输入尺寸(CIFAR10/ImageNet)
4. 验证转换后的ONNX模型正确性

使用示例:
python pth_to_onnx.py `
    --pth_path models/best_resnet_cifar_resnet56_epoch15.pth `
    --onnx_path models/resnet56_cifar.onnx `
    --arch resnet_cifar `
    --cfg resnet56 `
    --data_set cifar10 `
    --honey

"""

import torch
import torch.nn as nn
import argparse
import os
from importlib import import_module
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch模型转ONNX格式')

    # 模型路径参数
    parser.add_argument(
        '--pth_path',
        type=str,
        required=True,
        help='输入的.pth模型文件路径')

    parser.add_argument(
        '--onnx_path',
        type=str,
        default=None,
        help='输出的.onnx文件路径(默认: 与pth同名同目录)')

    # 模型架构参数
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet_cifar',
        choices=('vgg_cifar', 'resnet_cifar', 'vgg', 'resnet', 'densenet', 'googlenet'),
        help='网络架构类型')

    parser.add_argument(
        '--cfg',
        type=str,
        default='resnet56',
        help='具体的网络配置(如vgg16, resnet56, resnet50等)')

    parser.add_argument(
        '--data_set',
        type=str,
        default='cifar10',
        choices=('cifar10', 'imagenet'),
        help='数据集名称(决定输入尺寸)')

    parser.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='分类数量(默认: cifar10=10, imagenet=1000)')

    # 剪枝参数
    parser.add_argument(
        '--honey',
        type=str,
        default=None,
        help='蜜蜂编码（剪枝配置），格式: "5,5,5,5,5,5" (如果模型是剪枝后的，必须提供)')

    parser.add_argument(
        '--depth',
        type=int,
        default=None,
        help='网络深度(用于某些架构)')

    # ONNX导出参数
    parser.add_argument(
        '--dynamic_axes',
        action='store_true',
        help='是否使用动态batch size')

    parser.add_argument(
        '--simplify',
        action='store_true',
        help='是否简化ONNX模型(需要安装onnx-simplifier)')

    parser.add_argument(
        '--verify',
        action='store_true',
        default=False,
        help='是否验证ONNX模型输出一致性')

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出模型结构')

    return parser.parse_args()


def load_model(args, device):
    """
    根据参数加载PyTorch模型

    Returns:
        model: 加载好权重的PyTorch模型
        input_size: 输入图像尺寸
    """
    print('==> Building model...')

    # 确定类别数
    if args.num_classes is None:
        num_classes = 10 if args.data_set == 'cifar10' else 1000
    else:
        num_classes = args.num_classes

    # 解析honey编码
    honey = None
    if args.honey is not None:
        honey = list(map(int, args.honey.split(',')))
        print(f'使用Honey编码: {honey}')

    # 根据架构创建模型
    if args.arch == 'vgg_cifar':
        if honey is None:
            model = import_module(f'model.{args.arch}').VGG(args.cfg, num_classes=num_classes)
        else:
            model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=honey, num_classes=num_classes)

    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=honey, num_classes=num_classes)

    elif args.arch == 'vgg':
        if honey is None:
            model = import_module(f'model.{args.arch}').VGG(num_classes=num_classes)
        else:
            model = import_module(f'model.{args.arch}').BeeVGG(honeysource=honey, num_classes=num_classes)

    elif args.arch == 'resnet':
        model = import_module(f'model.{args.arch}').resnet(args.cfg, honey=honey, num_classes=num_classes)

    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(honey=honey, num_classes=num_classes)

    elif args.arch == 'densenet':
        model = import_module(f'model.{args.arch}').densenet(honey=honey, num_classes=num_classes)

    else:
        raise ValueError(f'不支持的架构: {args.arch}')

    model = model.to(device)

    # 加载权重
    print(f'==> Loading checkpoint from {args.pth_path}...')
    if not os.path.exists(args.pth_path):
        raise FileNotFoundError(f'找不到模型文件: {args.pth_path}')

    checkpoint = torch.load(args.pth_path, map_location=device)

    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Checkpoint信息: Epoch={checkpoint.get('epoch', 'N/A')}, Acc={checkpoint.get('best_acc', 'N/A')}")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 处理DataParallel保存的模型(key包含'module.'前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉'module.'前缀
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # 确定输入尺寸
    input_size = 32 if args.data_set == 'cifar10' else 224

    print(f'\n模型加载成功: {args.cfg}')
    print(f'  - 类别数: {num_classes}')
    print(f'  - 输入尺寸: {input_size}x{input_size}')

    if args.verbose:
        print(f'\n模型结构:')
        print(model)

    return model, input_size


def export_to_onnx(model, input_size, onnx_path, args):
    """
    将PyTorch模型导出为ONNX格式

    Args:
        model: PyTorch模型
        input_size: 输入图像尺寸
        onnx_path: 输出ONNX文件路径
        args: 命令行参数
    """
    print('\n==> Exporting to ONNX...')

    # 创建dummy输入
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # 设置输入输出名称
    input_names = ['input']
    output_names = ['output']

    # 动态轴设置(支持动态batch size)
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        print('启用动态batch size')

    # 导出ONNX (使用PyTorch推荐的默认opset版本)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    # 清理可能生成的.data文件
    data_file = onnx_path + '.data'
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f'  - 已删除临时文件: {data_file}')

    print(f'ONNX模型已导出: {onnx_path}')

    # 获取文件大小
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f'  - 文件大小: {file_size:.2f} MB')

    return dummy_input


def verify_onnx(model, onnx_path, dummy_input, device):
    """
    验证ONNX模型输出与PyTorch模型输出的一致性

    Args:
        model: PyTorch模型
        onnx_path: ONNX模型路径
        dummy_input: 测试输入
        device: 设备
    """
    print('\n==> Verifying ONNX model...')

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print('需要安装onnx和onnxruntime来验证模型')
        print('  pip install onnx onnxruntime')
        return

    # 检查ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX模型格式检查通过')

    # 使用ONNX Runtime推理
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # PyTorch推理
    with torch.no_grad():
        torch_output = model(dummy_input).cpu().numpy()

    # ONNX推理
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # 比较输出
    max_diff = np.max(np.abs(torch_output - ort_output))
    mean_diff = np.mean(np.abs(torch_output - ort_output))

    print(f'输出一致性验证:')
    print(f'  - 最大误差: {max_diff:.6f}')
    print(f'  - 平均误差: {mean_diff:.6f}')

    if max_diff < 1e-5:
        print('  - 状态: ✓ 优秀 (误差 < 1e-5)')
    elif max_diff < 1e-3:
        print('  - 状态: ✓ 良好 (误差 < 1e-3)')
    else:
        print(f'  - 状态: ⚠ 警告 (误差较大: {max_diff})')

    # 输出模型信息
    print(f'\n==> ONNX模型信息:')
    print(f'  - IR版本: {onnx_model.ir_version}')
    print(f'  - 生产者: {onnx_model.producer_name}')
    print(f'  - 输入: {[inp.name for inp in ort_session.get_inputs()]}')
    print(f'  - 输出: {[out.name for out in ort_session.get_outputs()]}')
    print(f'  - 输入形状: {ort_session.get_inputs()[0].shape}')
    print(f'  - 输出形状: {ort_session.get_outputs()[0].shape}')


def simplify_onnx(onnx_path):
    """
    简化ONNX模型(可选)

    Args:
        onnx_path: ONNX模型路径
    """
    print('\n==> Simplifying ONNX model...')

    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print('需要安装onnx-simplifier来简化模型')
        print('  pip install onnx-simplifier')
        return

    # 加载模型
    onnx_model = onnx.load(onnx_path)

    # 简化
    model_simplified, check = simplify(onnx_model)

    if check:
        # 保存简化后的模型
        simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
        onnx.save(model_simplified, simplified_path)

        # 比较文件大小
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        simplified_size = os.path.getsize(simplified_path) / (1024 * 1024)

        print(f'简化成功:')
        print(f'  - 原始模型: {original_size:.2f} MB')
        print(f'  - 简化模型: {simplified_size:.2f} MB')
        print(f'  - 压缩率: {(1 - simplified_size/original_size)*100:.2f}%')
        print(f'  - 保存路径: {simplified_path}')
    else:
        print('简化验证失败')


def main():
    args = parse_args()

    print('=' * 80)
    print('PyTorch模型转ONNX工具')
    print('=' * 80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 确定输出路径
    if args.onnx_path is None:
        # 默认: 与pth同名同目录
        base_path = os.path.splitext(args.pth_path)[0]
        args.onnx_path = base_path + '.onnx'

    # 创建输出目录
    output_dir = os.path.dirname(args.onnx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'创建输出目录: {output_dir}')

    try:
        # 1. 加载模型
        model, input_size = load_model(args, device)

        # 2. 导出ONNX
        dummy_input = export_to_onnx(model, input_size, args.onnx_path, args)

        # 3. 验证模型
        if args.verify:
            verify_onnx(model, args.onnx_path, dummy_input, device)

        # 4. 简化模型(可选)
        if args.simplify:
            simplify_onnx(args.onnx_path)

        print('\n' + '=' * 80)
        print('转换完成!')
        print('=' * 80)
        print(f'输出文件: {args.onnx_path}')

    except Exception as e:
        print(f'\n转换失败: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
