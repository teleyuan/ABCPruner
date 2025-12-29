"""
PyTorch模型保存和加载方式演示

展示两种主要的模型保存方式及其区别：
1. 只保存state_dict（推荐）
2. 保存完整模型对象（不推荐）

以及对应的加载方式
"""

import torch
import torch.nn as nn
import os


# 定义一个简单的示例模型
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def demo_save_methods():
    """演示不同的保存方式"""
    print("="*80)
    print("PyTorch模型保存方式演示")
    print("="*80)

    # 创建并初始化模型
    model = SimpleModel(input_size=10, hidden_size=20, output_size=5)

    # 创建优化器和其他训练状态
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch = 100
    best_acc = 95.5

    # 创建测试输入
    test_input = torch.randn(1, 10)
    test_output = model(test_input)

    print(f"\n原始模型输出: {test_output}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")


    # ==================== 方式1: 只保存state_dict（推荐） ====================
    print("\n" + "="*80)
    print("方式1: 只保存state_dict（推荐方式）")
    print("="*80)

    # 1.1 最简单：只保存模型权重
    torch.save(model.state_dict(), 'model_state_dict_only.pth')
    print("\n✓ 保存: model_state_dict_only.pth")
    print("  内容: 只包含模型权重参数")
    print("  优点: 文件小、灵活、兼容性好")
    print("  缺点: 加载时需要先创建模型结构")

    # 1.2 保存完整训练状态（常用）
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'model_config': {
            'input_size': 10,
            'hidden_size': 20,
            'output_size': 5
        }
    }
    torch.save(checkpoint, 'model_checkpoint.pth')
    print("\n✓ 保存: model_checkpoint.pth")
    print("  内容: 模型权重 + 优化器状态 + epoch + 其他信息")
    print("  优点: 可以从任意epoch恢复训练")
    print("  使用场景: 训练中断后继续训练")


    # ==================== 方式2: 保存完整模型（不推荐） ====================
    print("\n" + "="*80)
    print("方式2: 保存完整模型对象（不推荐）")
    print("="*80)

    torch.save(model, 'model_entire.pth')
    print("\n✓ 保存: model_entire.pth")
    print("  内容: 完整的模型对象（包括结构和权重）")
    print("  优点: 加载简单，不需要预先定义模型")
    print("  缺点: ")
    print("    - 文件更大")
    print("    - 依赖模型类定义，代码改变可能导致加载失败")
    print("    - 不够灵活，难以部署到其他框架")
    print("    - PyTorch官方不推荐")


    # ==================== .pt vs .pth 后缀 ====================
    print("\n" + "="*80)
    print("关于文件后缀 .pt vs .pth")
    print("="*80)

    # 演示两种后缀完全等价
    torch.save(model.state_dict(), 'model_weights.pt')
    torch.save(model.state_dict(), 'model_weights.pth')

    print("\n.pt 和 .pth 在PyTorch中完全等价！")
    print("  - .pt  : PyTorch的传统后缀")
    print("  - .pth : 更直观地表示'PyTorch'")
    print("  - 两者可以互换使用，没有任何区别")
    print("  - 社区习惯: .pth 更常用")

    # 比较文件大小
    import os
    size1 = os.path.getsize('model_state_dict_only.pth') / 1024
    size2 = os.path.getsize('model_checkpoint.pth') / 1024
    size3 = os.path.getsize('model_entire.pth') / 1024

    print("\n" + "="*80)
    print("文件大小对比")
    print("="*80)
    print(f"model_state_dict_only.pth : {size1:.2f} KB (只有权重)")
    print(f"model_checkpoint.pth      : {size2:.2f} KB (权重+训练状态)")
    print(f"model_entire.pth          : {size3:.2f} KB (完整模型对象)")

    return model, test_input, test_output


def demo_load_methods(original_model, test_input, original_output):
    """演示不同的加载方式"""
    print("\n" + "="*80)
    print("PyTorch模型加载方式演示")
    print("="*80)


    # ==================== 加载方式1: 加载state_dict ====================
    print("\n方式1: 加载state_dict")
    print("-"*80)

    # 1.1 加载只包含权重的文件
    print("\n1.1 从 model_state_dict_only.pth 加载")
    model1 = SimpleModel(input_size=10, hidden_size=20, output_size=5)  # 必须先创建模型
    state_dict = torch.load('model_state_dict_only.pth')
    model1.load_state_dict(state_dict)
    model1.eval()

    output1 = model1(test_input)
    print(f"  加载后输出: {output1}")
    print(f"  与原始输出一致: {torch.allclose(output1, original_output)}")

    # 1.2 加载checkpoint（包含训练状态）
    print("\n1.2 从 model_checkpoint.pth 加载")
    checkpoint = torch.load('model_checkpoint.pth')

    # 恢复模型
    model2 = SimpleModel(
        input_size=checkpoint['model_config']['input_size'],
        hidden_size=checkpoint['model_config']['hidden_size'],
        output_size=checkpoint['model_config']['output_size']
    )
    model2.load_state_dict(checkpoint['model_state_dict'])

    # 恢复优化器（用于继续训练）
    optimizer = torch.optim.Adam(model2.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 恢复其他状态
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']

    model2.eval()
    output2 = model2(test_input)
    print(f"  加载后输出: {output2}")
    print(f"  与原始输出一致: {torch.allclose(output2, original_output)}")
    print(f"  恢复的训练状态: Epoch={epoch}, Best Acc={best_acc}")


    # ==================== 加载方式2: 加载完整模型 ====================
    print("\n" + "-"*80)
    print("方式2: 加载完整模型")
    print("-"*80)

    print("\n从 model_entire.pth 加载")
    model3 = torch.load('model_entire.pth')  # 直接加载，不需要预先定义模型
    model3.eval()

    output3 = model3(test_input)
    print(f"  加载后输出: {output3}")
    print(f"  与原始输出一致: {torch.allclose(output3, original_output)}")
    print("\n  注意: 这种方式虽然简单，但不推荐！")
    print("        如果模型类定义改变，可能无法加载")


    # ==================== 处理不同格式的checkpoint ====================
    print("\n" + "="*80)
    print("处理不同格式的checkpoint（通用加载代码）")
    print("="*80)

    def load_model_universal(checkpoint_path, model):
        """
        通用的模型加载函数，处理各种checkpoint格式

        参数:
            checkpoint_path: checkpoint文件路径
            model: 已创建的模型实例

        返回:
            model: 加载权重后的模型
            info: checkpoint中的其他信息（如epoch、accuracy等）
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        info = {}

        # 情况1: checkpoint是dict，包含'state_dict'键
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 提取其他信息
            info['epoch'] = checkpoint.get('epoch', None)
            info['best_acc'] = checkpoint.get('best_acc', None)
            info['optimizer'] = checkpoint.get('optimizer_state_dict', None)

        # 情况2: checkpoint是dict，包含'model'或'model_state_dict'键
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            info['epoch'] = checkpoint.get('epoch', None)
            info['best_acc'] = checkpoint.get('best_acc', None)

        # 情况3: checkpoint直接就是state_dict
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint

        # 情况4: checkpoint是完整模型对象
        else:
            return checkpoint, {}

        # 处理DataParallel保存的模型（权重键带'module.'前缀）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉'module.'前缀
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        return model, info

    print("\n通用加载示例:")
    model_test = SimpleModel(input_size=10, hidden_size=20, output_size=5)
    model_test, info = load_model_universal('model_checkpoint.pth', model_test)
    print(f"  加载成功!")
    print(f"  附加信息: {info}")


    # ==================== 最佳实践建议 ====================
    print("\n" + "="*80)
    print("最佳实践建议")
    print("="*80)

    practices = """
1. 保存方式：
   ✓ 推荐: 使用 torch.save(state_dict, 'model.pth')
   ✗ 避免: 使用 torch.save(model, 'model.pth')

2. 文件后缀：
   - .pth 和 .pt 完全等价，推荐使用 .pth
   - 保持项目内一致性即可

3. 保存内容：
   训练中: 保存完整checkpoint（包含epoch、optimizer等）
   {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'best_acc': best_acc,
       'scheduler_state_dict': scheduler.state_dict()  # 如果有学习率调度器
   }

   部署用: 只保存 model.state_dict()

4. 加载代码：
   - 总是先创建模型实例
   - 使用 map_location 参数处理不同设备
   - 处理 DataParallel 的 'module.' 前缀
   - 兼容不同的checkpoint格式

5. 版本兼容：
   - 在checkpoint中保存模型配置信息
   - 记录PyTorch版本号
   - 避免依赖模型类的序列化
"""
    print(practices)


def cleanup():
    """清理生成的示例文件"""
    files = [
        'model_state_dict_only.pth',
        'model_checkpoint.pth',
        'model_entire.pth',
        'model_weights.pt',
        'model_weights.pth'
    ]

    print("\n" + "="*80)
    print("清理演示文件")
    print("="*80)

    for f in files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  已删除: {f}")

    print("\n演示文件已清理完毕")


if __name__ == '__main__':
    # 运行演示
    model, test_input, test_output = demo_save_methods()
    demo_load_methods(model, test_input, test_output)

    # 询问是否清理
    print("\n" + "="*80)
    cleanup_choice = input("是否删除演示生成的文件? (y/n): ")
    if cleanup_choice.lower() == 'y':
        cleanup()
    else:
        print("保留演示文件，你可以查看它们")

    print("\n" + "="*80)
    print("演示完成!")
    print("="*80)
