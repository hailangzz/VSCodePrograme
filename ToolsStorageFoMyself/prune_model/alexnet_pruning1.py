import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models

# 注：非结构化剪枝不改变模型的参数量，只是对模型的权重参数做了0值掩码···
# 注：非结构化剪枝不改变模型的参数量，只是对模型的权重参数做了0值掩码···
# 注：非结构化剪枝不改变模型的参数量，只是对模型的权重参数做了0值掩码···

# 1. 加载预训练的 AlexNet 模型
model = models.alexnet(pretrained=True)

# 2. 打印模型结构，查看各层信息
print(model)

print("\n每层剪枝前的稀疏度：")
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        # 获取每层的权重
        weight = module.weight
        num_elements = weight.numel()  # 权重元素总数
        num_zeros = (weight == 0).sum().item()  # 计算零元素数量
        sparsity = num_zeros / num_elements  # 稀疏度 = 0 元素占比
        print(f"{name}: 稀疏度 = {sparsity:.4f}")


# 3. 定义一个剪枝的函数
def prune_model(model, pruning_percentage=0.2):
    """
    对模型进行剪枝：对卷积层和全连接层的权重进行剪枝
    :param model: 待剪枝的模型
    :param pruning_percentage: 剪枝比例
    :return: 剪枝后的模型
    """
    # 对卷积层进行剪枝
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Pruning Conv2d Layer: {name}")
            prune.l1_unstructured(module, name="weight", amount=pruning_percentage)
            # 直接移除剪枝掩码，减少参数
            prune.remove(module, 'weight')

        # 对全连接层进行剪枝
        if isinstance(module, nn.Linear):
            print(f"Pruning Linear Layer: {name}")
            prune.l1_unstructured(module, name="weight", amount=pruning_percentage)
            prune.remove(module, 'weight')

    print("剪枝后模型：",model)

    return model

# 4. 保存剪枝前的模型（创建模型副本）
model_before_pruning = models.alexnet(pretrained=True)  # 重新加载原始模型
torch.save(model_before_pruning.state_dict(), "alexnet_before_pruning.pth")
print("\n剪枝前的模型已保存：alexnet_before_pruning.pth")

# 5. 对模型进行剪枝
pruned_model = prune_model(model, pruning_percentage=0.2)

# 6. 保存剪枝后的模型
torch.save(pruned_model.state_dict(), "alexnet_after_pruning.pth")
print("\n剪枝后的模型已保存：alexnet_after_pruning.pth")

# 7. 打印剪枝后的模型结构
print("\n剪枝后的模型：")
print(pruned_model)

# 8. 打印每层的稀疏度（剪枝效果）
print("\n每层剪枝后的稀疏度：")
for name, module in pruned_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        # 获取每层的权重
        weight = module.weight
        num_elements = weight.numel()  # 权重元素总数
        num_zeros = (weight == 0).sum().item()  # 计算零元素数量
        sparsity = num_zeros / num_elements  # 稀疏度 = 0 元素占比
        print(f"{name}: 稀疏度 = {sparsity:.4f}")

# 9. 测试剪枝后的模型输出（使用随机输入）
input_data = torch.randn(1, 3, 224, 224)  # AlexNet 输入尺寸 (batch_size, 3, 224, 224)
output_data = pruned_model(input_data)

print("\n剪枝后的输出结果:")
# print(output_data)

# 注：非结构化剪枝不改变模型的参数量，只是对模型的权重参数做了0值掩码···
# 注：非结构化剪枝不改变模型的参数量，只是对模型的权重参数做了0值掩码···