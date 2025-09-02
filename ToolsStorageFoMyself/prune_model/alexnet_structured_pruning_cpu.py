import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 1. 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. 数据准备
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将MNIST图像调整为224x224
    transforms.Grayscale(num_output_channels=3),  # 转换为3通道图像
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print("data loader over!")
# # 3. 模型初始化
# model = AlexNet(num_classes=10)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# # 4. 训练模型
# def train_model(model, train_loader, criterion, optimizer, epochs=1):
#     for epoch in range(epochs):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# train_model(model, train_loader, criterion, optimizer)

# # 5. 结构化剪枝操作（剪除整个卷积核或通道）
# def structured_prune_model(model, pruning_percentage=0.2):
#     """
#     对模型进行结构化剪枝，剪掉整个卷积核或通道
#     :param model: 待剪枝的模型
#     :param pruning_percentage: 剪枝比例，剪去相应比例的卷积核
#     """
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             print(f"Pruning Conv2d Layer: {name}")
            
#             # 根据L1范数剪枝，剪去卷积核
#             # 获取当前卷积层的权重
#             weight = module.weight.data
#             # 计算每个卷积核的L1范数
#             l1_norm = torch.sum(torch.abs(weight), dim=(1,2,3))  # 每个卷积核的L1范数
            
#             # 排序并选择最小的卷积核进行剪枝
#             num_kernels_to_prune = int(pruning_percentage * weight.shape[0])
#             _, prune_indices = torch.topk(l1_norm, num_kernels_to_prune, largest=False)
            
#             # 将选择的卷积核的权重置为零
#             for idx in prune_indices:
#                 weight[idx] = 0  # 剪掉整个卷积核
                
#             # 更新卷积层的权重
#             module.weight.data = weight

# # 6. 保存剪枝前模型
# def save_model(model, filename):
#     torch.save(model.state_dict(), filename)
#     print(f"Model saved to {filename}")

# # 7. 保存剪枝后的模型
# def save_pruned_model(model, filename):
#     torch.save(model.state_dict(), filename)
#     print(f"Pruned model saved to {filename}")

# # 8. 保存剪枝前的模型
# save_model(model, 'alexnet_before_pruning.pth')

# # 执行剪枝
# structured_prune_model(model, pruning_percentage=0.2)

# # 9. 保存剪枝后的模型
# save_pruned_model(model, 'alexnet_after_pruning.pth')

# # 10. 验证剪枝后模型
# def evaluate_model(model, test_loader):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             outputs = model(data)
#             _, predicted = torch.max(outputs.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
#     print(f'Accuracy after pruning: {100 * correct / total:.2f}%')

# # 11. 测试数据准备
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# evaluate_model(model, test_loader)
