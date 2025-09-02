import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =======================
# 1. 选择设备
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =======================
# 2. 定义AlexNet模型 (适配MNIST 28x28 单通道)
# =======================
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 单通道
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出 64x14x14

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出 192x7x7

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 256x3x3
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),
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


# =======================
# 3. 模型训练
# =======================
def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} done!")


# =======================
# 4. 结构化剪枝
# =======================
def structured_prune_model(model, pruning_percentage=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Pruning Conv2d Layer: {name}")
            weight = module.weight.data
            l1_norm = torch.sum(torch.abs(weight), dim=(1, 2, 3))
            num_kernels_to_prune = int(pruning_percentage * weight.shape[0])
            _, prune_indices = torch.topk(l1_norm, num_kernels_to_prune, largest=False)
            for idx in prune_indices:
                weight[idx] = 0
            module.weight.data = weight


# =======================
# 5. 保存模型
# =======================
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def save_pruned_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Pruned model saved to {filename}")


# =======================
# 6. 测试模型
# =======================
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'Accuracy after pruning: {100 * correct / total:.2f}%')


# =======================
# 7. 主函数入口
# =======================
def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载数据
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print("Data loader over!")

    # 初始化模型、损失函数、优化器
    model = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 训练
    train_model(model, train_loader, criterion, optimizer)

    # 保存剪枝前模型
    save_model(model, 'alexnet_before_pruning.pth')

    # 剪枝
    structured_prune_model(model, pruning_percentage=0.2)

    # 保存剪枝后模型
    save_pruned_model(model, 'alexnet_after_pruning.pth')

    # 验证剪枝后模型
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
