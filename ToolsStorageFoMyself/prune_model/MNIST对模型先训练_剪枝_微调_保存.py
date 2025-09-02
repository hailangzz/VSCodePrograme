import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# 1. 设置设备
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# 2. 定义 AlexNet
# -----------------------------
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
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


# -----------------------------
# 3. 训练函数
# -----------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# -----------------------------
# 4. 验证函数
# -----------------------------
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc


# -----------------------------
# 5. 结构化剪枝函数
# -----------------------------
def prune_conv_layer(conv, prune_ratio):
    weight = conv.weight.data
    l1_norm = torch.sum(torch.abs(weight), dim=(1,2,3))
    num_kernels_to_prune = int(prune_ratio * weight.shape[0])
    if num_kernels_to_prune == 0:
        return conv, list(range(conv.out_channels))

    _, prune_indices = torch.topk(l1_norm, num_kernels_to_prune, largest=False)
    keep_indices = [i for i in range(conv.out_channels) if i not in prune_indices]

    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=len(keep_indices),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None)
    )
    new_conv.weight.data = weight[keep_indices].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_indices].clone()

    return new_conv, keep_indices


def structured_prune_alexnet(model, prune_ratio=0.2):
    old_features = list(model.features)  #获取原始模型神经元列表
    new_features = []                    #创建新模型神经元列表
    prev_kept = None                     #上一层保留的卷积层通道索引列表

    for layer in old_features:
        if isinstance(layer, nn.Conv2d): #只对卷积层做通道剪枝操作
            if prev_kept is not None:    #表示当上一层卷积层存在剪枝操作时，预示着当前卷积层的输入通道数要对应修改
                layer.in_channels = len(prev_kept) 
                layer.weight.data = layer.weight.data[:, prev_kept, :, :].clone() #当前卷积层，只保留剪枝后的卷积通道权值

            layer, kept = prune_conv_layer(layer, prune_ratio) #实时每个卷积层Conv2的剪枝操作
            prev_kept = kept                                   #获取当前卷积层的保留通道数，作为下一层卷积的输入通道数
        new_features.append(layer)

    model.features = nn.Sequential(*new_features)

    # 修改全连接层输入特征数
    last_conv_out_channels = len(prev_kept)
    model.classifier[0] = nn.Linear(last_conv_out_channels * 3 * 3, 4096)
    return model


# -----------------------------
# 6. 数据加载
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)


# -----------------------------
# 7. 主流程
# -----------------------------
if __name__ == "__main__":
    # 初始化模型
    model = AlexNet(num_classes=10).to(device)

    # -------- 训练原始模型 --------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print("Training original model...")
    train_model(model, train_loader, criterion, optimizer, epochs=1)
    evaluate_model(model, test_loader)

    # -------- 保存原始模型 --------
    torch.save(model.state_dict(), "alexnet_before_pruning.pth")
    print("Original model saved!")

    # -------- 对模型进行剪枝 --------
    print("Pruning model...")
    pruned_model = structured_prune_alexnet(model, prune_ratio=0.5).to(device)

    # -------- 微调剪枝后的模型 --------
    print("Fine-tuning pruned model...")
    optimizer_pruned = torch.optim.SGD(pruned_model.parameters(), lr=0.01)
    train_model(pruned_model, train_loader, criterion, optimizer_pruned, epochs=1)
    evaluate_model(pruned_model, test_loader)

    # -------- 保存剪枝后的模型 --------
    torch.save(pruned_model.state_dict(), "alexnet_after_pruning.pth")
    print("Pruned model saved!")
