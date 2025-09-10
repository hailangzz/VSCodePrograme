import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import numpy as np

from nni.compression.distillation import Distillation

# -----------------------------
# Dummy dataset
# -----------------------------
class RandomDataset(Dataset):
    def __init__(self, n=100, img_size=640):
        self.n = n
        self.img_size = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = np.random.rand(3, self.img_size, self.img_size).astype(np.float32)
        y = np.zeros((1, 6), dtype=np.float32)  # [cls, x, y, w, h, conf]
        return torch.from_numpy(x), torch.from_numpy(y)

def get_dataloader(n=20):
    return DataLoader(RandomDataset(n=n), batch_size=2, shuffle=True)

# -----------------------------
# 蒸馏示例
# -----------------------------
def distillation_demo():
    # Teacher model (大模型)
    teacher = YOLO("yolov5s.pt").model
    teacher.eval()

    # Student model (小模型，可以是剪枝或轻量化模型)
    student = YOLO("yolov5s.pt").model

    # 蒸馏配置
    config_list = [{
        'distill_type': 'logits',   # 使用 logits 蒸馏
        'temperature': 4.0,
        'alpha': 0.7,               # KD loss 权重
        'beta': 0.3                 # 原始任务 loss 权重
    }]

    # 构建 distiller
    distiller = Distillation(student_model=student, teacher_model=teacher, config_list=config_list)

    # 返回可训练的 student 模型
    student_model = distiller.compress()
    return student_model

# -----------------------------
# Training loop
# -----------------------------
def train_student(student_model, dataloader, epochs=1):
    optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.MSELoss()  # 这里只是示例，真实用 YOLO loss

    student_model.train()
    for epoch in range(epochs):
        for imgs, labels in dataloader:
            preds = student_model(imgs)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            loss = criterion(preds, torch.zeros_like(preds))  # dummy loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[KD] Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

# -----------------------------
# Main
# -----------------------------
def main():
    dataloader = get_dataloader(n=20)
    student_model = distillation_demo()
    train_student(student_model, dataloader, epochs=2)
    torch.save(student_model.state_dict(), "yolov5s_student_kd.pth")
    print("[Done] Knowledge Distillation finished.")

if __name__ == "__main__":
    main()
