"""
YOLOv5s Dependency-Aware Structured Pruning Demo with NNI

This script demonstrates:
1. Dependency-aware structured pruning of YOLOv5s using NNI
2. Ensures shortcut residual connections are handled safely
3. Exports the pruned model for later use

Requirements:
-------------
pip install ultralytics nni torch torchvision

Usage:
------
python yolov5s_nni_dep_prune_demo.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ultralytics import YOLO

from nni.compression.pruning import L1FilterPruner

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
        y = np.array([0, 0, 0, 10, 10], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def get_dataloader(batch_size=2, n=20):
    dataset = RandomDataset(n=n)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -----------------------------
# Dependency-aware pruning demo
# -----------------------------
def prune_demo(model, dataloader, sparsity=0.5):
    # Pruning configuration
    config_list = [{
        'sparsity': sparsity,
        'op_types': ['Conv2d'],  # prune only Conv2d
    }]

    # Dummy trainer for NNI
    def trainer(model, optimizer, criterion, epoch=1):
        model.train()
        for imgs, _ in dataloader:
            preds = model(imgs)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            loss = criterion(preds, torch.zeros_like(preds))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Dummy evaluator
    def evaluator(model):
        return 0.5  # placeholder score

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Dependency-aware pruner
    pruner = L1FilterPruner(
        model, 
        config_list, 
        optimizer=optimizer, 
        trainer=trainer, 
        evaluator=evaluator,
        dependency_aware=True  # <--- ensures residual shortcuts are safe
    )

    # Compress (prune) the model
    model = pruner.compress()
    print("[NNI-Pruning] Model has been dependency-aware pruned.")

    # Export pruned model
    pruner.export_model(model_path="yolov5s_dep_pruned.pth", mask_path="yolov5s_dep_pruned_mask.pth")
    print("[NNI-Pruning] Exported pruned model and mask.")

    return model


def main():
    print("[Load] Loading YOLOv5s pretrained model...")
    yolov5 = YOLO("yolov5s.pt").model

    dataloader = get_dataloader(n=20)

    model_pruned = prune_demo(yolov5, dataloader, sparsity=0.5)

    print("[Done] Dependency-aware pruning finished.")


if __name__ == "__main__":
    main()
