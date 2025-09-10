"""
NNI + YOLOv5 Quantization Demo (PTQ + QAT)

This demo shows how to use Microsoft NNI's compression/quantization APIs to
apply Post-Training Quantization (PTQ) and Quantization Aware Training (QAT)
on a YOLOv5s model (PyTorch).

Notes
-----
- This file provides a runnable *template* and clear hooks where you must
  plug your dataset, dataloaders, and evaluation function (mAP calculation).
- NNI APIs used (refer to NNI docs):
    - nni.compression.quantization.PtqQuantizer
    - nni.compression.quantization.QAT_Quantizer (or other quantizers such as LsqQuantizer)
    - nni.compression.TorchEvaluator

Requirements
------------
pip install nni ultralytics torch torchvision

Usage
-----
# 1) PTQ (no training required, needs a small calibration set and an evaluator to return metric)
python nni_yolov5_quant_demo.py --mode ptq --yolov5_weights yolov5s.pt --calib_dir ./calib_images --calib_num 100

# 2) QAT (performs training-aware quantization using your training loop)
python nni_yolov5_quant_demo.py --mode qat --yolov5_weights yolov5s.pt --epochs 3 --lr 1e-4

"""

import argparse
import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# NNI imports
import nni
from nni.compression import TorchEvaluator
from nni.compression.quantization import PtqQuantizer, QAT_Quantizer

# Ultralytics YOLO helper (we will extract the .model which is a torch.nn.Module)
from ultralytics import YOLO


# ---------------------------
# Minimal dataset placeholders
# ---------------------------
class SimpleImageDataset(Dataset):
    """A tiny dataset wrapper for calibration / eval. Replace with your own COCO/YOLO dataset.

    Items returned should match what your model.forward expects. For ultralytics YOLOv5, the
    model accepts either (images, ) as a single tensor (N,3,H,W) or a list of PIL/ndarray.
    Here we prepare NCHW float tensors normalized to [0,1].
    """

    def __init__(self, img_paths, imgsz: int = 640):
        from PIL import Image
        import numpy as np

        self.paths = img_paths
        self.imgsz = imgsz
        self.Image = Image
        self.np = np

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = self.Image.open(p).convert('RGB')
        w, h = img.size
        scale = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_resized = img.resize((nw, nh), self.Image.BILINEAR)
        canvas = self.Image.new('RGB', (self.imgsz, self.imgsz), (114, 114, 114))
        top = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas.paste(img_resized, (left, top))
        x = self.np.array(canvas).astype('float32') / 255.0
        x = self.np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x)


# ---------------------------
# Helpers: build dataloaders
# ---------------------------

def make_dataloader_from_dir(img_dir: str, imgsz: int = 640, batch_size: int = 8, shuffle: bool = False):
    p = Path(img_dir)
    files = [str(x) for x in sorted(p.glob('**/*')) if x.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    ds = SimpleImageDataset(files, imgsz=imgsz)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)


# ---------------------------
# Evaluation function (required by TorchEvaluator)
# ---------------------------

def default_evaluating_func(model: torch.nn.Module, imgsz: int = 640, val_dir: Optional[str] = None) -> float:
    """A minimal evaluating function for NNI evaluator.

    IMPORTANT: This is a *placeholder*. For object detection you should compute
    a detection metric such as mAP on a proper Val dataset. Here we return a
    simple proxy metric: average number of detections per image (higher -> better)
    so that NNI can still compare fp32 vs quantized models.
    """
    model.eval()
    if val_dir is None:
        return 0.0

    dl = make_dataloader_from_dir(val_dir, imgsz=imgsz, batch_size=8, shuffle=False)
    total_dets = 0
    total_imgs = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            # ultralytics Detect model returns a special Results object when called via YOLO wrapper,
            # but when using the raw module it will output tensors. To be generic, we try calling model(batch)
            out = model(batch)
            # out may be a list of detections or a tensor; we handle common cases
            if isinstance(out, (list, tuple)):
                # assume list of tensors per image
                for o in out:
                    if isinstance(o, torch.Tensor):
                        # assume shape (N, 6) boxes (x1,y1,x2,y2,conf,class)
                        total_dets += o.shape[0]
                        total_imgs += 1
                    else:
                        total_imgs += 1
            elif isinstance(out, torch.Tensor):
                # e.g., a batched tensor where first dim is batch
                total_imgs += out.shape[0]
                # crude heuristic: treat each nonzero row as a detection
                total_dets += (out.abs().sum(dim=1) > 0).sum().item()
            else:
                # fallback
                total_imgs += batch.shape[0]

    avg_dets = (total_dets / total_imgs) if total_imgs > 0 else 0.0
    return float(avg_dets)


# ---------------------------
# Training step and training_func for TorchEvaluator (used by QAT)
# ---------------------------

def detection_training_step(batch, model, *args, **kwargs):
    """A training_step used by TorchEvaluator. Should return loss (or dict/tuple).

    This is a minimal example that assumes `batch` contains images and targets suitable
    for your model. For a real YOLOv5 training step, integrate your original loss
    calculation (classification + bbox + objectness losses).
    """
    imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
    preds = model(imgs)
    # Placeholder: we compute a dummy loss (L2 on outputs) -- replace with YOLO loss.
    if isinstance(preds, (list, tuple)):
        # flatten to a tensor if possible
        preds_tensor = preds[0] if len(preds) > 0 and isinstance(preds[0], torch.Tensor) else torch.tensor(0.0, device=imgs.device)
    elif isinstance(preds, torch.Tensor):
        preds_tensor = preds
    else:
        preds_tensor = torch.tensor(0.0, device=imgs.device)

    loss = preds_tensor.float().abs().mean()
    return loss


def detection_training_func(model, optimizers, training_step, lr_schedulers=None, max_steps=None, max_epochs=None):
    """A training loop compatible with TorchEvaluator.

    The evaluator will call this `training_func` and pass `max_steps` / `max_epochs` to control duration.
    You MUST respect those limits in your loop (break when exceeded).
    """
    # Recreate dataloader inside training_func so it uses the (wrapped) model's device etc.
    # For brevity we use a tiny dummy dataloader. Replace with your real train loader.
    train_dl = make_dataloader_from_dir('./train_images', imgsz=640, batch_size=8, shuffle=True)

    device = next(model.parameters()).device
    optimizer = optimizers  # TorchEvaluator will pass a traced optimizer already

    step = 0
    epoch = 0
    while True:
        if max_epochs is not None and epoch >= max_epochs:
            break
        for batch in train_dl:
            if max_steps is not None and step >= max_steps:
                return
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = training_step(batch, model)
            loss.backward()
            optimizer.step()
            step += 1
        epoch += 1
        if lr_schedulers is not None:
            for s in lr_schedulers:
                s.step()


# ---------------------------
# Main flows: PTQ and QAT
# ---------------------------

def run_ptq(model: torch.nn.Module, config_list: list, eval_func: Callable, calib_dataloader: DataLoader, out_path: str):
    """Run NNI PtqQuantizer and save the quantized model.
    """
    evaluator = TorchEvaluator(training_func=None, optimizers=None, training_step=None, evaluating_func=lambda m: eval_func(m))
    quantizer = PtqQuantizer(model, config_list, evaluator)

    # compress returns (compressed_model, calibration_config)
    compressed_model, calibration_config = quantizer.compress(max_steps=None, max_epochs=None, calib_dataloader=calib_dataloader)

    print('PTQ finished. Saving compressed model to', out_path)
    torch.save(compressed_model.state_dict(), out_path)


def run_qat(model: torch.nn.Module, config_list: list, eval_func: Callable, epochs: int, lr: float, out_path: str):
    """Run NNI QAT_Quantizer (training-aware) and save the fine-tuned quantized model.
    """
    # Prepare optimizer and wrap with nni.trace
    optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=lr)

    # Create evaluator: we provide training_func, traced optimizer, training_step and evaluating_func
    evaluator = TorchEvaluator(training_func=detection_training_func,
                               optimizers=optimizer,
                               training_step=detection_training_step,
                               evaluating_func=lambda m: eval_func(m))

    quantizer = QAT_Quantizer(model, config_list, evaluator)
    compressed_model, calibration_config = quantizer.compress(max_steps=None, max_epochs=epochs)

    # After compress, you typically want to export/save the quantized model weights
    print('QAT finished. Saving compressed model to', out_path)
    torch.save(compressed_model.state_dict(), out_path)


# ---------------------------
# Example config_list for quantization
# ---------------------------
# This is a simple config that targets Conv/Linear modules for per-channel weight quantization.
# Customize according to your model structure. See NNI docs for full config schema.
DEFAULT_CONFIG_LIST = [
    {
        "op_types": ["Conv", "Conv2d", "Linear"],
        "quant_scheme": "sym",
        "granularity": "per_channel",
        "bit": 8,
        "quant_dtype": "int8"
    }
]


# ---------------------------
# CLI and orchestration
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser("NNI YOLOv5 Quantization Demo")
    ap.add_argument('--mode', choices=['ptq', 'qat'], required=True)
    ap.add_argument('--yolov5_weights', type=str, default='yolov5s.pt')
    ap.add_argument('--calib_dir', type=str, default='./calib_images')
    ap.add_argument('--calib_num', type=int, default=100)
    ap.add_argument('--val_dir', type=str, default='./val_images')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=str, default='yolov5s_nni_quant.pth')
    return ap.parse_args()


def main():
    args = parse_args()

    # Load YOLOv5 model via ultralytics and extract the underlying torch module
    y = YOLO(args.yolov5_weights)
    model = y.model  # torch.nn.Module

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if args.mode == 'ptq':
        # Build calibration dataloader
        calib_dl = make_dataloader_from_dir(args.calib_dir, imgsz=640, batch_size=8, shuffle=False)
        # Simple evaluator that uses val_dir (YOU SHOULD REPLACE with proper mAP computation)
        eval_func = lambda m: default_evaluating_func(m, imgsz=640, val_dir=args.val_dir)
        run_ptq(model, DEFAULT_CONFIG_LIST, eval_func, calib_dl, args.out)

    elif args.mode == 'qat':
        eval_func = lambda m: default_evaluating_func(m, imgsz=640, val_dir=args.val_dir)
        run_qat(model, DEFAULT_CONFIG_LIST, eval_func, epochs=args.epochs, lr=args.lr, out_path=args.out)


if __name__ == '__main__':
    main()
