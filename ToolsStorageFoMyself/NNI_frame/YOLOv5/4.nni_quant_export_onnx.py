"""
YOLOv5s Quantization + ONNX Export + ONNXRuntime Inference Demo

This script demonstrates:
1. Quantizing YOLOv5s (PTQ or QAT) with NNI
2. Exporting the quantized model to ONNX
3. Running inference with ONNX Runtime (dummy input or real image)

Requirements
------------
pip install ultralytics nni torch torchvision onnx onnxruntime pillow

Usage examples
--------------
# 1) Post-Training Quantization (PTQ)
python yolov5s_quant_onnx_demo.py --mode ptq --epochs 1

# 2) Quantization-Aware Training (QAT)
python yolov5s_quant_onnx_demo.py --mode qat --epochs 5

# 3) Export quantized model to ONNX
python yolov5s_quant_onnx_demo.py --mode ptq --export_onnx yolov5s_int8.onnx

# 4) Run ONNXRuntime inference on real image
python yolov5s_quant_onnx_demo.py --mode ptq --export_onnx yolov5s_int8.onnx --infer path/to/image.jpg
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ultralytics import YOLO
from PIL import Image

import nni
from nni.compression.quantization import QAT_Quantizer, PtqQuantizer

import onnx
import onnxruntime as ort


# -----------------------------
# Dummy dataset for quick demo
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


def get_dataloader(batch_size=2, n=50):
    dataset = RandomDataset(n=n)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -----------------------------
# Quantization routines
# -----------------------------
def ptq_demo(model, config_list):
    quantizer = PtqQuantizer(model, config_list)
    quantizer.compress()
    print("[NNI-PTQ] Quantization complete.")
    return model


def qat_demo(model, dataloader, config_list, epochs=1):
    quantizer = QAT_Quantizer(model, config_list)
    model = quantizer.compress()

    optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for imgs, _ in dataloader:
            preds = model(imgs)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            loss = criterion(preds, torch.zeros_like(preds))
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"[NNI-QAT] Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

    quantizer.export_model(model_path="yolov5s_qat_model.pth", calibration_path="qat_calib.pth")
    print("[NNI-QAT] Exported quantized model.")
    return model


# -----------------------------
# Export and ONNX Runtime
# -----------------------------
def export_to_onnx(model, export_path="yolov5s_int8.onnx", img_size=640):
    dummy_input = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["images"],
        output_names=["outputs"],
        opset_version=12,
        dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}},
    )
    print(f"[Export] ONNX model saved to {export_path}")


def run_onnx_inference(onnx_path, img_path=None, img_size=640):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    if img_path is None:
        dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
        outputs = sess.run(None, {"images": dummy_input})
        print(f"[ONNXRuntime] Dummy input -> output shape: {outputs[0].shape}")
    else:
        img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
        arr = np.array(img).transpose(2, 0, 1) / 255.0
        arr = arr.astype(np.float32)[None]  # (1,3,H,W)
        outputs = sess.run(None, {"images": arr})
        print(f"[ONNXRuntime] Real image inference done. Output shape: {outputs[0].shape}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("YOLOv5s NNI Quantization + ONNX Demo")
    parser.add_argument("--mode", choices=["ptq", "qat"], required=True, help="Quantization mode")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (for QAT)")
    parser.add_argument("--export_onnx", type=str, default=None, help="Export quantized model to ONNX path")
    parser.add_argument("--infer", type=str, default=None, help="Image path for ONNXRuntime inference")
    args = parser.parse_args()

    print("[Load] Loading YOLOv5s pretrained model...")
    yolov5 = YOLO("yolov5s.pt").model

    dataloader = get_dataloader(n=20)

    config_list = [{
        'quant_types': ['weight', 'input'],
        'quant_bits': {'weight': 8, 'input': 8},
        'op_types': ['Conv2d', 'Linear']
    }]

    if args.mode == "ptq":
        model_q = ptq_demo(yolov5, config_list)
    else:
        model_q = qat_demo(yolov5, dataloader, config_list, epochs=args.epochs)

    if args.export_onnx:
        export_to_onnx(model_q, args.export_onnx)
        run_onnx_inference(args.export_onnx, args.infer)

    print("[Done] Quantization + Export + Inference finished.")


if __name__ == "__main__":
    main()
