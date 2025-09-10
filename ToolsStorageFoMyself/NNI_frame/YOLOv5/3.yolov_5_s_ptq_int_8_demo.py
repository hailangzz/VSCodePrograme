"""
YOLOv5s Quantization Demo (ONNX + ONNX Runtime)

This script shows two practical ways to quantize a pretrained YOLOv5s model to INT8:
  1) Dynamic quantization (weights-only)
  2) Static quantization (with calibration images)

It also exports YOLOv5s to ONNX first (via ultralytics), benchmarks latency, and compares file sizes.

Usage examples
--------------
# 1) Export YOLOv5s to ONNX (once)
python yolov5s_ptq_int8_demo.py --step export --onnx yolov5s.onnx --imgsz 640

# 2) Dynamic quantization
python yolov5s_ptq_int8_demo.py --step dynamic --onnx yolov5s.onnx --out yolov5s.int8.dynamic.onnx

# 3) Static quantization (post-training) with calibration images
python yolov5s_ptq_int8_demo.py --step static --onnx yolov5s.onnx --out yolov5s.int8.static.onnx --calib_dir ./calib_images --num_calib 50

# 4) Benchmark a model (fp32 or int8)
python yolov5s_ptq_int8_demo.py --step bench --onnx yolov5s.onnx --warmup 10 --iters 100 --img path/to/image.jpg

Requirements
------------
- Python 3.8+
- pip install ultralytics onnx onnxruntime onnxruntime-tools (optional)

Notes
-----
- This demo focuses on quantization and latency/size. It does not implement full post-processing/NMS visualization.
- Static INT8 usually needs AVX/AVX2 capable CPU for best speedups.
- The ONNX exported here uses opset 12+ and NCHW layout at fixed img size (default 640).
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Iterator, Optional

import numpy as np

# Lazy imports (only when needed)


def export_yolov5s_to_onnx(onnx_path: str, imgsz: int = 640) -> None:
    """Export pretrained YOLOv5s to ONNX using ultralytics.

    Produces a single ONNX file with fixed input size (B,3,H,W) = (1,3,imgsz,imgsz).
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics not installed. Run: pip install ultralytics\n" \
            "Original error: %r" % (e,)
        )

    print(f"[Export] Loading pretrained weights 'yolov5s.pt' via ultralytics...")
    model = YOLO("yolov5s.pt")

    print(f"[Export] Exporting to ONNX -> {onnx_path} (imgsz={imgsz})")
    # NOTE: ultralytics will save into a 'weights' folder by default; we move/rename below.
    export_results = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=False, dynamic=False)
    # export_results is a path to the generated file
    exported = Path(export_results)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != Path(onnx_path).resolve():
        os.replace(str(exported), onnx_path)
    print(f"[Export] Saved: {onnx_path}  (size={Path(onnx_path).stat().st_size/1e6:.2f} MB)")


def preprocess_image(img_path: str, imgsz: int = 640) -> np.ndarray:
    """Load and letterbox an image to (1,3,imgsz,imgsz) float32 in [0,1].
    This is a minimal preprocessing compatible with YOLOv5 ONNX export.
    """
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = min(imgsz / w, imgsz / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas.paste(img_resized, (left, top))

    x = np.array(canvas).astype(np.float32) / 255.0  # HWC [0,1]
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, 0)  # NCHW
    return x


class YOLOv5CalibrationDataReader:
    """A simple calibration dataloader for ONNX Runtime static quantization.

    Expects a directory of images. Yields NCHW float32 batches of size 1.
    """

    def __init__(self, calib_dir: str, imgsz: int = 640, limit: Optional[int] = None):
        self.files = [
            str(p) for p in Path(calib_dir).glob("**/*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if limit is not None:
            self.files = self.files[:limit]
        self.imgsz = imgsz
        self.enum: Optional[Iterator] = None

    def get_next(self) -> Optional[dict]:
        if self.enum is None:
            self.enum = iter(self.files)
        try:
            img_path = next(self.enum)
        except StopIteration:
            return None
        x = preprocess_image(img_path, self.imgsz)
        return {"images": x}  # match input name used by ultralytics ONNX export



def quantize_dynamic_onnx(fp32_path: str, int8_path: str) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"[Quantize-Dynamic] -> {int8_path}")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    print(f"[Quantize-Dynamic] Done. Size={Path(int8_path).stat().st_size/1e6:.2f} MB")


def quantize_static_onnx(fp32_path: str, int8_path: str, calib_dir: str, imgsz: int, num_calib: int) -> None:
    from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat, QuantType
    reader = YOLOv5CalibrationDataReader(calib_dir, imgsz=imgsz, limit=num_calib)
    print(f"[Quantize-Static] Calibrating with {len(reader.files)} images from: {calib_dir}")
    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=reader,
        calibration_method=CalibrationMethod.MinMax,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        optimize_model=True,
        quant_format=QuantFormat.QDQ,
    )
    print(f"[Quantize-Static] Done. Size={Path(int8_path).stat().st_size/1e6:.2f} MB")


def benchmark_onnx(model_path: str, imgsz: int, img_path: Optional[str], warmup: int, iters: int) -> None:
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = os.cpu_count() or 1
    sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    if img_path is None:
        # Create a dummy image
        x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        print("[Benchmark] Using random input")
    else:
        x = preprocess_image(img_path, imgsz)
        print(f"[Benchmark] Using image: {img_path}")

    # Warmup
    for _ in range(max(0, warmup)):
        _ = sess.run(None, {input_name: x})

    # Measure
    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg = float(np.mean(times))
    p50 = float(np.percentile(times, 50))
    p90 = float(np.percentile(times, 90))
    size_mb = Path(model_path).stat().st_size / 1e6

    print(f"[Benchmark] Model: {model_path}")
    print(f"[Benchmark] Size:  {size_mb:.2f} MB | Iters: {iters} | Warmup: {warmup}")
    print(f"[Benchmark] Avg:   {avg:.2f} ms  | P50: {p50:.2f} ms  | P90: {p90:.2f} ms")


def parse_args():
    ap = argparse.ArgumentParser("YOLOv5s ONNX INT8 Quantization Demo")
    ap.add_argument("--step", required=True, choices=["export", "dynamic", "static", "bench"],
                    help="Which step to run")
    ap.add_argument("--onnx", default="yolov5s.onnx", help="Path to (input) ONNX model")
    ap.add_argument("--out", default=None, help="Path to save quantized ONNX model")
    ap.add_argument("--imgsz", type=int, default=640, help="Export/Preprocess image size")
    ap.add_argument("--calib_dir", default=None, help="Directory of calibration images for static quantization")
    ap.add_argument("--num_calib", type=int, default=100, help="Number of calibration images to use")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup runs for benchmark")
    ap.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    ap.add_argument("--img", default=None, help="Image file to run during benchmark (optional)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.step == "export":
        export_yolov5s_to_onnx(args.onnx, imgsz=args.imgsz)
        return

    if args.step == "dynamic":
        out = args.out or Path(args.onnx).with_suffix(".int8.dynamic.onnx")
        quantize_dynamic_onnx(args.onnx, str(out))
        return

    if args.step == "static":
        if not args.calib_dir:
            raise SystemExit("--calib_dir is required for static quantization")
        out = args.out or Path(args.onnx).with_suffix(".int8.static.onnx")
        quantize_static_onnx(args.onnx, str(out), calib_dir=args.calib_dir, imgsz=args.imgsz, num_calib=args.num_calib)
        return

    if args.step == "bench":
        benchmark_onnx(args.onnx, imgsz=args.imgsz, img_path=args.img, warmup=args.warmup, iters=args.iters)
        return


if __name__ == "__main__":
    main()
