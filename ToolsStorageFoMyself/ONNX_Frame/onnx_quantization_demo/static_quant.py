
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader
import numpy as np

class YOLODataReader(CalibrationDataReader):
    def __init__(self, calibration_images):
        self.images = calibration_images
        self.index = 0

    def get_next(self):
        if self.index < len(self.images):
            image = self.images[self.index]
            self.index += 1
            return {"input": image}
        return None

def static_quantization(model_path, output_path, calibration_data):
    dr = YOLODataReader(calibration_data)
    quantize_static(
        model_path,
        output_path,
        dr,
        activation_type=onnxruntime.quantization.QuantType.QUInt8,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
        per_channel=True,
        optimize_model=True
    )

if __name__ == "__main__":
    # 示例：加载校准数据（实际应替换为真实数据）
    calibration_data = [np.random.rand(1,3,640,640).astype(np.float32) for _ in range(10)]
    static_quantization("yolov8n.onnx", "yolov8n_quant_static.onnx", calibration_data)
