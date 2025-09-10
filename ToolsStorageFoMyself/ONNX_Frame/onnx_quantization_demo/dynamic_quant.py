
import onnxruntime
from onnxruntime.quantization import quantize_dynamic

def dynamic_quantization(model_path, output_path):
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=onnxruntime.quantization.QuantType.QInt8,
        optimize_model=True
    )

if __name__ == "__main__":
    dynamic_quantization("yolov8n.onnx", "yolov8n_quant_dynamic.onnx")
