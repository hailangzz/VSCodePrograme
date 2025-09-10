
import torch
import onnx
import onnxruntime as ort
import numpy as np

def export_dynamic_model(model, dummy_input, dynamic_axes):
    torch.onnx.export(
        model,
        dummy_input,
        "dynamic_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=13
    )
    print("动态模型导出成功")

def convert_to_static(input_path, output_path):
    model = onnx.load(input_path)
    # 移除动态维度标记
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:
                dim.dim_value = 640  # 设为固定值(示例用640x640)
    
    onnx.save(model, output_path)
    print(f"静态模型已保存至 {output_path}")

def verify_model(model_path):
    sess = ort.InferenceSession(model_path)
    input_shape = sess.get_inputs()[0].shape
    print(f"模型输入形状: {input_shape} (静态)")

if __name__ == "__main__":
    # 示例模型(替换为实际模型)
    class SampleModel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=0.5)

    model = SampleModel().eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # 步骤1: 导出动态模型
    export_dynamic_model(model, dummy_input, 
        dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}})
    
    # 步骤2: 转换为静态模型
    convert_to_static("dynamic_model.onnx", "static_model.onnx")
    
    # 步骤3: 验证
    verify_model("static_model.onnx")
