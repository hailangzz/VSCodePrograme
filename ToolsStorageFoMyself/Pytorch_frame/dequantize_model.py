
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class QuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv = nn.Conv2d(3, 64, kernel_size=3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        return self.dequant(x)

def convert_quantized_to_float(quant_model):
    # 创建非量化模型副本
    float_model = QuantizedModel().eval()
    
    # 复制反量化后的权重
    with torch.no_grad():
        # 处理量化卷积层
        if hasattr(quant_model.conv, 'weight'):
            quant_weight = quant_model.conv.weight()
            float_model.conv.weight.copy_(quant_weight.dequantize())
            
        # 处理量化偏置（如果存在）
        if hasattr(quant_model.conv, 'bias') and quant_model.conv.bias is not None:
            float_model.conv.bias.copy_(quant_model.conv.bias)
    
    return float_model

if __name__ == "__main__":
    # 1. 创建并量化示例模型
    model = QuantizedModel().eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    quantized_model = torch.quantization.prepare(model, inplace=False)
    quantized_model = torch.quantization.convert(quantized_model)
    
    # 2. 执行转换
    float_model = convert_quantized_to_float(quantized_model)
    
    # 3. 验证输出一致性
    test_input = torch.randn(1, 3, 224, 224)
    quant_output = quantized_model(test_input)
    float_output = float_model(test_input)
    print(f"输出差异: {torch.max(torch.abs(quant_output - float_output))}")
