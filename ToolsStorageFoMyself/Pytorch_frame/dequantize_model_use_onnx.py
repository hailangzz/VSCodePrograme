
import torch
import torchvision
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert

# 1. 定义带量化支持的模型
class QuantizableResNet18(torchvision.models.ResNet):
    def __init__(self):
        super(QuantizableResNet18, self).__init__(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        return self.dequant(x)

# 2. 模型准备与量化配置
def prepare_model(pretrained=True):
    model = QuantizableResNet18()
    if pretrained:
        state_dict = torchvision.models.resnet18(pretrained=True).state_dict()
        model.load_state_dict(state_dict)
    
    # 量化配置
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = prepare_qat(model.train())
    return model

# 3. 校准函数（模拟实际校准过程）
def calibrate_model(model, calib_data):
    model.eval()
    with torch.no_grad():
        for data in calib_data:
            model(torch.randn(1, 3, 224, 224))  # 使用真实数据时应替换

# 4. 主转换流程
def convert_to_onnx():
    # 准备模型
    model = prepare_model()
    
    # 模拟校准数据（实际应用需替换为真实数据）
    calib_data = [torch.randn(1, 3, 224, 224) for _ in range(32)]
    calibrate_model(model, calib_data)
    
    # 转换为静态量化模型
    quantized_model = convert(model.eval())
    
    # 导出ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        quantized_model,
        dummy_input,
        "quantized_resnet18.onnx",
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}},
        export_params=True
    )
    print("ONNX模型导出成功")

if __name__ == "__main__":
    convert_to_onnx()
