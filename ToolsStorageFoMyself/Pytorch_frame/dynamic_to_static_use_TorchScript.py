
import torch
import torch.nn as nn

class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # 包含动态控制流
        if x.shape[2] > 224:
            x = nn.functional.interpolate(x, size=224)
        return self.pool(self.conv(x))

def export_dynamic_model():
    model = DynamicModel().eval()
    
    # 动态输入示例（可变尺寸）
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # 方法1：trace模式（适合无复杂控制流）
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("dynamic_traced.pt")
    
    # 方法2：script模式（保留动态逻辑）
    scripted_model = torch.jit.script(model)
    scripted_model.save("dynamic_scripted.pt")
    return traced_model, scripted_model

def convert_to_static(model_path, fixed_size=(224, 224)):
    # 加载动态模型
    dynamic_model = torch.jit.load(model_path)
    
    # 创建静态输入
    static_input = torch.randn(1, 3, *fixed_size)
    
    # 重新trace固定尺寸
    static_model = torch.jit.trace(dynamic_model, static_input)
    static_model.save("static_model.pt")
    return static_model

def verify_models():
    # 验证动态模型
    dyn_model = torch.jit.load("dynamic_traced.pt")
    print(f"动态模型输入示例: {dyn_model(torch.randn(1,3,256,256)).shape}")
    
    # 验证静态模型
    static_model = torch.jit.load("static_model.pt")
    print(f"静态模型输入示例: {static_model(torch.randn(1,3,224,224)).shape}")

if __name__ == "__main__":
    print("1. 导出动态模型...")
    traced, scripted = export_dynamic_model()
    
    print("\n2. 转换为静态模型...")
    static_model = convert_to_static("dynamic_traced.pt")
    
    print("\n3. 验证模型:")
    verify_models()
