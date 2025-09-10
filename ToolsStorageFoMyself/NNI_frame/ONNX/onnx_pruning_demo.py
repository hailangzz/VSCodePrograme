
import onnx
import torch
import nni
# from nni.compression.pytorch import apply_compression_results
from nni.compression.pruning import L1NormPruner
from nni.compression.pruning import AGPPruner

# 1. 加载ONNX模型
onnx_model = onnx.load("model.onnx")  # 替换为实际模型路径
dummy_input = torch.randn(1, 3, 224, 224)  # 根据模型输入调整维度

# 2. 剪枝配置（延续历史对话中的配置示例）
config_list = [{
    'op_types': ['Conv', 'Gemm'],  
    'sparse_ratio': 0.6,           
    'exclude_op_names': ['output'] 
}]

# 3. 初始化剪枝器（提供L1Norm和AGP两种示例）
pruner = L1NormPruner(torch_model, config_list)
# 或使用自动渐进式剪枝：
# pruner = AGPPruner(torch_model, config_list, pruning_algorithm='l1', total_iteration=10)

# 4. 计算掩码并压缩模型
_, masks = pruner.compress()
pruner.show_pruned_weights()  # 可视化剪枝结果

# 5. 模型加速（永久移除被剪枝参数）
from nni.compression.speedup import ModelSpeedup
model_speedup = ModelSpeedup(torch_model, dummy_input, masks)
model_speedup.speedup_model()

# 6. 保存剪枝后模型
onnx.save(onnx_model, "pruned_model.onnx")

# 可选：剪枝后微调（需准备训练数据）
def train(model, optimizer, criterion):
    # 实现微调训练逻辑
    pass

optimizer = torch.optim.Adam(torch_model.parameters())
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(5):  # 示例微调5个epoch
    train(torch_model, optimizer, criterion)
