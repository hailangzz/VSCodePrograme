# yolov_5_s_ptq_int_8_demo.py说明：

一般模型量化的步骤为：
1、使用pytorch，训练pt、pth模型
2、对pt模型进行剪枝操作
3、对pt模型进行蒸馏、微调训练
4、将微调训练后的模型，export为onnx模型
5、对onnx模型进行量化（静态量化、动态量化）