import torch
import torchvision

# 加载训练好的模型
model_path='./ShuffleNet.pth'
model = torch.load(model_path,map_location="cuda")
model.eval()

# 创建示例输入
dummy_input = torch.randn(1, 3, 224, 224).cuda()  # 根据实际输入尺寸调整

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "shufflenet.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # 支持动态批次
)