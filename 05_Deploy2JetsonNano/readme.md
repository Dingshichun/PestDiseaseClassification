# 模型部署到 jetson-nano-b01
将 GPU 训练好的 ShuffleNet 图像分类模型部署到 Jetson Nano B01
## 模型转换流程
ShuffleNet.pth 需转换为 TensorRT 引擎以提升推理性能。

### 通过 ONNX 中间格式（推荐）
1. **导出为 ONNX 格式**
   ```python
   import torch
   import torchvision
   
   # 加载训练好的模型
   model = torch.load('shufflenet.pth')
   model.eval().cuda()
   
   # 创建示例输入
   dummy_input = torch.randn(1, 3, 224, 224).cuda()  # 根据实际输入尺寸调整
   
   # 导出 ONNX
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
   ```

2. **优化 ONNX 模型**
   ```bash
   python -m onnxsim shufflenet.onnx shufflenet_sim.onnx
   ```

3. **转换为 TensorRT 引擎**
   ```python
   import tensorrt as trt
   
   TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
   
   def build_engine(onnx_path, engine_path, precision="fp16"):
       with trt.Builder(TRT_LOGGER) as builder:
           config = builder.create_builder_config()
           config.max_workspace_size = 1 << 30  # 1GB
           
           if precision == "fp16" and builder.platform_has_fast_fp16:
               config.set_flag(trt.BuilderFlag.FP16)
           
           network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
           with builder.create_network(network_flags) as network:
               parser = trt.OnnxParser(network, TRT_LOGGER)
               with open(onnx_path, "rb") as f:
                   if not parser.parse(f.read()):
                       for error in range(parser.num_errors):
                           print(parser.get_error(error))
                       return None
               
               # 设置动态形状配置文件
               profile = builder.create_optimization_profile()
               profile.set_shape("input", (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))
               config.add_optimization_profile(profile)
               
               engine = builder.build_serialized_network(network, config)
               with open(engine_path, "wb") as f:
                   f.write(engine)
               return engine
   
   build_engine("shufflenet_sim.onnx", "shufflenet.engine", precision="fp16")
   ```

## 部署与测试
1. **编写推理脚本**
   ```python
   import tensorrt as trt
   import pycuda.driver as cuda
   import pycuda.autoinit
   import numpy as np
   import cv2
   
   class ShuffleNetTRT:
       def __init__(self, engine_path):
           self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
           with open(engine_path, "rb") as f:
               engine_data = f.read()
           
           self.runtime = trt.Runtime(self.TRT_LOGGER)
           self.engine = self.runtime.deserialize_cuda_engine(engine_data)
           self.context = self.engine.create_execution_context()
           
           # 分配输入输出缓冲区
           self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
           for binding in self.engine:
               size = trt.volume(self.engine.get_binding_shape(binding))
               dtype = trt.nptype(self.engine.get_binding_dtype(binding))
               host_mem = cuda.pagelocked_empty(size, dtype)
               device_mem = cuda.mem_alloc(host_mem.nbytes)
               self.bindings.append(int(device_mem))
               if self.engine.binding_is_input(binding):
                   self.inputs.append({'host': host_mem, 'device': device_mem})
               else:
                   self.outputs.append({'host': host_mem, 'device': device_mem})
       
       def infer(self, image):
           # 预处理（根据训练时的预处理调整）
           image = cv2.resize(image, (224, 224))
           image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
           image = np.expand_dims(image, axis=0)
           
           # 复制数据到GPU
           np.copyto(self.inputs[0]['host'], image.ravel())
           cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
           
           # 执行推理
           self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
           
           # 复制结果回CPU
           cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
           self.stream.synchronize()
           
           output = self.outputs[0]['host']
           return output
   
   # 使用示例
   detector = ShuffleNetTRT("shufflenet.engine")
   image = cv2.imread("test.jpg")
   result = detector.infer(image)
   print("推理结果:", result)
   ```

2. **实时摄像头推理**
   ```python
   cap = cv2.VideoCapture(0)
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       result = detector.infer(frame)
       # 解析结果并显示
       cv2.imshow("ShuffleNet Inference", frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
   cap.release()
   cv2.destroyAllWindows()
   ```

## 性能优化建议
1. **精度选择**
   - **FP32**：最高精度，兼容性好。
   - **FP16**：平衡精度与性能，在 Jetson Nano 上推荐使用。
   - **INT8**：极致性能，但需要校准数据集，可能损失精度。

2. **内存管理**
   - 控制 batch size（Jetson Nano 建议不超过 8）。
   - 定期清理缓存：`torch.cuda.empty_cache()`。
   - 启用共享内存：`export CUDA_LAUNCH_BLOCKING=1`。

3. **散热与供电**
   - 确保良好散热，避免过热降频。
   - 使用 5V/4A 电源，避免电压不稳导致重启。

4. **监控工具**
   - 安装 jtop：`sudo pip3 install jetson-stats`，监控GPU/CPU使用率。
   - 使用 tegrastats：`tegrastats --interval 1000`查看实时状态。

## 常见问题解决
1. **CUDA 内存不足**：减小 batch size 或启用 `torch.backends.cudnn.enabled=False`。
2. **导入 torch 失败**：检查 Python 路径，使用`which python3`确认。
3. **模型转换失败**：确保 ONNX opset_version≥11，检查自定义算子兼容性。
4. **推理速度慢**：启用 TensorRT FP16 模式，进行层融合优化。
