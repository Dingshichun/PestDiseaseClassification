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
        # 改变维度。OpenCV加载的图像形状是：(高度, 宽度, 通道)，即(H, W, C)
        # 转换后是(C, H, W)，再增加一个批次维度 batch，
        # 形状即为常见的(B, C, H, W)
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


# 摄像头实时推理
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