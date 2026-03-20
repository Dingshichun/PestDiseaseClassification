# 导入所需的Python库
import os  # 操作系统接口，用于文件路径操作
import io  # 输入输出流处理，用于处理字节数据
import json  # JSON格式数据处理
import torch  
import torchvision.transforms as transforms  # PyTorch图像预处理转换
from PIL import Image  # Python图像处理库
from flask import Flask, jsonify, request, render_template  # Flask Web框架相关模块
from flask_cors import CORS  # 跨域资源共享支持

# 创建Flask应用实例
app = Flask(__name__) 
# 启用跨域资源共享，允许前端从不同域访问API
CORS(app)  

# 模型文件路径 - 包含完整的模型结构和权重参数
model_path = "./ShuffleNet_best_complete.pth"
# 类别标签映射文件路径 - 包含类别索引到名称的映射
class_json_path = "./class_indices.json"
# 检查模型文件是否存在，不存在则抛出断言错误
assert os.path.exists(model_path), "模型文件不存在..."
# 检查类别映射文件是否存在，不存在则抛出断言错误
assert os.path.exists(class_json_path), "类别标签映射文件不存在..."

# 选择计算设备：如果可用则使用GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 打印当前使用的设备信息
print(f"使用的设备: {device}")

# 加载完整模型（包含模型结构和权重参数）
# torch.load从文件加载模型，map_location指定加载到指定设备
model = torch.load(model_path, map_location=device)
# 设置模型为评估模式，禁用dropout和batch normalization的随机性
model.eval()

# 加载类别映射信息
# 以二进制读取模式打开JSON文件
with open(class_json_path, 'rb') as json_file:
    # 解析JSON文件内容为Python字典
    class_indict = json.load(json_file)


def transform_image(image_bytes):
    """
    图像预处理转换函数
    
    参数:
        image_bytes: 图像的字节数据
        
    返回:
        tensor: 预处理后的图像张量，包含batch维度，已转移到指定设备
    """
    # 定义图像预处理流程组合
    my_transforms = transforms.Compose([
        transforms.Resize(224),  # 调整图像大小为224×224
        transforms.ToTensor(),  # 将PIL图像或numpy数组转换为张量，并归一化到[0,1]
        transforms.Normalize(  # 标准化处理，使用ImageNet数据集统计量
            [0.485, 0.456, 0.406],  # 均值
            [0.229, 0.224, 0.225]   # 标准差
        )
    ])
    
    # 从字节数据创建 PIL 图像对象
    image = Image.open(io.BytesIO(image_bytes))
    
    # 确保图像为RGB格式（3通道）
    if image.mode != "RGB":
        # 如果不是RGB格式，转换为RGB格式
        image = image.convert('RGB')
    
    # 应用预处理转换
    # unsqueeze(0)在维度0添加batch维度（形状从[C,H,W]变为[1,C,H,W]）
    # to(device)将张量转移到指定设备
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    """
    对输入图像进行预测并返回结果
    
    参数:
        image_bytes: 输入图像的字节数据
        
    返回:
        return_info: 包含预测结果的字典
    """
    try:
        # 对输入图像进行预处理转换
        tensor = transform_image(image_bytes=image_bytes)
        
        # 模型前向传播并计算概率分布
        # 1. model.forward(tensor): 执行模型前向计算
        # 2. squeeze(): 去除batch维度（从[1,num_classes]变为[num_classes]）
        # 3. torch.softmax(dim=0): 沿类别维度计算softmax，得到概率分布
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        
        # 将模型输出转换为numpy数组
        # 1. detach(): 从计算图中分离张量，不跟踪梯度
        # 2. cpu(): 将张量转移到CPU内存
        # 3. numpy(): 将张量转换为numpy数组
        prediction = outputs.detach().cpu().numpy()
        
        # 定义输出格式模板
        template = "class:{:<15} probability:{:.3f}"
        
        # 创建(类别名称, 概率)元组列表
        # enumerate(prediction)遍历每个类别索引和对应的概率
        index_pre = [(class_indict[str(index)], float(p)) 
                     for index, p in enumerate(prediction)]
        
        # 按概率值从高到低排序
        # key=lambda x: x[1] 表示按元组第二个元素（概率）排序
        # reverse=True 表示降序排列
        index_pre.sort(key=lambda x: x[1], reverse=True)
        
        # 格式化输出文本
        text = [template.format(k, v) for k, v in index_pre]
        
        # 构建返回结果字典
        return_info = {"result": text}
        
    except Exception as e:
        # 如果发生异常，捕获并返回错误信息
        return_info = {"result": [str(e)]}
    
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()  # 装饰器，禁用梯度计算，提高推理效率
def predict():
    """
    预测API接口，处理POST请求进行图像分类
    
    返回:
        JSON响应: 包含预测结果
    """
    # 从请求中获取上传的文件
    image = request.files["file"]
    
    # 读取文件的字节数据
    img_bytes = image.read()
    
    # 调用预测函数获取结果
    info = get_prediction(image_bytes=img_bytes)
    
    # 返回JSON格式的响应
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    """
    根路径路由，返回上传页面
    
    返回:
        HTML页面: 文件上传界面
    """
    # 渲染并返回上传页面的HTML模板
    return render_template("up.html")


if __name__ == '__main__':
    """
    主程序入口
    """
    # 启动Flask应用
    # host="0.0.0.0" 允许所有网络接口访问
    # port=5000 设置服务端口为5000
    app.run(host="0.0.0.0", port=5000)


