# PestDiseaseClassification
实现农业病虫害图像分类。具体内容包括图像数据准备、模型训练、模型测试、部署到网页端。  
### 数据准备
内容见 01_DataPreparation 文件夹  
数据集下载：公开数据集，如 **https://www.kaggle.com/datasets/emmarex/plantdisease**。包含 38 种植物病虫害约五万张图像。

数据集划分：训练：验证：测试 = 6:2:2，先将图像按照 8:2 划分为训练和验证：测试，后面实际训练时再将训练和验证按照 6:2 划分为训练：验证。[图像划分脚本](./01_DataPreparation/split_TrainVal_Test.py)

可以生成类别索引文件 class_indices.json。[生成类别索引的脚本](./01_DataPreparation/create_class_json.py)

### 模型训练

内容见文件夹 02_TrainModel，其中 my_dataset.py 和 utils.py 中是一些数据处理和模型训练的函数。  
训练使用阿里云服务器，cpu 是 16 核，64G，GPU 是 NVIDIA A10 ，实例规格是 ecs.gn7i-c16g1.4xlarge，操作系统是 ubuntu22.04 64位，预装 NVIDIA GPU 驱动和 CUDA 。  
使用 vscode 远程连接云服务器之后，安装 gpu 版本的 pytorch，安装时应该先创建虚拟环境，再将 pytorch 安装到虚拟环境中，以实现环境隔离。安装成功与否的验证方法如下  
```python
# 终端输入 python3 并执行
import torch
print(torch.cuda.is_available()) # 应该输出 True，False 表明是 cpu 版本的 torch
```
训练时选择安装了 pytorch 的虚拟环境作为解释器。训练使用 pytorch 官方的预训练模型，加载预训练模型之后，冻结除分类层之外的所有层，只训练分类层的参数，以快速达到较高的准确率。采用早停机制，即如果验证集的准确率连续 11 个 epoch 没有上升，就停止训练，保存验证集准确率最高的那个 epoch 对应的模型参数，详细实现见[ShuffleNet训练脚本](./02_TrainModel/ShuffleNet.py)

### 模型测试
内容见 03_TestModel 文件夹。  
将训练好的带有模型权重和结构的 pth 文件复制到 03_TestModel 文件夹，在测试集中选择几张图像也复制到这里。  
执行 ConfusionMatrix.py 可以得到混淆矩阵、准确率、召回率、模型计算量、模型参数量以及模型推理速度，执行 predict.py 文件可以得到单张图像的分类结果。

### 部署到网页端
内容见 04_WebService 文件夹
利用 flask 将模型部署到网页端，在网页端加载本地图像数据并分类，将分类结果进行展示，