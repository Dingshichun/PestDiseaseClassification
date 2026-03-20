# 农业病虫害分类图像数据准备
数据集下载：公开数据集，如 **https://www.kaggle.com/datasets/emmarex/plantdisease**。包含 38 种植物病虫害约五万张图像。

数据集划分：训练：验证：测试 = 6:2:2，先将图像按照 8:2 划分为训练和验证：测试，后面再将训练和验证按照 6:2 划分为训练：验证。[图像划分脚本](./split_TrainVal_Test.py)

如果有必要，还可以生成类别索引文件 class_indices.json。[生成类别索引的脚本](./create_class_json.py)
