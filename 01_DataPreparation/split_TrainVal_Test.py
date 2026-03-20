"""
将每种病虫害的图像按照 6:2:2 分为训练、验证和测试，
训练和验证的图像分到同一个文件夹 TrainVal，测试分到 Test
所以 TrainVal 占总图像的比例为 0.8 ，Test 占 0.2

后期需要将 TrainVal 文件夹中的图像进一步划分为 Train 和 Value
此时 Train:Value:TrainVal=3:1:4
"""


import os
import random
import shutil
from pathlib import Path

def split_dataset(base_dir, TrainVal_ratio=0.8):
    """
    将每种病虫害的图像按比例划分到 TrainVal 和 Test 文件夹
    
    参数:
        base_dir: 包含 38 个病虫害文件夹的根目录路径
        TrainVal_ratio: 训练集比例，默认 0.8
    """
    # 创建 TrainVal 和 Test 主文件夹
    TrainVal_dir = Path(base_dir) / "TrainVal"
    Test_dir = Path(base_dir) / "Test"
    TrainVal_dir.mkdir(exist_ok=True)
    Test_dir.mkdir(exist_ok=True)
    
    # 获取所有病虫害文件夹
    pest_folders = [f for f in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, f)) and f not in ["TrainVal", "Test"]]
    
    print(f"发现 {len(pest_folders)} 个病虫害文件夹")
    
    for pest_folder in pest_folders:
        pest_path = Path(base_dir) / pest_folder
        
        # 获取该病虫害的所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in os.listdir(pest_path) 
                 if os.path.isfile(os.path.join(pest_path, f)) and 
                 Path(f).suffix.lower() in image_extensions]
        
        if not images:
            print(f"警告: 文件夹 '{pest_folder}' 中没有图片文件")
            continue
        
        # 打乱顺序
        random.shuffle(images)
        
        # 计算划分点
        split_idx = int(len(images) * TrainVal_ratio)
        TrainVal_images = images[:split_idx]
        Test_images = images[split_idx:]
        
        # 创建病虫害对应的子文件夹
        TrainVal_pest_dir = TrainVal_dir / pest_folder
        Test_pest_dir = Test_dir / pest_folder
        TrainVal_pest_dir.mkdir(exist_ok=True)
        Test_pest_dir.mkdir(exist_ok=True)
        
        # 复制训练集图片
        for img in TrainVal_images:
            src = pest_path / img
            dst = TrainVal_pest_dir / img
            shutil.copy2(src, dst)
        
        # 复制测试集图片
        for img in Test_images:
            src = pest_path / img
            dst = Test_pest_dir / img
            shutil.copy2(src, dst)
        
        print(f"{pest_folder}: 总图片 {len(images)}, "
              f"训练集 {len(TrainVal_images)} ({len(TrainVal_images)/len(images)*100:.1f}%), "
              f"测试集 {len(Test_images)} ({len(Test_images)/len(images)*100:.1f}%)")
    
    print(f"\n数据集划分完成！")
    print(f"训练集路径: {TrainVal_dir}")
    print(f"测试集路径: {Test_dir}")

def main():
    # 使用方法示例
    base_path = r"./color"  # 请修改为你的实际路径
    split_dataset(base_path, TrainVal_ratio=0.8)
    
    # 或者使用相对路径
    # split_dataset("./病虫害数据集", TrainVal_ratio=0.8)

if __name__ == "__main__":
    main()