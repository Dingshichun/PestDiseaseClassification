import os
import json
from pathlib import Path

def save_folder_names_to_json(base_dir, output_path="class_indices.json"):
    """
    将病虫害文件夹名称保存到JSON文件
    
    参数:
        base_dir: 包含病虫害文件夹的根目录路径
        output_path: JSON文件保存路径
    """
    # 获取所有文件夹，排除 Train、Test 和隐藏文件夹
    all_items = os.listdir(base_dir)
    
    # 筛选出文件夹，并排除 Train 和 Test
    folder_names = []
    for item in all_items:
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item not in ["Train", "Test"] and not item.startswith("."):
            folder_names.append(item)
    
    # 排序以确保一致性
    folder_names.sort()
    
    # 创建字典，索引从 0 开始
    folder_dict = {str(i): name for i, name in enumerate(folder_names)}
    
    # 保存到JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(folder_dict, f, ensure_ascii=False, indent=2)
    
    print(f"已保存 {len(folder_dict)} 个文件夹名称到 {output_path}")
    print("JSON内容预览:")
    print(json.dumps(folder_dict, ensure_ascii=False, indent=2)[:500] + "...")
    
    return folder_dict

# 使用示例
if __name__ == "__main__":
    # 请修改为你的实际路径
    base_path = r"./color"
    
    # 调用函数
    folder_dict = save_folder_names_to_json(base_path, "class_indices.json")
    
    # 可选：同时打印详细信息
    print(f"\n详细列表 (共{len(folder_dict)}个):")
    for idx, name in folder_dict.items():
        print(f"{idx}: {name}")