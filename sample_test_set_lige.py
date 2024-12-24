import json
import random
import os
from PIL import Image
import numpy as np
from pathlib import Path
from data_reader import read_general
from prefix_instruction import lige_2x1_instruction

# 在文件开头添加全局计数器
count = 0

def load_jsonl(file_path):
    """读取jsonl文件"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_2x1_grid(img1, img2):
    """创建2x1的grid图像"""
    # 确保两张图片尺寸相同
    width = max(img1.size[0], img2.size[0])
    height = max(img1.size[1], img2.size[1])
    
    img1 = img1.resize((width, height))
    img2 = img2.resize((width, height))
    
    # 创建新图像
    new_img = Image.new('RGB', (width * 2, height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (width, 0))
    
    return new_img

def process_sample(sample, output_dir, sample_idx):
    """处理单个样本，生成2x1 grid组合"""
    global count  # 声明使用全局变量
    results = []
    
    # 读取原始图像
    image = Image.open(read_general(sample['image_path'])).convert('RGB')
    
    # 获取所有condition类型
    condition_types = list(sample['condition'].keys())
    
    # 为每个condition生成grid
    for cond_type in condition_types:
        try:
            # 读取condition图像
            cond_path = sample['condition'][cond_type]
            cond_image = Image.open(read_general(cond_path)).convert('RGB')
            
            # 只创建一种排列方式：condition在左，原图在右
            grid = create_2x1_grid(cond_image, image)
            
            # 保存grid图像
            grid_path = os.path.join(output_dir, f'{count}_{sample_idx}_{cond_type}.jpeg')
            grid.save(grid_path, format='JPEG', quality=95)
            print(grid_path)
            
            # 记录结果
            results.append({
                'input_path': cond_path,
                'target_path': sample['image_path'],
                'grid_path': grid_path,
                'prompt': lige_2x1_instruction[cond_type] + " The content of the image is : " + sample['prompt'],
                'condition_type': cond_type,
            })
            count += 1
        except Exception as e:
            print(f"Error processing {cond_type} for sample {sample_idx}: {str(e)}")
            continue
            
    return results

def main():
    # 设置路径
    input_jsonl = "/mnt/hwfile/alpha_vl/lizhongyu/in_context_control_images/flux_qinqi_5m/flux_qinqi_5m_condition_100k.jsonl"  # 请替换为实际的输入文件路径
    output_dir = "lige_test_gt"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    data = load_jsonl(input_jsonl)
    
    # 随机采样50个样本
    samples = random.sample(data, min(50, len(data)))
    
    # 处理所有样本
    all_results = []
    for idx, sample in enumerate(samples):
        results = process_sample(sample, output_dir, idx)
        all_results.extend(results)
    
    # 保存结果到json文件
    output_json = os.path.join("lige_test.json")
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"处理完成! 共生成 {len(all_results)} 个grid组合")
    print(f"结果已保存到: {output_json}")

if __name__ == "__main__":
    main() 