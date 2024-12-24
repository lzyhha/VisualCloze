import json
import random
import os
from PIL import Image
import numpy as np
from pathlib import Path
from data_reader import read_general

# 在文件开头添加全局计数器
count = 0

def load_json(file_path):
    """读取json文件"""
    data = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

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

def create_nxn_grid(img_list, grid_type="2x1"):
    """将图片列表按顺序排列成nxn的grid"""
    # 确保所有图片尺寸相同
    width, height = 512, 512
    

    # 根据grid_type确定行列数
    cols, rows = grid_type.split("x")
    cols, rows = int(cols), int(rows)
        
    # 调整所有图片尺寸
    resized_imgs = [img.resize((width, height)) for img in img_list]
    
    # 创建新图像
    new_img = Image.new('RGB', (width * cols, height * rows))
    
    # 按顺序粘贴图片
    for idx, img in enumerate(resized_imgs):
        row = idx // cols
        col = idx % cols
        new_img.paste(img, (col * width, row * height))
    
    return new_img

def process_sample(sample, output_dir, sample_idx):
    """处理单个样本，生成2x1 grid组合"""
    global count  # 声明使用全局变量
    results = []
    
    # 读取原始图像
    image_path = f"/mnt/hwfile/alpha_vl/duruoyi/subjects200k/images_all/{sample['image_name']}.jpg"
    image = Image.open(image_path).convert('RGB')

    # 将原始图像分割为左右两半
    width, height = image.size
    half_width = width // 2
    
    # 分割图像
    target_image = image.crop((0, 0, half_width, height))
    ref_image = image.crop((half_width, 0, width, height))
    
    # 对两个图像进行center crop,去掉8像素的padding
    padding = 8
    target_image = target_image.crop((padding, padding, half_width-padding, height-padding))
    ref_image = ref_image.crop((padding, padding, half_width-padding, height-padding))

    # 获取所有condition类型
    condition_types = list(sample['condition'].keys())
    
    # 3x1 grid
    task_type = "3x1"
    for cond_type in condition_types:
        try:
            # 读取condition图像
            cond_path = sample['condition'][cond_type]
            cond_image = Image.open(read_general(cond_path)).convert('RGB')
            
            # 只创建一种排列方式：condition在左，原图在右
            grid = create_nxn_grid([cond_image, ref_image, target_image], task_type)
            
            # 保存grid图像
            grid_path = os.path.join(output_dir, f'{count}_{sample_idx}_{cond_type}.jpeg')
            grid.save(grid_path, format='JPEG', quality=95)
            print(grid_path)

            cols, rows = task_type.split("x")   
            cols, rows = int(cols), int(rows)
            prefix_instruction = f"Demonstration of the conditional image generation process, {cols*rows} images form a grid with {rows} rows and {cols} columns, side by side. The content of the last image is: "
            
            # 记录结果
            results.append({
                'cond_path': cond_path,
                'pair_path': sample['image_name'],
                'grid_path': grid_path,
                'prompt': prefix_instruction + sample['description']['description_0'],
                'condition_type': cond_type,
            })
            count += 1
        except Exception as e:
            print(f"Error processing {cond_type} for sample {sample_idx}: {str(e)}")
            continue
            
    return results

def main():
    # 设置路径
    input_json = "/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_test.json"  # 请替换为实际的输入文件路径
    output_dir = "object200k_test_gt_3x1"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    data = load_json(input_json)
    
    # 随机采样50个样本
    samples = random.sample(data, min(50, len(data)))
    
    # 处理所有样本
    all_results = []
    for idx, sample in enumerate(samples):
        results = process_sample(sample, output_dir, idx)
        all_results.extend(results)
    
    # 保存结果到json文件
    output_json = os.path.join("object200k_test_3x1.json")
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"处理完成! 共生成 {len(all_results)} 个grid组合")
    print(f"结果已保存到: {output_json}")

if __name__ == "__main__":
    main() 