# import os
# import json
# import yaml
# from pathlib import Path
# from data_reader import read_general
# from PIL import Image

# def show_first_sample(folder_path):
#     """展示文件夹中每个json文件的第一个样本并生成yaml配置文件"""
#     folder = Path(folder_path)
    
#     # 准备yaml配置数据
#     yaml_config = {"META": []}
    
#     # 遍历文件夹中的所有json文件
#     test_set = []
#     count = 1
#     for json_file in folder.glob("*.json"):
#         print(f"\n文件名: {json_file.name}")
#         print("-" * 50)
        
#         # 添加到yaml配置
#         yaml_config["META"].append({
#             "path": str(json_file.absolute()),
#             "type": json_file.name.split(".")[0].replace("meta_info_", "")
#         })
        
#         try:
#             with open(json_file, "r", encoding="utf-8") as f:
#                 data = json.load(f)

#                 # if data[0]["target_path"] and data[0]["input_path"]:
#                 #     print(f"{json_file.name}: check!")
#                 # else:
#                 #     print(f"{json_file.name}: ERROR!")
#                 #     print(data[0])

#                 test_sampels = data[:2]
#                 for s in test_sampels:
#                     s["task_type"] = json_file.name.split(".")[0].replace("meta_info_", "")
#                 test_set = test_set + test_sampels
#                 print(len(test_set), json_file.name.split(".")[0].replace("meta_info_", ""))

#                 input_image_path = data[0]["input_path"]
#                 input_image = Image.open(read_general(input_image_path)).convert("RGB")
#                 target_image_path = data[0]["target_path"]
#                 target_image = Image.open(read_general(target_image_path)).convert("RGB")
#                 input_image.save(f"pixwizard_test_gt/{count}_{json_file.name.split('.')[0].replace('meta_info_', '')}_input.jpeg")
#                 target_image.save(f"pixwizard_test_gt/{count}_{json_file.name.split('.')[0].replace('meta_info_', '')}_target.jpeg")
#                 count += 1
#                 # text = data[0]["prompt"]
#                 # print(input_image.size)
#                 # print(target_image.size)

#                 # with open("pixwizard_statistics.txt", "a", encoding="utf-8") as f:
#                 #     f.write(f"{json_file.name}: {len(data)}\n")
#                 #     print(f"{json_file.name}: {len(data)}")
                
#                 # if isinstance(data, list) and len(data) > 0:
#                 #     print("第一个样本:")
#                 #     print(json.dumps(data[0], indent=2, ensure_ascii=False))
#                 # elif isinstance(data, dict):
#                 #     print("内容:")
#                 #     print(json.dumps(data, indent=2, ensure_ascii=False))
#                 # else:
#                 #     print("文件格式不是列表或字典")
                    
#         except Exception as e:
#             print(f"读取文件 {json_file.name} 时发生错误: {str(e)}")
    
#     # # 生成yaml文件
#     # output_yaml = "configs/data/pixwizard.yaml"
#     # with open(output_yaml, "w", encoding="utf-8") as f:
#     #     yaml.dump(yaml_config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
#     # print(f"\n已生成配置文件: {output_yaml}")
#     with open("pixwizard_test.json", "w", encoding="utf-8") as f:
#         json.dump(test_set, f, ensure_ascii=False, indent=4)

# if __name__ == "__main__":
#     folder_path = "/mnt/hwfile/alpha_vl/duruoyi/pixwizard"  # 替换为实际的文件夹路径
#     show_first_sample(folder_path)





# import os
# os.environ['HF_TOKEN'] = "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"
# import torch
# # os.environ['APEX_DISABLE_FUSED_LAYERNORM'] = '1'
# # os.environ['T5_USE_PYTORCH_LAYER_NORM'] = '1'
# from flux.util import load_ae, load_clip, load_flow_model, load_t5
# from diffusers import AutoencoderKL

# # t5 = load_t5(max_length=512).to('cpu')
# # clip = load_clip().to('cpu')
# model = load_flow_model("flux-dev-fill").to('cpu')
# # ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16)








# import json
# import random
# subset_list = [f"subjects200k-{i:05d}-of-00022" for i in range(22)]
# all_data = []
# good_quality_count = [0, 0, 0, 0]
# for subset in subset_list:
#     input_jsonl = f"/mnt/hwfile/alpha_vl/lizhongyu/in_context_control_images/subjects200k/{subset}/annotations/{subset}/{subset}-condition-v6.jsonl"
#     # 读取jsonl文件
#     with open(input_jsonl, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line.strip())
#             all_data.append(data)
#             # if data["quality_assessment"] is not None:
#             #     if data["quality_assessment"]["compositeStructure"] > 3:
#             #         good_quality_count[0] += 1
#             #     if data["quality_assessment"]["imageQuality"] > 3:
#             #         good_quality_count[1] += 1
#             #     if data["quality_assessment"]["objectConsistency"] > 3: 
#             #         good_quality_count[2] += 1
#             #     if data["quality_assessment"]["compositeStructure"] > 3 and data["quality_assessment"]["imageQuality"] > 3 and data["quality_assessment"]["objectConsistency"] > 3:
#             #         good_quality_count[3] += 1
#             #         all_data.append(data)
#     print(input_jsonl, len(all_data))
# print(f"总共读取了 {len(all_data)} 条数据")
# print(good_quality_count)

# # 随机打乱数据
# random.shuffle(all_data)
# # 如果需要保存合并后的数据
# with open('/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_train_v6.json', 'w', encoding='utf-8') as f:
#     json.dump(all_data[:len(all_data)-50], f, ensure_ascii=False, indent=4)
# with open('/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_test_v6.json', 'w', encoding='utf-8') as f:
#     json.dump(all_data[len(all_data)-50:], f, ensure_ascii=False, indent=4)


import json

def count_non_empty_values(data_list):
    # 初始化结果字典
    result = {}
    
    # 遍历列表中的每个字典
    for item in data_list:
        # 递归处理每个字典
        count_dict_values(item, result)
    
    return result

def count_dict_values(d, result, prefix='', depth=0):
    # 如果递归深度超过3层，直接返回
    if depth >= 2:
        return
        
    # 遍历当前字典的所有键值对
    for key, value in d.items():
        # 构建完整的键路径
        current_key = f"{prefix}.{key}" if prefix else key
        
        # 如果值是字典，递归处理（并增加深度计数）
        if value not in [None, '', [], {}] and depth == 1:
            result[current_key] = result.get(current_key, 0) + 1
        if isinstance(value, dict):
            count_dict_values(value, result, current_key, depth + 1)

with open("/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_train_v6.json", "r", encoding="utf-8") as f:
    data = json.load(f)
result = count_non_empty_values(data)
for key, count in result.items():
    print(f"{key}: {count} {count / len(data)}")


# from datasets import Dataset
# import os
# from PIL import Image
# import io

# # 创建保存图片的目标文件夹
# output_dir = "/mnt/hwfile/alpha_vl/duruoyi/subjects200k/images_all/"
# os.makedirs(output_dir, exist_ok=True)

# subset_list = [f"data-{i:05d}-of-00022" for i in range(22)]
# for i in range(22):
#     # 加载数据集
#     ds = Dataset.from_file(f"/mnt/hwfile/alpha_vl/duruoyi/subjects200k/train/{subset_list[i]}.arrow")
    
#     # 遍历数据集中的所有图片
#     for idx, sample in enumerate(ds):
#         # 获取图片数据
#         image = sample['image']
        
#         # # 将字节数据转换为PIL图像
#         # image = Image.open(io.BytesIO(image))
        
#         # 构建保存路径（使用数据集编号和图片索引作为文件名）
#         save_path = os.path.join(output_dir, f"{subset_list[i]}-{idx}.jpg")
        
#         # 保存图片
#         image.save(save_path)
        
#         # 每100张图片打印一次进度
#         if idx % 100 == 0:
#             print(f"处理数据集 {i+1}/22, 已保存 {idx} 张图片")





# import json

# # 读取json文件
# with open("/mnt/hwfile/alpha_vl/duruoyi/flux_qinqi_5m.json", "r", encoding="utf-8") as f:
#     data = f.read()
# json_data = json.loads(data)

# # 提取prompt并保存到txt文件
# with open("captions_5m2k.txt", "w", encoding="utf-8") as f:
#     for idx, sample in enumerate(json_data):
#         if idx > 2000:
#             break
#         if "prompt" in sample:
#             if sample["prompt"] != "":
#                 f.write(sample["prompt"].replace("\n", " ") + "\n")

# print("已将prompt保存到prompts.txt文件中")




# import numpy as np
# from PIL import Image
# from data_reader import read_general
# import json
# json_file = "./chunlian_ocr_en.json"
# with open(json_file, "r", encoding="utf-8") as f:
#     data = f.read()
# data = json.loads(data)
# print(len(data))
# cannot_open_count = 0
# for idx, data_item in enumerate(data):
#     if idx > 5000:
#         break
#     if idx % 50 != 0:
#         continue
#     image_path = "lc2:" + data_item["image_path"]
#     image_data = read_general(image_path)
#     image = Image.open(image_data).convert("RGB")

#     try:
#         rendered_image_path = "lc2:" + data_item["black_bg_output_path"]
#         rendered_image_data = read_general(rendered_image_path)
#         rendered_image = Image.open(rendered_image_data).convert("RGB")
#     except Exception as e:
#         print(e)
#         cannot_open_count += 1
#         continue

#     # 创建一个新的空白图像,宽度是两张图的和,高度取最大值
#     width = image.size[0] + rendered_image.size[0]
#     height = max(image.size[1], rendered_image.size[1])
#     merged_image = Image.new('RGB', (width, height))
    
#     # 将两张图片粘贴到新图像上
#     merged_image.paste(image, (0, 0))
#     merged_image.paste(rendered_image, (image.size[0], 0))
    
#     # 保存拼接后的图片
#     merged_image.save(f"chunlian_ocr/{idx:05d}_merged.jpg")


# from os.path import splitext
# from petrel_client.client import Client
# from PIL import Image
# import PIL
# import json
# import json
# import csv
# from io import BytesIO
# from data_reader import *
 
# conf_path = './petreloss.conf'
# client = Client(conf_path) # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件

# def read_dir(url='lc2:s3://shitian_poster/ocr_results_rendered_images/'):
#     contents = client.list(url)
#     # print(len(list(contents)))
#     for idx, content in enumerate(contents):
#         # if idx > 0:
#         #     break
#         if content.endswith('/'):
#             print('directory:', content)
#         else:
#             print('object:', content)

# read_dir()