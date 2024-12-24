import yaml
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def count_json_items(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data)


def balance_dataset_sizes(yaml_path, target_size, final_size):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    file_counts = {}

    # 统计每个文件的项目数量
    for item in config['META']:
        json_path = item['path']
        count = count_json_items(json_path)
        file_counts[json_path] = count

    # 计算scale和ratio
    for item in config['META']:
        json_path = item['path']
        count = file_counts[json_path]

        # 计算scale，确保大于等于1
        scale = max(1, target_size / count)
        item['scale'] = round(scale, 3)

        # 计算ratio，确保小于1
        ratio = min(0.999, final_size / (count * scale))
        item['ratio'] = round(ratio, 3)

    # 保存更新后的YAML文件
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return file_counts, config


def analyze_dataset_statistics(yaml_path):

    # 读取配置文件
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # 收集数据集大小信息
    sizes = []
    for item in config['META']:
        json_path = item['path']
        with open(json_path, 'r') as f:
            data = json.load(f)
            sizes.append(len(data))

    # 计算统计信息
    stats = {"最小值": np.min(sizes), "最大值": np.max(sizes), "中位数": np.median(sizes), "平均值": np.mean(sizes)}

    # 打印统计信息
    print("\n数据集统计信息:")
    for key, value in stats.items():
        print(f"{key}: {value:,.0f}")

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sizes)), sizes)
    plt.title("各数据集样本数量分布")
    plt.xlabel("数据集索引")
    plt.ylabel("样本数量")

    # 添加数值标签
    for i, v in enumerate(sizes):
        plt.text(i, v, f'{v:,}', ha='center', va='bottom', rotation=45)

    # 自动调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('dataset_distribution.png')
    plt.close()

    return stats


# 使用函数
yaml_path = 'options/data_options/006_OmniLV_2B_mGPT_7B_1024_genlv_sub.yml'
stats = analyze_dataset_statistics(yaml_path)
print(stats)
# yaml_path = 'options/data_options/006_OmniLV_2B_mGPT_7B_1024_genlv_sub.yml'
# target_size = 50000  # 设置目标数据集大小为5万
# final_size = 20000  # 设置最终采样大小为2万
# file_counts, updated_config = balance_dataset_sizes(yaml_path, target_size, final_size)

# # 打印每个数据集的信息
# print("\n每个数据集的信息:")
# for item in updated_config['META']:
#     path = item['path']
#     count = file_counts[path]
#     scale = item['scale']
#     ratio = item['ratio']
#     scaled_size = count * scale
#     final_size = scaled_size * ratio
#     print(f"{path}:")
#     print(f"  原始大小: {count}")
#     print(f"  scale: {scale:.3f}")
#     print(f"  扩展后大小: {scaled_size:.2f}")
#     print(f"  ratio: {ratio:.3f}")
#     print(f"  最终大小: {final_size:.2f}")
