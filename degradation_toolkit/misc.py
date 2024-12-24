import os
import h5py
from PIL import Image
import numpy as np
import cv2
def convert_h5_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.h5'):
            # 构建输入文件的完整路径
            input_path = os.path.join(input_folder, filename)

            # 打开.h5文件并读取数据集
            with h5py.File(input_path, 'r') as file:
                # 获取数据集的形状
                # 遍历文件中的所有数据集
                for dataset_name in file:
                    # 构建输出文件的完整路径
                    output_path = os.path.join(output_folder, f"{filename}_{dataset_name}.png")

                    # 将数据集转换为图像
                    dataset = file[dataset_name]
                    image_data = dataset[()]  # 读取所有数据
                    num_images = image_data.shape[0]

                    # 遍历每张图片
                    for i in range(num_images):
                        # 构建输出文件的完整路径
                        

                        # 获取当前图片的数据
                        image = image_data[i]
                        for idx, sub_image in enumerate(image):
                            output_filename = f"{filename}_{i}_{idx}.png"
                            output_path = os.path.join(output_folder, output_filename)
                            scaled_image = (sub_image - np.min(sub_image)) / (np.max(sub_image) - np.min(sub_image))
                            scaled_image = np.round(scaled_image * 255).astype(np.uint8)
                            # 保存图像
                            cv2.imwrite(output_path, scaled_image)
                            print(f"Saved {output_path}")

# 指定输入文件夹和输出文件夹的路径
data_folder = 'data/superbench_v1/nskt32000_1024'
data_folder_list = os.listdir(data_folder)
for folder_name in data_folder_list:
    input_folder = os.path.join(data_folder, folder_name)
    output_folder = os.path.join(data_folder, folder_name + '_png')
    # 调用函数进行转换
    convert_h5_to_png(input_folder, output_folder)

data_folder = 'data/superbench_v1/cosmo_2048'
data_folder_list = os.listdir(data_folder)
for folder_name in data_folder_list:
    input_folder = os.path.join(data_folder, folder_name)
    output_folder = os.path.join(data_folder, folder_name + '_png')
    # 调用函数进行转换
    convert_h5_to_png(input_folder, output_folder)

data_folder = 'data/superbench_v1/cosmo_lres_sim_2048'
data_folder_list = os.listdir(data_folder)
for folder_name in data_folder_list:
    input_folder = os.path.join(data_folder, folder_name)
    output_folder = os.path.join(data_folder, folder_name + '_png')
    # 调用函数进行转换
    convert_h5_to_png(input_folder, output_folder)

