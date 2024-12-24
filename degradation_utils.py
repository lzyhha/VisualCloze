import numpy as np
import cv2
import random
from PIL import Image

from degradation_toolkit.add_degradation_various import *
from degradation_toolkit.image_operators import *
from degradation_toolkit.x_distortion import *

degradation_list1 = [
    'blur',
    'noise',
    'compression',
    'brighten',
    'darken',
    'spatter',
    'contrast_strengthen',
    'contrast_weaken',
    'saturate_strengthen',
    'saturate_weaken',
    'oversharpen',
    'pixelate',
    'quantization',
]
degradation_list2 = [
    'Rain', 
    'Ringing', 
    'r_l', 
    'Inpainting', 
    'mosaic', 
    'SRx2', 
    'SRx4',
    'GaussianNoise',
    'GaussianBlur',
    'JPEG',
    'Resize',
    'SPNoise',
    'LowLight',
    'PoissonNoise',
    'gray',
    'ColorDistortion',
]
degradation_list3 = [
    'Laplacian', 
    'Canny',
    'Sobel',
    'Defocus',
    'Mosaic',
    'Barrel',
    'Pincushion',
    'Spatter',
    'Elastic',
    'Frost',
    'Contrast',
]
degradation_list4 = [
    'flip',
    'rotate90',
    'rotate180',
    'rotate270',
    'identity',
]
all_degradation_types = degradation_list1 + degradation_list2 + degradation_list3 + degradation_list4

def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.0).round())

def uint2single(img):
    return np.float32(img / 255.0)

def add_x_distortion_single_images(img_gt1, deg_type):
    # np.uint8, BGR
    x_distortion_dict = distortions_dict
    severity = random.choice([1, 2, 3, 4, 5])
    if deg_type == 'compression' or deg_type == "quantization":
        severity = min(3, severity)
    deg_type = random.choice(x_distortion_dict[deg_type])

    img_gt1 = cv2.cvtColor(img_gt1, cv2.COLOR_BGR2RGB)
    img_lq1 = globals()[deg_type](img_gt1, severity)

    img_gt1 = cv2.cvtColor(img_gt1, cv2.COLOR_RGB2BGR)
    img_lq1 = cv2.cvtColor(img_lq1, cv2.COLOR_RGB2BGR)

    return img_lq1, img_gt1, deg_type


def add_degradation_single_images(img_gt1, deg_type):
    # np.float32
    if deg_type == 'Rain':
        value = random.uniform(40, 200)
        img_lq1 = add_rain(img_gt1, value=value)
    elif deg_type == 'Ringing':
        img_lq1 = add_ringing(img_gt1)
    elif deg_type == 'r_l':
        img_lq1 = r_l(img_gt1)
    elif deg_type == 'Inpainting':
        l_num = random.randint(20, 50)
        l_thick = random.randint(10, 20)
        img_lq1 = inpainting(img_gt1, l_num=l_num, l_thick=l_thick)
    elif deg_type == 'mosaic':
        img_lq1 = mosaic_CFA_Bayer(img_gt1)
    elif deg_type == 'SRx2':
        H, W, _ = img_gt1.shape
        img_lq1 = cv2.resize(img_gt1, (W//2, H//2), interpolation=cv2.INTER_CUBIC)
        img_lq1 = cv2.resize(img_lq1, (W, H), interpolation=cv2.INTER_CUBIC)
    elif deg_type == 'SRx4':
        H, W, _ = img_gt1.shape
        img_lq1 = cv2.resize(img_gt1, (W//4, H//4), interpolation=cv2.INTER_CUBIC)
        img_lq1 = cv2.resize(img_lq1, (W, H), interpolation=cv2.INTER_CUBIC)

    elif deg_type == 'GaussianNoise':
        level = random.uniform(10, 50)
        img_lq1 = add_Gaussian_noise(img_gt1, level=level)
    elif deg_type == 'GaussianBlur':
        sigma = random.uniform(2, 4)
        img_lq1 = iso_GaussianBlur(img_gt1, window=15, sigma=sigma)
    elif deg_type == 'JPEG':
        level = random.randint(10, 40)
        img_lq1 = add_JPEG_noise(img_gt1, level=level)
    elif deg_type == 'Resize':
        img_lq1 = add_resize(img_gt1)
    elif deg_type == 'SPNoise':
        img_lq1 = add_sp_noise(img_gt1)
    elif deg_type == 'LowLight':
        lum_scale = random.uniform(0.3, 0.4)
        img_lq1 = low_light(img_gt1, lum_scale=lum_scale)
    elif deg_type == 'PoissonNoise':
        img_lq1 = add_Poisson_noise(img_gt1, level=2)
    elif deg_type == 'gray':
        img_lq1 = cv2.cvtColor(img_gt1, cv2.COLOR_BGR2GRAY)
        img_lq1 = np.expand_dims(img_lq1, axis=2)
        img_lq1 = np.concatenate((img_lq1, img_lq1, img_lq1), axis=2)
    elif deg_type == 'None':
        img_lq1 = img_gt1
    elif deg_type == 'ColorDistortion':
        # 随机选择一个操作: 交换通道顺序或修改单个通道
        if random.random() < 0.5:
            # 随机交换通道顺序
            channels = list(range(3))
            random.shuffle(channels) 
            img_lq1 = img_gt1[..., channels]
        else:
            # 随机选择一个通道变成黑色或白色
            channel = random.randint(0, 2)
            img_lq1 = img_gt1.copy()
            # 随机选择黑色或白色
            if random.random() < 0.5:
                img_lq1[..., channel] = 0  # 黑色
            else:
                img_lq1[..., channel] = 1  # 白色
    else:
        print('Error!', '-', deg_type, '-')
        exit()
    img_lq1 = np.clip(img_lq1 * 255, 0, 255).round().astype(np.uint8)
    img_lq1 = img_lq1.astype(np.float32) / 255.0
    img_gt1 = np.clip(img_gt1 * 255, 0, 255).round().astype(np.uint8)
    img_gt1 = img_gt1.astype(np.float32) / 255.0

    return img_lq1, img_gt1


def calculate_operators_single_images(img_gt1, deg_type):
    # np.uint8
    img_gt1 = img_gt1.copy()
    
    if deg_type == 'Laplacian':
        img_lq1 = Laplacian_edge_detector(img_gt1)
    elif deg_type == 'Canny':
        img_lq1 = Canny_edge_detector(img_gt1)
    elif deg_type == 'Sobel':
        img_lq1 = Sobel_edge_detector(img_gt1)
    elif deg_type == 'Defocus':
        img_lq1 = defocus_blur(img_gt1, level=(3, 0.2))
    elif deg_type == 'Mosaic':
        img_lq1 = mosaic_CFA_Bayer(img_gt1)
    elif deg_type == 'Barrel':
        img_lq1 = simulate_barrel_distortion(img_gt1, k1=0.1, k2=0.05)
    elif deg_type == 'Pincushion':
        img_lq1 = simulate_pincushion_distortion(img_gt1, k1=-0.1, k2=-0.05)
    elif deg_type == 'Spatter':
        img_lq1 = uint2single(spatter((img_gt1), severity=1))
    elif deg_type == 'Elastic':
        img_lq1 = elastic_transform((img_gt1), severity=4)
    elif deg_type == 'Frost':
        img_lq1 = uint2single(frost(img_gt1, severity=4))
    elif deg_type == 'Contrast':
        img_lq1 = adjust_contrast(img_gt1, clip_limit=4.0, tile_grid_size=(4, 4))
        
    # check zero images
    if np.mean(img_lq1).astype(np.float16) == 0:
        print(deg_type, 'prompt&query zero images.')
        img_lq1 = img_gt1.copy()
        
    return img_lq1, img_gt1

def add_degradation(image, deg_type):
    if deg_type in degradation_list1:
        list_idx = 1
        img_lq1, _, _ = add_x_distortion_single_images(np.copy(image), deg_type)
        img_lq1 = uint2single(img_lq1)
    elif deg_type in degradation_list2:
        list_idx = 2
        img_lq1, _ = add_degradation_single_images(np.copy(uint2single(image)), deg_type)
    elif deg_type in degradation_list3:
        list_idx = 3
        if deg_type in ['Laplacian', 'Canny', 'Sobel', 'Frost']:
            # print(np.copy(image).max(), np.copy(image).min())
            img_lq1, _ = calculate_operators_single_images(np.copy(image), deg_type)
            # print(img_lq1.max(), img_lq1.min())
        else:
            # print(np.copy(uint2single(image)).max(), np.copy(uint2single(image)).min())
            img_lq1, _ = calculate_operators_single_images(np.copy(uint2single(image)), deg_type)
            # print(img_lq1.max(), img_lq1.min())
        if img_lq1.max() > 1:
            img_lq1 = uint2single(img_lq1)
    elif deg_type in degradation_list4:
        list_idx = 4
        img_lq1 = np.copy(uint2single(image))
        if deg_type == 'flip':
            img_lq1 = np.flip(img_lq1, axis=1)
        elif deg_type == 'rotate90':
            img_lq1 = np.rot90(img_lq1, k=1)
        elif deg_type == 'rotate180':
            img_lq1 = np.rot90(img_lq1, k=2)
        elif deg_type == 'rotate270':
            img_lq1 = np.rot90(img_lq1, k=3)
        elif deg_type == 'identity':
            pass
    # 保存处理后的图片
    return Image.fromarray(single2uint(img_lq1)), list_idx

def process_image(image_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用PIL读取图片
    pil_image = Image.open(image_path)
    # 转换为numpy数组，并确保是BGR格式(因为后续代码假设使用BGR)
    image = np.array(pil_image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 处理每种degradation
    for deg_type in all_degradation_types:
        # print(deg_type)
        
        img_lq1, list_idx = add_degradation(image, deg_type)
        
        # 保存处理后的图片
        output_path = os.path.join(output_dir, f"degradation_list{list_idx}_{deg_type}.png")
        img_lq1.save(output_path)

if __name__ == "__main__":
    # 设置输入图片路径和输出目录
    image_path = "./degradation_toolkit/test.png"  # 替换为你的图片路径
    output_dir = "./degradation_toolkit/degradation_results"      # 输出目录名
    
    process_image(image_path, output_dir)
    print("处理完成！")