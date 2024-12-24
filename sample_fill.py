import argparse
import functools
import json
import math
import os
import random
import socket
import time

from einops import rearrange, repeat
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from safetensors.torch import load_file as load_sft
from tqdm import tqdm
from einops import rearrange, repeat

from data import ItemProcessor
from flux.model import Flux, FluxParams
from flux.sampling import prepare_modified
from flux.util import configs, load_clip, load_t5, load_flow_model
from transport import Sampler, create_transport
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop

from data_reader import read_general
from data.prefix_instruction import get_layout_instruction, get_task_instruction, get_content_instruction, get_image_prompt, task_dicts, condition_list, degradation_list, style_list, editing_list, test_task_dicts
from degradation_utils import add_degradation

def resize_with_aspect_ratio(img, resolution, divisible=16, aspect_ratio=None):
    """调整图片大小,保持长宽比,使面积接近resolution**2,且宽高能被16整除
    
    Args:
        img: PIL Image 或 torch.Tensor (C,H,W)/(B,C,H,W)
        resolution: 目标分辨率
        divisible: 确保输出尺寸能被此数整除
    
    Returns:
        调整大小后的图像,与输入类型相同
    """
    # 检查输入类型并获取尺寸
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        if img.dim() == 3:
            c, h, w = img.shape
            batch_dim = False
        else:
            b, c, h, w = img.shape
            batch_dim = True
    else:
        w, h = img.size
        
    # 计算新尺寸
    if aspect_ratio is None:
        aspect_ratio = w / h
    target_area = resolution * resolution
    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)
    
    # 确保能被divisible整除
    new_w = max(new_w // divisible, 1) * divisible
    new_h = max(new_h // divisible, 1) * divisible
    
    # 根据输入类型调整大小
    if is_tensor:
        # 使用torch的插值方法
        mode = 'bilinear'
        align_corners = False
        if batch_dim:
            return F.interpolate(img, size=(new_h, new_w), 
                               mode=mode, align_corners=align_corners)
        else:
            return F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                               mode=mode, align_corners=align_corners).squeeze(0)
    else:
        # 使用PIL的LANCZOS重采样
        return img.resize((new_w, new_h), Image.LANCZOS)

class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform, resolution=512):
        self.image_transform = transform
        self.resolution = resolution
    
    def pixwizard_process_item(self, data_item):
        input_image_path = data_item["input_path"]
        input_image = Image.open(read_general(input_image_path)).convert("RGB")
        target_image_path = data_item["target_path"]
        target_image = Image.open(read_general(target_image_path)).convert("RGB")
        text = data_item["prompt"]
        return input_image, target_image, text

    def lige_process_item(self, data_item):
        random.shuffle(condition_list)
        for condition in condition_list:
            try:
                input_image_path = data_item["condition"][condition]
                input_image = Image.open(read_general(input_image_path)).convert("RGB")
                break 
            except Exception as e:
                continue 
        
        target_image_path = data_item["image_path"]
        target_image = Image.open(read_general(target_image_path)).convert("RGB")
        text = lige_2x1_instruction[condition] + " The content of the image is : " + data_item["prompt"]
        return input_image, target_image, text

    def chunlian_ocr_process_item(self, data_item):
        # data_item = data_item[0]
        target_image_path = "lc2:" + data_item["image_path"]
        target_image = Image.open(read_general(target_image_path)).convert("RGB")
        try:
            if random.random() < 1:
                input_image_path = "lc2:" + data_item["black_bg_output_path"]
                input_image = Image.open(read_general(input_image_path)).convert("RGB")
            else:
                input_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
        except Exception as e:
            input_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
        text = data_item["caption_en"]
        return input_image, target_image, text

    def jimeng_ocr_process_item(self, data_item):
        # data_item = data_item[0]
        target_image_path = "lc2:" + data_item["image_path"]
        target_image = Image.open(read_general(target_image_path)).convert("RGB")
        try:
            if random.random() < 1:
                input_image_path = "lc2:" + data_item["rendered_image_path"].replace("/rendered_images/", "/")
                input_image = Image.open(read_general(input_image_path)).convert("RGB")
            else:
                input_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
        except Exception as e:
            input_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
        text = data_item["caption_en"]
        return input_image, target_image, text

    def get_image_object200k(self, data_item, image_type):
        if image_type == "target":
            source_image_path = f"/mnt/hwfile/alpha_vl/duruoyi/subjects200k/images_all/{data_item['image_name']}.jpg"
            image = Image.open(source_image_path).convert('RGB')
            width, height = image.size
            half_width = width // 2
            target_image = image.crop((0, 0, half_width, height))
            padding = 8
            target_image = target_image.crop((padding, padding, half_width-padding, height-padding))
            return [target_image]
        elif image_type == "reference":
            source_image_path = f"/mnt/hwfile/alpha_vl/duruoyi/subjects200k/images_all/{data_item['image_name']}.jpg"
            image = Image.open(source_image_path).convert('RGB')
            width, height = image.size
            half_width = width // 2
            ref_image = image.crop((half_width, 0, width, height))
            padding = 8
            ref_image = ref_image.crop((padding, padding, half_width-padding, height-padding))
            return [ref_image]
        elif "qwen" in image_type:
            try:
                if "bbox" in image_type:
                    source_image_path = data_item["condition"]["qwen_grounding_caption"]["qwen_grounding_caption_bbox"]
                elif "mask" in image_type:
                    source_image_path = data_item["condition"]["qwen_grounding_caption"]["qwen_grounding_caption_mask"]
                image = Image.open(read_general(source_image_path)).convert("RGB")
            except Exception as e:
                image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
            return [image]
        elif "frontground" in image_type or "background" in image_type:
            mask_path = data_item["condition"]["bmbg_background_removal"]
            try:
                # 读取mask并转换为单通道灰度图
                mask = Image.open(read_general(mask_path)).convert("L")
            except Exception as e:
                mask = Image.new('L', (self.resolution, self.resolution), 0)
                
            # 读取并裁剪原始图像
            source_image_path = f"/mnt/hwfile/alpha_vl/duruoyi/subjects200k/images_all/{data_item['image_name']}.jpg"
            image = Image.open(source_image_path).convert('RGB')
            width, height = image.size
            half_width = width // 2
            target_image = image.crop((0, 0, half_width, height))
            padding = 8
            target_image = target_image.crop((padding, padding, half_width-padding, height-padding))
            
            # 将mask转换为numpy数组并归一化到0-1
            mask_np = np.array(mask).astype(np.float32) / 255.0
            
            # 根据image type返回前景或背景
            if "frontground" in image_type:
                # 前景: 保留mask为1的部分
                mask_np = mask_np[..., None]  # 添加通道维度
                result = Image.fromarray((np.array(target_image) * mask_np).astype(np.uint8))
            else:
                # 背景: 保留mask为0的部分
                mask_np = 1 - mask_np
                mask_np = mask_np[..., None]  # 添加通道维度
                result = Image.fromarray((np.array(target_image) * mask_np).astype(np.uint8))
            return [result]
        elif image_type in style_list:
            try:
                if image_type == "InstantStyle":
                    source_dict = data_item["condition"]["instantx_style"]
                elif image_type == "ReduxStyle":
                    source_dict = data_item["condition"]["flux_redux_style_shaping"]
                style_idx = random.randint(0, 2)
                style_image = source_dict["style_path"][style_idx]
                style_image = Image.open(read_general(style_image)).convert("RGB")
                target_image = source_dict["image_name"][style_idx]
                target_image = Image.open(read_general(target_image)).convert("RGB")
            except Exception as e:
                style_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
                target_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
            return [style_image, target_image]
        elif image_type in editing_list:
            try:
                if image_type == "DepthEdit":
                    editing_image_path = data_item["condition"]["flux_dev_depth"]
                elif image_type == "FrontEdit":
                    editing_image_path = random.choice(data_item["condition"]["qwen_subject_replacement"]["image_name"])
                editing_image = Image.open(read_general(editing_image_path)).convert('RGB')
            except Exception as e:
                editing_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
            return [editing_image]
        elif image_type in condition_list:
            try:
                cond_image = data_item["condition"][image_type]
                cond_image = Image.open(read_general(cond_image)).convert("RGB")
            except Exception as e:
                cond_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
            return [cond_image]
        elif image_type in degradation_list:
            source_image_path = f"/mnt/hwfile/alpha_vl/duruoyi/subjects200k/images_all/{data_item['image_name']}.jpg"
            image = Image.open(source_image_path).convert('RGB')
            width, height = image.size
            half_width = width // 2
            target_image = image.crop((0, 0, half_width, height))
            padding = 8
            target_image = target_image.crop((padding, padding, half_width-padding, height-padding))
            deg_image, _ = add_degradation(np.array(target_image), image_type)
            return [deg_image]

                
    def object200k_process_item(self, data_item, image_type_list=None, context_num=1):
        text_emb = None
        # context_num_prob = [0.3, 0.4, 0.3]  # 概率分别对应context_num=1,2,3
        # context_num = random.choices([1, 2, 3], weights=context_num_prob)[0]
        task_weight_prob = [task["sample_weight"] for task in task_dicts]  # 概率分别对应task_weight=0.3,1
        block_list = ["depth_anything_v2"]
        if image_type_list is None:
            while True:
                task_type = random.choices(task_dicts, weights=task_weight_prob)[0]
                image_type_list = random.choice(task_type["image_list"])
                if not any(block_type in image_type_list for block_type in block_list):
                    break
        image_list = [[] for _ in range(context_num)]
        for i in range(context_num):
            for image_type in image_type_list:
                images = self.get_image_object200k(data_item[i], image_type) 
                images = [resize_with_aspect_ratio(image, self.resolution, aspect_ratio=1.0) for image in images]
                image_list[i] += images
        image_prompt_list = []
        for image_type in image_type_list:
            image_prompt_list += get_image_prompt(image_type)
        image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]

        # shuffle n-1 elements
        indices = list(range(len(image_prompt_list)-1))
        random.shuffle(indices)
        for i in range(context_num):
            image_list[i][:len(image_prompt_list)-1] = [image_list[i][j] for j in indices]
        image_prompt_list[:len(image_prompt_list)-1] = [image_prompt_list[j] for j in indices]

        # 对每个图片应用transform并拼接
        processed_images = []
        for images in image_list:
            transformed_row = []
            for img in images:
                try:
                    transformed = self.image_transform(img)
                except Exception as e:
                    print("error image_type: ", image_type_list)
                transformed_row.append(transformed)
            row = torch.cat(transformed_row, dim=2)  # 在宽度维度上拼接
            processed_images.append(row)
        image = torch.cat(processed_images, dim=1)  # 在高度维度上拼接

        instruction = get_layout_instruction(context_num, len(image_type_list))
        if random.random() < 0.8:
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            instruction = instruction + " " + get_task_instruction(condition_prompt, target_prompt)
        if random.random() < 0.8 and image_type_list[-1] == "target":
            instruction = instruction + " " + get_content_instruction() + data_item[i]['description']['item'] + " " + data_item[i]['description']['description_0']
        
        return image, instruction, text_emb

    def ood_process_item(self, data_item, image_type_list=None, context_num=1):
        text_emb = None
        if "easydrawingguides" in data_item[0]["target"]:
            image_list = [[] for _ in range(context_num)]
            for i in range(context_num):
                for url in [data_item[i]["condition"][0], data_item[i]["condition"][len(data_item[i]["condition"]) // 2], data_item[i]["condition"][-1]]:
                    cond_image = Image.open(read_general(url)).convert('RGB')
                    image_list[i].append(cond_image)
                target_image = Image.open(read_general(data_item[i]["target"])).convert('RGB')
                image_list[i].append(target_image)
                image_list[i] = [resize_with_aspect_ratio(image, self.resolution, aspect_ratio=1.0) for image in image_list[i]]
            rows = context_num
            cols = 4
            content = data_item[-1]["prompt"]
            instruction = f"A grid layout with {rows} rows and {cols} columns, displaying {cols*rows} images arranged side by side. Each line illustrates a painting process, from rough lines to detailed drawings. The content of the last image in the final row is: {content}"
            print(instruction)
        elif "9x9_face" in data_item[0]["target"]:
            image_list = [[] for _ in range(context_num)]
            for i in range(context_num):
                for url in data_item[i]["condition"]:
                    cond_image = Image.open(read_general(url)).convert('RGB')
                    image_list[i].append(cond_image)
                target_image = Image.open(read_general(data_item[i]["target"])).convert('RGB')
                image_list[i].append(target_image)
                image_list[i] = [resize_with_aspect_ratio(image, self.resolution, aspect_ratio=1.0) for image in image_list[i]]
            rows = context_num
            cols = 3
            instruction = f"A grid layout with {rows} rows and {cols} columns, displaying {cols*rows} images arranged side by side. Each line illustrates a multi-view portrait of a person."
            print(instruction)
        elif "Yangguang" in data_item[0]["target"]:
            image_list = [[] for _ in range(context_num)]
            for i in range(context_num):
                for url in data_item[i]["condition"]:
                    cond_image = Image.open(read_general(url + ".png")).convert('RGB')
                    image_list[i].append(cond_image)
                target_image = Image.open(read_general(data_item[i]["target"] + ".png")).convert('RGB')
                image_list[i].append(target_image)
                image_list[i] = [resize_with_aspect_ratio(image, self.resolution, aspect_ratio=1.0) for image in image_list[i]]
            rows = context_num
            cols = 3
            instruction = f"A grid layout with {rows} rows and {cols} columns, displaying {cols*rows} images arranged side by side. Each line displays a multi-view illustration of a 3D object."
            print(instruction)
        elif "iteractive_editing" in data_item[0]["target"]:
            image_list = [[] for _ in range(context_num)]
            for i in range(context_num):
                for url in data_item[i]["condition"]:
                    cond_image = Image.open(read_general(url)).convert('RGB')
                    image_list[i].append(cond_image)
                target_image = Image.open(read_general(data_item[i]["target"])).convert('RGB')
                image_list[i].append(target_image)
                image_list[i] = [resize_with_aspect_ratio(image, self.resolution, aspect_ratio=1.0) for image in image_list[i]]
            rows = context_num
            cols = 3
            instruction = f"A grid layout with {rows} rows and {cols} columns, displaying {cols*rows} images arranged side by side. Each row outlines a logical process, starting from an image featuring the primary object, a source image, the source image with bounding box, to achieve a high-quality image."
            print(instruction)
        elif "iclight" in data_item[0]["target"]:
            image_list = [[] for _ in range(context_num)]
            for i in range(context_num):
                for url in [data_item[i]["condition"][-1], data_item[i]["condition"][1]]:
                    cond_image = Image.open(read_general(url)).convert('RGB')
                    image_list[i].append(cond_image)
                target_image = Image.open(read_general(data_item[i]["target"])).convert('RGB')
                image_list[i].append(target_image)
                image_list[i] = [resize_with_aspect_ratio(image, self.resolution) for image in image_list[i]]
            rows = context_num
            cols = 3
            instruction = f"A grid layout with {rows} rows and {cols} columns, displaying {cols*rows} images arranged side by side. Each line displays a image re-lighting process."
            print(instruction)

        # 对每个图片应用transform并拼接
        processed_images = []
        for images in image_list:
            transformed_row = []
            for img in images:
                try:
                    transformed = self.image_transform(img)
                except Exception as e:
                    print("error image_type: ", image_type_list)
                transformed_row.append(transformed)
            row = torch.cat(transformed_row, dim=2)  # 在宽度维度上拼接
            processed_images.append(row)
        image = torch.cat(processed_images, dim=1)  # 在高度维度上拼接

        return image, instruction, text_emb

    # def object200k_process_item(self, data_items):
    #     condition_list = [condition for condition in condition_list if condition != "depth_anything_v2"]
    #     random.shuffle(condition_list)
    #     condition_list = condition_list[:2]
    #     cond_images_1 = []
    #     cond_images_2 = []
    #     target_images = []
    #     ref_images = []
    #     texts = []
    #     for data_item in data_items:
    #         # target image
    #         target_image_path = f"/mnt/hwfile/alpha_vl/duruoyi/subjects200k/images_all/{data_item['image_name']}.jpg"
    #         image = Image.open(target_image_path).convert('RGB')
    #         width, height = image.size
    #         half_width = width // 2
    #         target_image = image.crop((0, 0, half_width, height))
    #         ref_image = image.crop((half_width, 0, width, height))
    #         padding = 8
    #         target_image = target_image.crop((padding, padding, half_width-padding, height-padding))
    #         # ref image
    #         ref_image = ref_image.crop((padding, padding, half_width-padding, height-padding))
    #         # condition image 1
    #         try:
    #             cond_image_1 = data_item["condition"][condition_list[0]]
    #             cond_image_1 = Image.open(read_general(cond_image_1)).convert("RGB")
    #         except Exception as e:
    #             cond_image_1 = Image.new('RGB', (half_width-padding*2, height-padding*2), (0, 0, 0))
    #         # condition image 2
    #         try:
    #             cond_image_2 = data_item["condition"][condition_list[1]]
    #             cond_image_2 = Image.open(read_general(cond_image_2)).convert("RGB")
    #         except Exception as e:
    #             cond_image_2 = Image.new('RGB', (half_width-padding*2, height-padding*2), (0, 0, 0))
    #         text = data_item['description']['description_0']
    #         cond_images_1.append(cond_image_1)
    #         cond_images_2.append(cond_image_2)
    #         target_images.append(target_image)
    #         ref_images.append(ref_image)
    #         texts.append(text) 

    #     return cond_images_1, cond_images_2, ref_images, target_images, texts

    def output_pair_data(self, input_image, target_image, text, text_emb):
        # Resize input image keeping aspect ratio
        input_image = resize_with_aspect_ratio(input_image, self.resolution)
        # Get input image dimensions for target image processing
        input_w, input_h = input_image.size
        # Center crop target image to match input aspect ratio
        target_w, target_h = target_image.size
        target_aspect = target_w / target_h
        input_aspect = input_w / input_h
        if target_aspect > input_aspect:
            # Target is wider - crop width
            new_target_w = int(target_h * input_aspect)
            left = (target_w - new_target_w) // 2
            target_image = target_image.crop((left, 0, left + new_target_w, target_h))
        else:
            # Target is taller - crop height  
            new_target_h = int(target_w / input_aspect)
            top = (target_h - new_target_h) // 2
            target_image = target_image.crop((0, top, target_w, top + new_target_h))
        # Resize target to match input dimensions
        target_image = target_image.resize((input_w, input_h), Image.LANCZOS)
        # Apply remaining transforms
        input_image = self.image_transform(input_image)
        target_image = self.image_transform(target_image)
        image = torch.cat([input_image, target_image], dim=2)

        return image, text, text_emb

    def output_grid_data(self, cond_images_1, cond_images_2, ref_images, target_images, texts, text_emb):
        context_num_prob = [0.3, 0.4, 0.3]  # 概率分别对应context_num=1,2,3
        context_num = random.choices([1, 2, 3], weights=context_num_prob)[0]
        cond_images_1, cond_images_2, ref_images, target_images, texts = cond_images_1[-context_num:], cond_images_2[-context_num:], ref_images[-context_num:], target_images[-context_num:], texts[-context_num:]

        image_list = [[] for _ in range(context_num)]
        with_condition = False
        while not with_condition:
            if random.random() < 0.7:
                with_condition = True
                for i in range(context_num):
                    image_list[i].append(cond_images_1[i])
            if random.random() < 0.2:
                with_condition = True
                for i in range(context_num):
                    image_list[i].append(cond_images_2[i])
            if random.random() < 0.7:
                with_condition = True
                for i in range(context_num):
                    image_list[i].append(ref_images[i])

        # 对每个context的图像列表进行相同顺序的打乱
        if len(image_list[0]) > 1:  # 只有当有多个图像时才需要打乱
            # 生成一个打乱的索引序列
            indices = list(range(len(image_list[0])))
            random.shuffle(indices)
            # 按照相同的索引序列重排每个context的图像列表
            for i in range(context_num):
                image_list[i] = [image_list[i][j] for j in indices]
        
        for i in range(context_num):
            image_list[i].append(target_images[i])
        
        # 对每个图片应用transform并拼接
        processed_images = []
        for images in image_list:
            # 处理每一行的图片
            transformed_row = []
            for img in images:
                transformed = self.image_transform(img)
                transformed_row.append(transformed)
            # 水平拼接同一行的图片
            row = torch.cat(transformed_row, dim=2)  # 在宽度维度上拼接
            processed_images.append(row)
        # 垂直拼接所有行
        image = torch.cat(processed_images, dim=1)  # 在高度维度上拼接

        layout_instruction = f"Demonstration of the conditional image generation process, {context_num*len(images)} images form a grid with {context_num} rows and {len(images)} columns, side by side. "
        content_instruction = f"The content of the last image is: " + texts[-1]
        if random.random() < 0.9:
            instruction = layout_instruction + content_instruction
        else:
            instruction = layout_instruction
        return image, instruction, text_emb
        

    def process_item(self, data_item, training_mode=False, image_type_list=None, context_num=1):
        text_emb = None
        
        if "input_path" in data_item:
            input_image, target_image, text = self.pixwizard_process_item(data_item)
            return self.output_pair_data(input_image, target_image, text, text_emb)
        elif "condition" in data_item:
            input_image, target_image, text = self.lige_process_item(data_item)
            return self.output_pair_data(input_image, target_image, text, text_emb)
        elif "bg_caption" in data_item:
            input_image, target_image, text = self.chunlian_ocr_process_item(data_item)
            return self.output_pair_data(input_image, target_image, text, text_emb)
        elif "caption_en" in data_item:
            input_image, target_image, text = self.jimeng_ocr_process_item(data_item)
            return self.output_pair_data(input_image, target_image, text, text_emb)
        elif isinstance(data_item, list) and "description" in data_item[0]:
            return self.object200k_process_item(data_item, image_type_list, context_num)
            # cond_images_1, cond_images_2, ref_images, target_images, texts = self.object200k_process_item(data_item)
            # return self.output_grid_data(cond_images_1, cond_images_2, ref_images, target_images, texts, text_emb)
        else:
            raise ValueError(f"Unknown data item: {data_item}")

def sample_random_mask(h, w, data_source="2x1_grid", random_mask=False):
    w_grid, h_grid = data_source.split("_")[0].split("x")
    w_grid, h_grid = int(w_grid), int(h_grid)
    # w_grid=3
    print(w, h, w_grid, h_grid)
    w_stride, h_stride = w // w_grid, h // h_grid
    mask = torch.zeros([1, 1, h, w])
    if random.random() < 0.5 and random_mask:
        w_idx = random.randint(0, w_grid - 1)
        h_idx = random.randint(0, h_grid - 1)
        mask[:, :, h_idx * h_stride: (h_idx + 1) * h_stride, w_idx * w_stride: (w_idx + 1) * w_stride] = 1
    else:
        mask[:, :, h - h_stride: h, w - w_stride: w] = 1
        # mask[:, :, h - h_stride: h, :w - w_stride] = 1
        # mask[:, :, h - h_stride: h, w - w_stride * 2: w - w_stride] = 1
        # mask[:, :, :h_stride, :w_stride] = 1
    # print(mask.sum() / w / h)
    return mask

def none_or_str(value):
    if value == "None":
        return None
    return value


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device_str = f"cuda:{rank}"

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    print("Init model")
    model = load_flow_model(args.model_name, device=device_str, lora_rank=args.lora_rank)
    # params = configs[args.model_name].params
    # with torch.device(device_str):
    #     if "lora" in args.model_name:
    #         model = FluxLoraWrapper(params).to(dtype)
    #     else:
    #         model = Flux(params).to(dtype)
    # for name, param in model.named_parameters():
    #     print(name)

    print("Init vae")
    ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device_str)
    ae.requires_grad_(False)
    
    print("Init text encoder")
    t5 = load_t5(device_str, max_length=args.max_length)
    clip = load_clip(device_str)
        
    model.eval().to(device_str, dtype=dtype)

    # data processor
    image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    item_processor = T2IItemProcessor(image_transform, resolution=args.grid_resolution)

    if not args.debug:
        # assert train_args.model_parallel_size == args.num_gpus
        if (args.ckpt).endswith(".safetensors"):
            ckpt = load_sft(args.ckpt, device=device_str)
            missing, unexpected = model.load_state_dict(ckpt, strict=False, assign=True)
        else:
            if args.ema:
                print("Loading ema model.")
            ckpt = torch.load(
                os.path.join(
                    args.ckpt,
                    f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.pth",
                )
            )
            model.load_state_dict(ckpt, strict=True)
        del ckpt
        
    # begin sampler
    transport = create_transport(
        "Linear",
        "velocity",
        do_shift=args.do_shift,
    ) 
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=args.solver,
        num_steps=args.num_sampling_steps,
        atol=args.atol,
        rtol=args.rtol,
        reverse=args.reverse,
        do_shift=args.do_shift,
        time_shifting_factor=args.time_shifting_factor,
    )
    # end sampler

    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, "images"), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, "data.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.loads(f.read())
        collected_id = []
        for i in info:
            collected_id.append(f'{i["idx"]}_{i["high_res"]}')
    else:
        info = []
        collected_id = []

    # 检查文件扩展名并相应地读取数据
    data = []
    with open(args.caption_path, "r", encoding="utf-8") as file:
        if args.caption_path.endswith('.jsonl'):
            # 读取 jsonl 文件
            for line in file:
                data.append(json.loads(line))
        else:
            # 读取 json 文件
            data = json.load(file)
    
    total = len(info)
    with torch.autocast("cuda", dtype):
        for idx, item in tqdm(enumerate(data[::1])):
            if idx > 10:
                break
            for context_num in [1, 2, 3]:   
                for task_type in test_task_dicts:
                    for image_type_list in task_type["image_list"]:
                    # for image_type_list in ["ood"]:
                        task_name = "_".join(image_type_list) 
                        ref = True
                        # 使用循环索引确保能获取足够的上下文数据
                        context_items = []
                        for i in range(context_num - 1):
                            next_idx = (idx + 1 + i * 1) % len(data)  # 使用取模运算实现循环索引
                            context_items.append(data[next_idx])
                        all_items = context_items + [item]
                        # data processing
                        target_image, text, _ = item_processor.object200k_process_item(all_items, image_type_list, context_num)
                        # target_image, text, _ = item_processor.ood_process_item(all_items, image_type_list, context_num)
                        # target_image, text, _ = item_processor.process_item(all_items, context_num, cond1, cond2, ref)
                        caps_list = [text]
                        x = [target_image]

                        task_types = [f"{img.shape[2]//args.grid_resolution}x{img.shape[1]//args.grid_resolution}_grid" for img in x]
                        if args.total_resolution > 0:
                            x = [resize_with_aspect_ratio(img, int(args.total_resolution)) for img in x]  

                        fill_masks = []
                        fill_conds = []
                        for img, task_type in zip(x, task_types):
                            h, w = img.shape[-2:]
                            fill_masks.append(sample_random_mask(h, w, task_type).to(img.device))
                            fill_conds.append(img * (1 - fill_masks[-1][0]))
                        fill_conds = [img.to(device_str, non_blocking=True) for img in fill_conds]
                        fill_masks = [mask.to(device_str, non_blocking=True) for mask in fill_masks]
                        with torch.no_grad():
                            # fill_conds = [ae.encode(img[None].to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor for img in fill_conds]
                            fill_conds = [(ae.encode(img[None].to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in fill_conds]
                            fill_masks = [mask.to(torch.bfloat16) for mask in fill_masks]
                            fill_masks = [rearrange(mask, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=8, pw=8) for mask in fill_masks]
                            fill_masks = [rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for mask in fill_masks]
                        fill_conds = [cond.to(torch.bfloat16) for cond in fill_conds]
                        fill_conds = [rearrange(cond.unsqueeze(0), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for cond in fill_conds]
                        fill_conds = torch.cat(fill_conds, dim=0)
                        fill_masks = torch.cat(fill_masks, dim=0)
                        img_cond = torch.cat((fill_conds, fill_masks), dim=-1)
                        
                        for high_res in args.resolution:
                                            
                            if int(args.seed) != 0:
                                torch.random.manual_seed(int(args.seed))

                            sample_id = f'{idx}_{high_res}'
                            if sample_id in collected_id:
                                continue

                            do_extrapolation = int(high_res) > 1024

                            n = len(caps_list)
                            # w, h = resolution.split("x")
                            # h, w = int(h), int(w)
                            latent_w, latent_h = w // 8, h // 8
                            x = torch.randn([1, 16, latent_h, latent_w], device=device_str).to(dtype)
                            with torch.no_grad():
                                if args.do_classifier_free_guidance:
                                    x = x.repeat(n * 2, 1, 1, 1)
                                    inp = prepare_modified(t5=t5, clip=clip, img=x, prompt=[caps_list] + [""] * n, proportion_empty_prompts=0.0)
                                else:
                                    inp = prepare_modified(t5=t5, clip=clip, img=x, prompt=caps_list, proportion_empty_prompts=0.0)

                            if args.do_classifier_free_guidance:
                                model_kwargs = dict(
                                    txt=inp["txt"], 
                                    txt_ids=inp["txt_ids"], 
                                    txt_mask=inp["txt_mask"],
                                    y=inp["vec"], 
                                    img_ids=inp["img_ids"], 
                                    img_mask=inp["img_mask"], 
                                    cond=img_cond,
                                    guidance=torch.full((x.shape[0],), 1, device=x.device, dtype=x.dtype), 
                                    cfg_scale=args.guidance_scale,
                                )
                                samples = sample_fn(
                                    inp["img"], model.forward_with_cfg, model_kwargs
                                )[-1]
                            else:
                                model_kwargs = dict(
                                    txt=inp["txt"], 
                                    txt_ids=inp["txt_ids"], 
                                    txt_mask=inp["txt_mask"],
                                    y=inp["vec"], 
                                    img_ids=inp["img_ids"], 
                                    img_mask=inp["img_mask"], 
                                    cond=img_cond,
                                    guidance=torch.full((x.shape[0],), args.guidance_scale, device=x.device, dtype=x.dtype), 
                                )
                                samples = sample_fn(
                                    inp["img"], model.forward, model_kwargs
                                )[-1]
                            samples = samples[:n]
                            samples = rearrange(samples, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=latent_h//2, w=latent_w//2)
                            samples = ae.decode(samples / ae.config.scaling_factor + ae.config.shift_factor)[0]
                            samples = (samples + 1.0) / 2.0
                            samples.clamp_(0.0, 1.0)

                            # Save samples to disk as individual .png files
                            for i, (sample, cap) in enumerate(zip(samples, caps_list)):
                                
                                img = to_pil_image(sample.float())
                                save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}_{task_types[i]}_{task_name}.jpg"
                                img.save(save_path, format='JPEG', quality=95)
                                
                                info.append(
                                    {
                                        "idx": idx,
                                        "caption": cap,
                                        "image_url": f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}_{task_types[i]}_{task_name}.jpg",
                                        "high_res": high_res,
                                        "solver": args.solver,
                                        "num_sampling_steps": args.num_sampling_steps,
                                    }
                                )

                            with open(info_path, "w") as f:
                                f.write(json.dumps(info))

                            total += len(samples)
                            dist.barrier()

    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_encoder", type=str, nargs='+', default=['t5', 'clip'], help="List of text encoders to use (e.g., t5, clip, gemma)")
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    # parser.set_defaults(ema=True)
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
        "(sample{_ema}.png in the model checkpoint directory).",
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="prompts.txt",
    )
    parser.add_argument(
        "--low_res_list",
        type=str,
        default="256,512,1024",
        help="Comma-separated list of low resolution for sampling."
    )
    parser.add_argument(
        "--high_res_list",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated list of high resolution for sampling."
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="",
        nargs="+",
    )
    parser.add_argument(
        "--grid_resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--total_resolution",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
    )
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="Time-aware",
    )
    parser.add_argument(
        "--scaling_watershed",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    parser.add_argument("--do_shift", default=True)
    parser.add_argument("--attn_token_select", action="store_true")
    parser.add_argument("--mlp_token_select", action="store_true")
    parser.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    parser.add_argument("--use_flash_attn", type=bool, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512, help="Max length for T5.")
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="flux-dev")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--do_classifier_free_guidance", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    args = parser.parse_known_args()[0]
    
    args.low_res_list = [int(res) for res in args.low_res_list.split(",")]
    args.high_res_list = [int(res) for res in args.high_res_list.split(",")]
    
    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)