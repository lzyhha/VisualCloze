# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict, defaultdict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time
import warnings
from safetensors import safe_open
import wandb
import cv2
import numpy as np

from PIL import Image
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from diffusers import AutoencoderKL
from einops import rearrange, repeat

from data import ItemProcessor, MyDataset
from flux.sampling import prepare_modified, prepare_fill_modified
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport
from util.misc import SmoothedValue

from data_reader import read_general
from data.prefix_instruction import get_layout_instruction, get_task_instruction, get_content_instruction, get_image_prompt, task_dicts, condition_list, degradation_list, style_list, editing_list
from degradation_utils import add_degradation

#############################################################################
#                            Data item Processor                            #
#############################################################################



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
        block_list = []
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
            instruction = f"A grid layout with {rows} rows and {cols} columns, displaying {cols*rows} images arranged side by side. Each line illustrates a painting process, from rough lines to detailed drawings."
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


#############################################################################
#                           Training Helper Functions                       #
#############################################################################


def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    prompt = [x[1] for x in samples]
    text_emb = [x[2] for x in samples]
    return image, prompt, text_emb


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.95):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module, auto_wrap_policy) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


def parameter_count(model):
    """计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (total_params, trainable_params): 总参数量和可训练参数量的元组
    """
    unique_params = {p for n, p in model.named_parameters()}
    
    total_params = sum(p.numel() for p in unique_params)
    trainable_params = sum(p.numel() for p in unique_params if p.requires_grad)
    
    return total_params, trainable_params

def sample_random_mask(h, w, data_source="2x1_grid"):
    w_grid, h_grid = data_source.split("_")[0].split("x")
    w_grid, h_grid = int(w_grid), int(h_grid)
    w_stride, h_stride = w // w_grid, h // h_grid
    mask = torch.zeros([1, 1, h, w])
    if random.random() < 0.5:
        w_idx = random.randint(0, w_grid - 1)
        h_idx = random.randint(0, h_grid - 1)
        mask[:, :, h_idx * h_stride: (h_idx + 1) * h_stride, w_idx * w_stride: (w_idx + 1) * w_stride] = 1
    else:
        mask[:, :, h - h_stride: h, w - w_stride: w] = 1
    return mask

#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    device_str = f"cuda:{device}"
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        tb_logger = SummaryWriter(
            os.path.join(
                args.results_dir, "tensorboard", datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname()
            )
        )
        # Create wandb logger
        if args.use_wandb:
            wandb.init(
                project="FLUX",
                name=args.results_dir.split("/")[-1],
                config=args.__dict__,  # Use args.__dict__ to pass all arguments
                dir=args.results_dir,  # Set the directory for wandb files
                job_type="training",
                reinit=True,  # Allows multiple runs in the same process
            )
    else:
        logger = create_logger(None)
        tb_logger = None

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    if args.load_t5:
        t5 = load_t5(max_length=512)
        t5 = setup_lm_fsdp_sync(
            t5,
            functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in list(t5.hf_module.encoder.block),
            ),
        )
        logger.info("T5 loaded")
    else:
        t5 = None

    if args.load_clip:
        clip = load_clip()
        clip = setup_lm_fsdp_sync(
            clip,
            functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in list(clip.hf_module.text_model.encoder.layers),
            ),
        )
        logger.info(f"CLIP loaded")
    else:
        clip = None

    model = load_flow_model(args.model_name, device=device_str, lora_rank=args.lora_rank)
    # for block in model.double_blocks:
        # block.init_cond_weights()
    
    ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    ae.requires_grad_(False)

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    if args.use_model_ema:
        model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )

            logger.info(f"Resuming ema weights from: {args.resume}")
            if args.use_model_ema:
                model_ema.load_state_dict(
                    torch.load(
                        os.path.join(
                            args.resume,
                            f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
                        ),
                        map_location="cpu",
                    ),
                    strict=True,
                )

    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(
                os.path.join(
                    args.init_from,
                    f"consolidated.{0:02d}-of-{1:02d}.pth",
                ),
                map_location="cpu",
            )

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")
    dist.barrier()

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.get_checkpointing_wrap_module_list())
        # checkpointing_list = [param for name, param in model.named_parameters() if "lora" in name]
        if args.use_model_ema:
            checkpointing_list_ema = list(model_ema.get_checkpointing_wrap_module_list())
        else:
            checkpointing_list_ema = []
    else:
        checkpointing_list = []
        checkpointing_list_ema = []

    model = setup_fsdp_sync(model, args)
    if args.use_model_ema:
        model_ema = setup_fsdp_sync(model_ema, args)
    
    # checkpointing (part2, after FSDP wrapping)
    if args.checkpointing:
        print("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        if args.use_model_ema:
            apply_activation_checkpointing(
                model_ema,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: submodule in checkpointing_list_ema,
            )

    logger.info(f"model:\n{model}\n")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    # learning rate of 1e-4 in our paper):
    model_params = []
    for name, param in model.named_parameters():
        # print(name)
        if args.training_type == "full_model":
            param.requires_grad = True
            model_params.append(param)
        # elif "cond" in name or 'norm' in name or 'bias' in name:
        elif args.training_type == "double_block" and 'double_blocks' in name:
            param.requires_grad = True
            model_params.append(param)
        elif args.training_type == "bias" and 'bias' in name:
            param.requires_grad = True
            model_params.append(param)
        elif args.training_type == "norm" and 'norm' in name:
            param.requires_grad = True
            model_params.append(param)
        elif args.training_type == "lora" and 'lora' in name:
            param.requires_grad = True
            model_params.append(param)
        else:
            param.requires_grad = False
    print("Trainable params:")
    print(model_params)
    total_params, trainable_params = parameter_count(model)
    size_in_gb = total_params * 4 / 1e9
    logger.info(f"Model Size: {size_in_gb:.2f} GB, Total Parameters: {total_params / 1e9:.2f} B, Trainable Parameters: {trainable_params / 1e9:.2f} B")
    if len(model_params) > 0:
        opt = torch.optim.AdamW(model_params, lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr  # todo learning rate and weight decay
            param_group["weight_decay"] = args.wd  # todo learning rate and weight decay

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0
    
    # default: 1000 steps, linear noise schedule
    transport = create_transport(
        "Linear",
        "velocity",
        None,
        None,
        None,
        snr_type=args.snr_type,
        do_shift=args.do_shift,
    )  # default: velocity;

    # Setup data:
    logger.info(f"Creating data")
    data_collection = {}
    global_bsz = args.global_bsz
    local_bsz = global_bsz // dp_world_size  # todo caution for sequence parallel
    micro_bsz = args.micro_bsz
    num_samples = global_bsz * args.max_steps
    assert global_bsz % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
    logger.info(f"Global bsz: {global_bsz} Local bsz: {local_bsz} Micro bsz: {micro_bsz}")
    for data_source in ['2x1_grid']:
        image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        dataset = MyDataset(
            args.data_path,
            train_res=None,
            item_processor=T2IItemProcessor(image_transform, resolution=args.grid_resolution),
            cache_on_disk=args.cache_data_on_disk,
        )
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        logger.info(f"Total # samples to consume: {num_samples:,} " f"({num_samples / len(dataset):.2f} epochs)")
        sampler = get_train_sampler(
            dataset,
            dp_rank,
            dp_world_size,
            global_bsz,
            args.max_steps,
            resume_step,
            args.global_seed
        )
        loader = DataLoader(
            dataset,
            batch_size=local_bsz,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataloader_collate_fn,
        )

        data_collection[data_source] = {
            "loader": loader,
            "loader_iter": iter(loader),
            "global_bsz": global_bsz,
            "local_bsz": local_bsz,
            "micro_bsz": micro_bsz,
            "metrics": defaultdict(lambda: SmoothedValue(args.log_every)),
            "transport": transport,
        }

    # Prepare models for training:
    model.train()

    # Variables for monitoring/logging purposes:
    logger.info(f"Training for {args.max_steps:,} steps...")

    start_time = time()
    for step in range(resume_step, args.max_steps):
        data_source = random.choices(['2x1_grid'], weights=[1.0])[0]
        # torch.distributed.broadcast(data_source, src=0)
        data_pack = data_collection[data_source]
        
        x, caps, text_emb = next(data_pack["loader_iter"])
        x = [img.to(device, non_blocking=True) for img in x]
        bsz = len(x)

        task_types = [f"{img.shape[2]//args.grid_resolution}x{img.shape[1]//args.grid_resolution}_grid" for img in x]
        # print(1, [f"{img.shape[2]}x{img.shape[1]}" for img in x])
        # print(task_types)
        if args.total_resolution > 0:
            x = [resize_with_aspect_ratio(img, args.total_resolution) for img in x]
        # print(2, [f"{img.shape[2]}x{img.shape[1]}" for img in x])

        if "fill" in args.model_name:
            fill_masks = []
            fill_conds = []
            for img, task_type in zip(x, task_types):
                h, w = img.shape[-2:]
                fill_masks.append(sample_random_mask(h, w, task_type).to(img.device))
                fill_conds.append(img * (1 - fill_masks[-1][0]))
            fill_conds = [img.to(device, non_blocking=True) for img in fill_conds]
            fill_masks = [mask.to(device, non_blocking=True) for mask in fill_masks]
            with torch.no_grad():
                fill_conds = [(ae.encode(img[None].to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in fill_conds]
                fill_masks = [mask.to(torch.bfloat16) for mask in fill_masks]
                loss_masks = [F.interpolate(1 - mask, size=(mask.shape[2]//16, mask.shape[3]//16), mode='nearest') for mask in fill_masks]
                loss_masks = [rearrange(mask, "b c h w -> b (h w) c") for mask in loss_masks]
                fill_masks = [rearrange(mask, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=8, pw=8) for mask in fill_masks]
                fill_masks = [rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for mask in fill_masks]
            fill_conds = [cond.to(torch.bfloat16) for cond in fill_conds]
            fill_conds = [rearrange(cond.unsqueeze(0), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for cond in fill_conds]

            max_len = max([cond.shape[1] for cond in fill_conds])
            # print(max([cond.shape[1] for cond in fill_conds]), [mask.shape[1] for mask in fill_masks])
            fill_conds = [F.pad(cond, (0, 0, 0, max_len - cond.shape[1])) for cond in fill_conds]
            fill_masks = [F.pad(mask, (0, 0, 0, max_len - mask.shape[1])) for mask in fill_masks]
            loss_masks = [F.pad(mask, (0, 0, 0, max_len - mask.shape[1])) for mask in loss_masks]

            fill_conds = torch.cat(fill_conds, dim=0)
            fill_masks = torch.cat(fill_masks, dim=0)
            loss_masks = torch.cat(loss_masks, dim=0)
            img_cond = torch.cat((fill_conds, fill_masks), dim=-1)
            # print(img_cond.shape)

        dataload_time = time()
        
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            # x = [(ae.tiled_encode(img[None].to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in x]
            x = [(ae.encode(img[None].to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in x]
            
        with torch.no_grad():
            inp = prepare_modified(t5=t5, clip=clip, img=x, prompt=caps, proportion_empty_prompts=args.caption_dropout_prob)

        # Prepare text embedding if needed:
        with torch.no_grad():
            vec_uncond = clip([""] * bsz)
            txt_uncond = t5([""] * bsz)
            txt_uncond_ids = torch.zeros(bsz, txt_uncond.shape[1], 3, device=txt_uncond.device)
            txt_uncond_mask = torch.ones(bsz, txt_uncond.shape[1], device=txt_uncond.device, dtype=torch.int32)

        encode_time = time()

        loss_item = 0.0
        diff_loss_item = 0.0
        opt.zero_grad()

        if args.masking_loss:
            inp["img_mask"] = inp["img_mask"] * loss_masks.squeeze(-1)
        
        for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
            mb_st = mb_idx * data_pack["micro_bsz"]
            mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
            last_mb = mb_ed == data_pack["local_bsz"]

            x_mb = inp["img"][mb_st:mb_ed]
            
            model_kwargs = dict(
                img_ids=inp["img_ids"][mb_st:mb_ed],
                txt=inp["txt"][mb_st:mb_ed],
                txt_ids=inp["txt_ids"][mb_st:mb_ed],
                y=inp["vec"][mb_st:mb_ed],
                guidance=torch.full((x_mb.shape[0],), 1.0, device=x_mb.device, dtype=x_mb.dtype),
                img_mask=inp["img_mask"][mb_st:mb_ed],
                txt_mask=inp["txt_mask"][mb_st:mb_ed],
            )
            extra_kwargs = {}
            if "fill" in args.model_name:
                extra_kwargs["cond"] = img_cond[mb_st:mb_ed]
            else:
                extra_kwargs["cond"] = None

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                loss_dict = data_pack["transport"].training_losses(model, x_mb, model_kwargs, extra_kwargs)
            loss = loss_dict["loss"].sum() / data_pack["local_bsz"]
            diff_loss = loss_dict["task_loss"].sum() / data_pack["local_bsz"]
            loss_item += loss.item()
            diff_loss_item += diff_loss.item()
            with model.no_sync() if args.data_parallel in ["sdp"] and not last_mb else contextlib.nullcontext():
                loss.backward()

        grad_norm = model.clip_grad_norm_(max_norm=args.grad_clip)

        if tb_logger is not None:
            tb_logger.add_scalar(f"train/loss", loss_item, step)
            tb_logger.add_scalar(f"train/grad_norm", grad_norm.float(), step)
            tb_logger.add_scalar(f"train/lr", opt.param_groups[0]["lr"], step)
                
        if args.use_wandb and rank == 0:
            wandb.log({
                "train/loss": loss_item,
                "train/grad_norm": grad_norm,
                "train/lr": opt.param_groups[0]["lr"],
            }, step=step)

        opt.step()
        end_time = time()

        # Log loss values:
        metrics = data_pack["metrics"]
        metrics["loss"].update(loss_item)
        metrics["diff_loss"].update(diff_loss_item)
        metrics["grad_norm"].update(grad_norm)
        metrics["DataloadSecs/Step"].update(dataload_time - start_time)
        metrics["EncodeSecs/Step"].update(encode_time - dataload_time)
        metrics["TrainSecs/Step"].update(end_time - encode_time)
        metrics["Secs/Step"].update(end_time - start_time)
        metrics["Imgs/Sec"].update(data_pack["global_bsz"] / (end_time - start_time))
        metrics["grad_norm"].update(grad_norm)
        if (step + 1) % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            logger.info(
                f"Task_{data_source}: (step{step + 1:07d}) "
                + f"lr{opt.param_groups[0]['lr']:.6f} "
                + " ".join([f"{key}:{str(val)}" for key, val in metrics.items()])
            )

        start_time = time()

        if args.use_model_ema:
            update_ema(model_ema, model)

        # Save DiT checkpoint:
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            if args.use_model_ema:
                with FSDP.state_dict_type(
                    model_ema,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_ema_state_dict = model_ema.state_dict()
                    if fs_init.get_data_parallel_rank() == 0:
                        consolidated_ema_fn = (
                            "consolidated_ema."
                            f"{fs_init.get_model_parallel_rank():02d}-of-"
                            f"{fs_init.get_model_parallel_world_size():02d}"
                            ".pth"
                        )
                        torch.save(
                            consolidated_ema_state_dict,
                            os.path.join(checkpoint_path, consolidated_ema_fn),
                        )
                dist.barrier()
                del consolidated_ema_state_dict
                logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    # parser.add_argument("--global_bsz_256", type=int, default=256)
    # parser.add_argument("--micro_bsz_256", type=int, default=1)
    # parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--global_bsz", type=int, default=256)
    parser.add_argument("--micro_bsz", type=int, default=1)
    # parser.add_argument("--global_bsz_1024", type=int, default=256)
    # parser.add_argument("--micro_bsz_1024", type=int, default=1)
    parser.add_argument("--load_t5", action="store_true")
    parser.add_argument("--load_clip", action="store_true")
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0, help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument("--do_shift", default=False)
    parser.add_argument(
        "--no_shift",
        action="store_false",
        dest="do_shift",
        help="Do dynamic time shifting",
    )
    parser.add_argument(
        "--task_list",
        type=str,
        default="2x1_grid",
        help="Comma-separated list of task for training."
    )
    parser.add_argument(
        "--task_probs",
        type=str,
        default="1.0",
        help="Comma-separated list of probabilities for sampling tasks."
    )
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--total_resolution", type=int, default=1024)
    parser.add_argument("--grid_resolution", type=int, default=512)
    parser.add_argument("--masking_loss", action="store_true")
    parser.add_argument("--full_model", action="store_true")
    parser.add_argument("--training_type", type=str, default="lora") # ["lora", "full_model", "double_block", "bias", "norm"]
    parser.add_argument("--use_model_ema", action="store_true")
    parser.add_argument("--use_model_watermark", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.task_list = args.task_list.split(",")
    args.task_probs = [float(prob) for prob in args.task_probs.split(",")]
    main(args)
