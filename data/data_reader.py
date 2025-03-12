import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import random

from data.prefix_instruction import get_layout_instruction, get_task_instruction, get_content_instruction, get_image_prompt, condition_list, \
    degradation_list, style_list, editing_list
from data.degradation_utils import add_degradation
from data.dataset import ItemProcessor


def resize_with_aspect_ratio(img, resolution, divisible=16, aspect_ratio=None):
    """Resize the image while maintaining the aspect ratio, 
    so that the area is close to resolution**2 and the width and height are divisible by 16.

    Args:
        img: PIL Image or torch.Tensor (C,H,W)/(B,C,H,W)
        resolution: Target resolution
        divisible: Ensures that the output dimensions are divisible by this number

    Returns:
        The resized image, with the same type as the input
    """
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
        
    if aspect_ratio is None:
        aspect_ratio = w / h
    target_area = resolution * resolution
    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)
    
    new_w = max(new_w // divisible, 1) * divisible
    new_h = max(new_h // divisible, 1) * divisible
    
    if is_tensor:
        mode = 'bilinear'
        align_corners = False
        if batch_dim:
            return F.interpolate(img, size=(new_h, new_w), 
                               mode=mode, align_corners=align_corners)
        else:
            return F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                               mode=mode, align_corners=align_corners).squeeze(0)
    else:
        return img.resize((new_w, new_h), Image.LANCZOS)


class T2IItemProcessor(ItemProcessor):

    def __init__(self, transform, resolution=512):
        self.image_transform = transform
        self.resolution = resolution

    def get_image_object200k(self, data_item, image_type):
        if image_type in ["target", "reference"]:
            image = Image.open(data_item["condition"][image_type]).convert('RGB')
            return [image]
        elif image_type == "foreground" or image_type == "background":
            target_image = Image.open(data_item["condition"]["target"]).convert('RGB')
            
            mask = Image.open(data_item["condition"]["foreground"]).convert("L")
            mask_np = np.array(mask).astype(np.float32) / 255.0
            mask_np = (mask_np > 0.5).astype(np.int32)
            if "foreground" in image_type:
                mask_np = mask_np[..., None]
            else:
                mask_np = 1 - mask_np
                mask_np = mask_np[..., None]

            result = Image.fromarray((np.array(target_image) * mask_np).astype(np.uint8))
            return [result]
        elif image_type in style_list:
            if image_type == "InstantStyle":
                source_dict = data_item["condition"]["InstantStyle"]
            elif image_type == "ReduxStyle":
                source_dict = data_item["condition"]["ReduxStyle"]
            style_idx = random.randint(0, len(source_dict["style_path"]) - 1)
            style_image = Image.open(source_dict["style_path"][style_idx]).convert("RGB")
            target_image = Image.open(source_dict["image_path"][style_idx]).convert("RGB")
            return [style_image, target_image]
        elif image_type in editing_list:
            if image_type == "DepthEdit":
                editing_image_path = data_item["condition"]["DepthEdit"]
            elif image_type == "FillEdit":
                editing_image_path = random.choice(data_item["condition"]["FillEdit"]["image_path"])
            editing_image = Image.open(editing_image_path).convert('RGB')
            return [editing_image]
        elif image_type in condition_list:
            cond_image = Image.open(data_item["condition"][image_type]).convert("RGB")
            return [cond_image]
        elif image_type in degradation_list:
            target_image = Image.open(data_item["condition"]["target"]).convert('RGB')
            deg_image, _ = add_degradation(np.array(target_image), image_type)
            return [deg_image]
        else:
            raise NotImplementedError()

    def graph200k_process_item(self, data_item, image_type_list=None, context_num=1, group_name=None, training_mode=True):

        image_list = [[] for _ in range(context_num)]
        for i in range(context_num):
            for image_type in image_type_list:
                images = self.get_image_object200k(data_item[i], image_type) 
                images = [resize_with_aspect_ratio(image, self.resolution, aspect_ratio=1.0) for image in images]
                image_list[i] += images
        image_prompt_list = []
        for image_type in image_type_list:
            image_prompt_list += get_image_prompt(image_type)

        # Shuffle n-1 elements
        if training_mode:
            indices = list(range(len(image_prompt_list)-1))
            random.shuffle(indices)
            for i in range(context_num):
                image_list[i][:len(image_prompt_list)-1] = [image_list[i][j] for j in indices]
            image_prompt_list[:len(image_prompt_list)-1] = [image_prompt_list[j] for j in indices]
        image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]

        if not training_mode:
            image = image_list
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            instruction = [
                get_layout_instruction(len(image_list[0]), context_num), 
                get_task_instruction(condition_prompt, target_prompt),
            ]
            if image_type_list[-1] == "target":
                instruction.append(get_content_instruction() + data_item[i]['description']['item'] + " " + data_item[i]['description']['description_0'])
            else:
                instruction.append("")
            return group_name, image, instruction, None, (len(image_list[0]), len(image_list))

        processed_images = []
        for images in image_list:
            transformed_row = []
            for img in images:
                transformed_row.append(self.image_transform(img))
            row = torch.cat(transformed_row, dim=2) 
            processed_images.append(row)
        image = processed_images

        instruction = get_layout_instruction(len(image_list[0]), context_num)
        if random.random() < (0.8 if training_mode else 1.0):
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            instruction = instruction + " " + get_task_instruction(condition_prompt, target_prompt)
        if random.random() < (0.8 if training_mode else 1.0) and image_type_list[-1] == "target":
            instruction = instruction + " " + get_content_instruction() + data_item[i]['description']['item'] + " " + data_item[i]['description']['description_0']
                
        return group_name, image, instruction, None, (len(image_list[0]), len(image_list))

    def process_item(self, data_item, training_mode=False, image_type_list=None, context_num=1, group_name=None):
        
        if group_name == 'image_grid_graph200k':
            return self.graph200k_process_item(data_item, image_type_list, context_num, group_name=group_name, training_mode=training_mode)
        else:
            raise ValueError(f"Unknown data item: {data_item}")
