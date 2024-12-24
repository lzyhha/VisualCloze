import json
from data import ItemProcessor, MyDataset
from data_reader import read_general
from data.prefix_instruction import get_layout_instruction, get_task_instruction, get_content_instruction, get_image_prompt, task_dicts, condition_list, degradation_list, style_list, editing_list
from degradation_utils import add_degradation
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop


def resize_with_aspect_ratio(img, resolution, divisible=16):
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

    def jimeng_ocr_process_item(self, data_item):
        data_item = data_item[0]
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
            if "bbox" in image_type:
                source_image_path = data_item["condition"]["qwen_grounding_caption"]["qwen_grounding_caption_bbox"]
            elif "mask" in image_type:
                source_image_path = data_item["condition"]["qwen_grounding_caption"]["qwen_grounding_caption_mask"]
            try:
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
            if image_type == "InstantStyle":
                source_dict = data_item["condition"]["instantx_style"]
            elif image_type == "ReduxStyle":
                source_dict = data_item["condition"]["flux_redux_style_shaping"]
            style_idx = random.randint(0, 2)
            try:
                style_image = source_dict["style_path"][style_idx]
                style_image = Image.open(read_general(style_image)).convert("RGB")
            except Exception as e:
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {str(e)}")
            try:
                target_image = source_dict["image_name"][style_idx]
                target_image = Image.open(read_general(target_image)).convert("RGB")
            except Exception as e:
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {str(e)}")
            return [style_image, target_image]
        elif image_type in editing_list:
            if image_type == "DepthEdit":
                editing_image_path = data_item["conditiomn"]["flux_dev_depth"]
            elif image_type == "FrontEdit":
                editing_image_path = random.choice(data_item["conditiomn"]["qwen_subject_replacement"]["image_name"])
            editing_image = Image.open(editing_image_path).convert('RGB')
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

                
    def object200k_process_item(self, data_item, image_type_list=None):
        text_emb = None
        context_num_prob = [0.3, 0.4, 0.3]  # 概率分别对应context_num=1,2,3
        context_num = random.choices([1, 2, 3], weights=context_num_prob)[0]
        task_weight_prob = [task["sample_weight"] for task in task_dicts]  # 概率分别对应task_weight=0.3,1
        block_list = ["depth_anything_v2", "qwen_mask", "qwen_bbox", "frontground", "background", "InstantStyle", "ReduxStyle", "DepthEdit"]
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
                images = [resize_with_aspect_ratio(image, self.resolution) for image in images]
                image_list[i] += images

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
            image_prompt_list = []
            for image_type in image_type_list:
                image_prompt_list += get_image_prompt(image_type)
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            # if len(image_type_list) > 2:
            #     condition_prompt = ", ".join([get_image_prompt(image_type) for image_type in image_type_list[:-2]]) + ", and " + get_image_prompt(image_type_list[-2])
            # else:
            #     condition_prompt = get_image_prompt(image_type_list[0])
            # target_prompt = get_image_prompt(image_type_list[-1])
            instruction = instruction + " " + get_task_instruction(condition_prompt, target_prompt)
        if random.random() < 0.8 and image_type_list[-1] == "target":
            instruction = instruction + " " + get_content_instruction() + data_item[i]['description']['item'] + " " + data_item[i]['description']['description_0']
        
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
        

    def process_item(self, data_item, training_mode=False):
        text_emb = None
        
        if "input_path" in data_item:
            input_image, target_image, text = self.pixwizard_process_item(data_item)
            return self.output_pair_data(input_image, target_image, text, text_emb)
        elif "condition" in data_item:
            input_image, target_image, text = self.lige_process_item(data_item)
            return self.output_pair_data(input_image, target_image, text, text_emb)
        elif "caption_en" in data_item[0]:
            input_image, target_image, text = self.jimeng_ocr_process_item(data_item)
            return self.output_pair_data(input_image, target_image, text, text_emb)
        elif isinstance(data_item, list) and "description" in data_item[0]:
            return self.object200k_process_item(data_item)
            # cond_images_1, cond_images_2, ref_images, target_images, texts = self.object200k_process_item(data_item)
            # return self.output_grid_data(cond_images_1, cond_images_2, ref_images, target_images, texts, text_emb)
        else:
            raise ValueError(f"Unknown data item: {data_item}")


def main():
    with open('/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_test_v6.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
    )   
    data_processor = T2IItemProcessor(image_transform, resolution=512)

    interest_list = ["InstantStyle", "ReduxStyle", "DepthEdit", "FrontEdit"]
    for index, data_item in enumerate(data):
        if index > 50:
            break
        for task_type in task_dicts:
            for image_type_list in task_type["image_list"]:
                if not any(block_type in image_type_list for block_type in interest_list):
                    continue
                data_items = [data[index-2], data[index-1], data_item]
                try:
                    image, instruction, text_emb = data_processor.object200k_process_item(data_items, image_type_list)
                except Exception as e:
                    continue
                print(instruction)
                # 将tensor转换为PIL图像并保存
                image_name = "_".join(image_type_list)  # 用image type list中的元素拼接文件名
                image_tensor = image.cpu().numpy()  # 转到CPU并转为numpy数组
                image_array = np.transpose(image_tensor, (1, 2, 0))  # 调整维度顺序从CxHxW到HxWxC
                image_array = ((image_array + 1) * 127.5).clip(0, 255).astype(np.uint8)  # 转换值域从[-1,1]到[0,255]
                image_pil = Image.fromarray(image_array)
                image_pil.save(f"object200k_test_gt_nxn_v6/sample_{index}_{image_name}.png")  # 保存为PNG格式

if __name__ == "__main__":
    main()

