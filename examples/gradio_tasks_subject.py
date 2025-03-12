import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image
from data.degradation_utils import add_degradation
import numpy as np


degradation_list = [
    # blur
    "blur",
    "compression",
    "SRx2",
    "SRx4",
    "pixelate",
    "Defocus",
    "GaussianBlur",
    # sharpen
    "oversharpen",
    # nosie
    "GaussianNoise",
    "PoissonNoise",
    "SPNoise",
    # mosaic
    "mosaic",
    # contrast
    "contrast_strengthen",
    "contrast_weaken",
    # quantization
    "quantization",
    "JPEG",
    # light
    "brighten",
    "darken",
    "LowLight",
    # color
    "saturate_strengthen",
    "saturate_weaken",
    "gray",
    "ColorDistortion",
    # infilling
    "Inpainting",
    # rotate
    "rotate180",
    # other
    "Barrel",
    "Pincushion",
    "Elastic",
    # spacial effect
    "Rain",
    "Frost",
]


def generate_paths_from_id(file_id: str, prompt: str) -> dict:
    """
    根据文件ID自动生成所有相关文件的路径
    
    Args:
        file_id: str - 文件的唯一标识符 (例如: '5c79f1ea582c3faa093d2e09b906321d')
        
    Returns:
        dict: 包含所有生成路径的字典
    """
    base_path = 'examples/examples/graph200k'
    
    paths = {
        'reference': f'{base_path}/{file_id}/{file_id}_reference.jpg',
        'target': f'{base_path}/{file_id}/{file_id}_target.jpg',
        'depth': f'{base_path}/{file_id}/{file_id}_depth-anything-v2_Large.jpg',
        'canny': f'{base_path}/{file_id}/{file_id}_canny_100_200_512.jpg',
        'hed': f'{base_path}/{file_id}/{file_id}_hed_512.jpg',
        'normal': f'{base_path}/{file_id}/{file_id}_dsine-normal-map.jpg',
        'style_target': f'{base_path}/{file_id}/{file_id}_instantx-style_0.jpg',
        'style_source': f'{base_path}/{file_id}/{file_id}_instantx-style_0_style.jpg', 
        'sam2_mask': f'{base_path}/{file_id}/{file_id}_sam2_mask.jpg',
        'prompt': prompt
    }
    
    return paths


dense_prediction_data = [
    generate_paths_from_id('data-00004-of-00022-7170', prompt="Travel VPN app on a desktop screen. The interface is visible on a laptop in a modern airport lounge, captured from a side angle with natural daylight highlighting the sleek design, while planes can be seen through the large window behind the device."),
    generate_paths_from_id('data-00005-of-00022-4396', prompt="A vintage porcelain collector's item. Beneath a blossoming cherry tree in early spring, this treasure is photographed up close, with soft pink petals drifting through the air and vibrant blossoms framing the scene."),
    generate_paths_from_id('data-00018-of-00022-4948', prompt="Decorative kitchen salt shaker with intricate design. On a quaint countryside porch in the afternoon's gentle breeze, accompanied by pastel-colored flowers and vintage cutlery, it adds a touch of charm to the rustic scene."),
    generate_paths_from_id('data-00013-of-00022-4696', prompt="A lifelike forest creature figurine. Nestled among drifting autumn leaves on a tree-lined walking path, it gazes out as pedestrians bundled in scarves pass by."),
    generate_paths_from_id('data-00017-of-00022-8377', prompt="A colorful bike for young adventurers. In a bustling city street during a bright afternoon, it leans against a lamppost, surrounded by hurried pedestrians, with towering buildings providing an urban backdrop."),
]


subject_driven = [
    dict(
        name='Subject-driven generation', 
        image_type=["reference", "target"]),
]
subject_driven_text = [[x['name']] for x in subject_driven]


style_transfer_with_subject = [
    dict(
        name='Style Transfer with Subject', 
        image_type=["reference", "style_source", "style_target"]),
]
style_transfer_with_subject_text = [[x['name']] for x in style_transfer_with_subject]


condition_subject_fusion = [
    dict(
        name='Depth+Subject to Image', 
        image_type=["reference", "depth", "target"]),
    dict(
        name='Canny+Subject to Image', 
        image_type=["reference", "canny", "target"]),
    dict(
        name='Hed+Subject to Image', 
        image_type=["reference", "hed", "target"]),
    dict(
        name='Normal+Subject to Image', 
        image_type=["reference", "normal", "target"]),
    dict(
        name='SAM2+Subject to Image', 
        image_type=["reference", "sam2_mask", "target"]),
]
condition_subject_fusion_text = [[x['name']] for x in condition_subject_fusion]

image_restoration_with_subject = [
    dict(name=degradation, image_type=["reference", degradation, "target"]) 
    for degradation in degradation_list
]
image_restoration_with_subject_text = [[x['name']] for x in image_restoration_with_subject]


condition_subject_style_fusion = [
    dict(
        name='Depth+Subject+Style to Image', 
        image_type=["reference", "depth", "style_source", "style_target"]),
    dict(
        name='Canny+Subject+Style to Image', 
        image_type=["reference", "canny", "style_source", "style_target"]),
    dict(
        name='Hed+Subject+Style to Image', 
        image_type=["reference", "hed", "style_source", "style_target"]),
    dict(
        name='Normal+Subject+Style to Image', 
        image_type=["reference", "normal", "style_source", "style_target"]),
    dict(
        name='SAM2+Subject+Style to Image', 
        image_type=["reference", "sam2_mask", "style_source", "style_target"]),
]
condition_subject_style_fusion_text = [[x['name']] for x in condition_subject_style_fusion]


def process_subject_driven_tasks(x):
    for task in subject_driven:
        if task['name'] == x[0]:
            image_type = task['image_type']
            image_prompt_list = [get_image_prompt(x)[0] for x in image_type]
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            task_prompt = get_task_instruction(condition_prompt, target_prompt)

            # sample examples
            valid_data = [x for x in dense_prediction_data if all([x.get(t, None) is not None and os.path.exists(x[t]) for t in image_type])]
            n_samples = random.randint(2, min(len(valid_data), 3))
            images = random.sample(valid_data, k=n_samples)
            rets = []
            for image in images:
                for t in image_type:
                    rets.append(Image.open(image[t]))
            
            content_prompt = get_content_instruction() + images[-1]['prompt']

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)

            upsampling_noise = 0.6
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs


def process_condition_subject_fusion_tasks(x):
    for task in condition_subject_fusion:
        if task['name'] == x[0]:
            image_type = task['image_type']
            image_prompt_list = [get_image_prompt(x)[0] for x in image_type]
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            task_prompt = get_task_instruction(condition_prompt, target_prompt)

            # sample examples
            valid_data = [x for x in dense_prediction_data if all([x.get(t, None) is not None and os.path.exists(x[t]) for t in image_type])]
            n_samples = random.randint(2, min(len(valid_data), 3))
            images = random.sample(valid_data, k=n_samples)
            rets = []
            for image in images:
                for t in image_type:
                    rets.append(Image.open(image[t]))

            content_prompt = get_content_instruction() + images[-1]['prompt']

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)

            upsampling_noise = 0.6
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs


def process_style_transfer_with_subject_tasks(x):
    for task in style_transfer_with_subject:
        if task['name'] == x[0]:
            image_type = task['image_type']
            image_prompt_list = [get_image_prompt(x)[0] for x in image_type]
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            task_prompt = get_task_instruction(condition_prompt, target_prompt)

            # sample examples
            valid_data = [x for x in dense_prediction_data if all([x.get(t, None) is not None and os.path.exists(x[t]) for t in image_type])]
            n_samples = random.randint(2, min(len(valid_data), 3))
            images = random.sample(valid_data, k=n_samples)
            rets = []
            for image in images:
                for t in image_type:
                    if t == "style_source":
                        target = Image.open(image["style_target"])
                        source = Image.open(image[t])
                        source = source.resize(target.size)
                        rets.append(source)
                    else:
                        rets.append(Image.open(image[t]))   

            content_prompt = ""

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)

            upsampling_noise = 0.6
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs  


def process_condition_subject_style_fusion_tasks(x):
    for task in condition_subject_style_fusion:
        if task['name'] == x[0]:
            image_type = task['image_type']
            image_prompt_list = [get_image_prompt(x)[0] for x in image_type]
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            task_prompt = get_task_instruction(condition_prompt, target_prompt)

            # sample examples
            valid_data = [x for x in dense_prediction_data if all([x.get(t, None) is not None and os.path.exists(x[t]) for t in image_type])]
            n_samples = random.randint(2, min(len(valid_data), 3))
            images = random.sample(valid_data, k=n_samples)
            rets = []
            for image in images:
                for t in image_type:
                    if t == "style_source":
                        target = Image.open(image["style_target"])
                        source = Image.open(image[t])
                        source = source.resize(target.size)
                        rets.append(source)
                    else:
                        rets.append(Image.open(image[t]))   

            content_prompt = ""

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)

            upsampling_noise = 0.6
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs  


def process_image_restoration_with_subject_tasks(x):
    for task in image_restoration_with_subject:
        if task['name'] == x[0]:
            image_type = task['image_type']
            image_prompt_list = [get_image_prompt(x)[0] for x in image_type]
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            task_prompt = get_task_instruction(condition_prompt, target_prompt)

            # sample examples
            valid_data = dense_prediction_data
            n_samples = random.randint(2, min(len(valid_data), 3))
            images = random.sample(valid_data, k=n_samples)
            rets = []
            for image in images:
                for t in image_type:
                    if t == "target":
                        rets.append(Image.open(image["target"]))
                    elif t == "reference":
                        rets.append(Image.open(image["reference"]))
                    else:
                        deg_image, _ = add_degradation(np.array(Image.open(image["target"])), deg_type=t)
                        rets.append(deg_image)

            content_prompt = get_content_instruction() + images[-1]['prompt']

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)

            upsampling_noise = 0.6
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs
