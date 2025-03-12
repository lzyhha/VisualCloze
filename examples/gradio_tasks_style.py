import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image
from .gradio_tasks import dense_prediction_data


style_transfer = [
    dict(
        name='Style Transfer', 
        image_type=["target", "style_source", "style_target"]),
]
style_transfer_text = [[x['name']] for x in style_transfer]


style_condition_fusion = [
    dict(
        name='Canny+Style to Image', 
        image_type=["canny", "style_source", "style_target"]),
    dict(
        name='Depth+Style to Image', 
        image_type=["depth", "style_source", "style_target"]),
    dict(
        name='Hed+Style to Image', 
        image_type=["hed", "style_source", "style_target"]),
    dict(
        name='Normal+Style to Image', 
        image_type=["normal", "style_source", "style_target"]),
    dict(
        name='Pose+Style to Image', 
        image_type=["openpose", "style_source", "style_target"]),
    dict(
        name='SAM2+Style to Image', 
        image_type=["sam2_mask", "style_source", "style_target"]),
    dict(
        name='Mask+Style to Image', 
        image_type=["mask", "style_source", "style_target"]),
]
style_condition_fusion_text = [[x['name']] for x in style_condition_fusion]


def process_style_transfer_tasks(x):
    for task in style_transfer:
        if task['name'] == x[0]:
            image_type = task['image_type']
            image_prompt_list = [get_image_prompt(x)[0] for x in image_type]
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            task_prompt = get_task_instruction(condition_prompt, target_prompt)

            # sample examples
            valid_data = [x for x in dense_prediction_data if all([(x.get(t, None) is not None and os.path.exists(x[t])) for t in image_type])]
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

            upsampling_noise = None
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs


def process_style_condition_fusion_tasks(x):
    for task in style_condition_fusion:
        if task['name'] == x[0]:
            image_type = task['image_type']
            image_prompt_list = [get_image_prompt(x)[0] for x in image_type]
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = ", ".join(image_prompt_list[:-1])
            target_prompt = image_prompt_list[-1]
            task_prompt = get_task_instruction(condition_prompt, target_prompt)

            # sample examples
            valid_data = [x for x in dense_prediction_data if all([(x.get(t, None) is not None and os.path.exists(x[t])) for t in image_type])]
            x = dense_prediction_data[0]
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

            upsampling_noise = None
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs
