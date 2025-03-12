import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image
from .gradio_tasks import dense_prediction_data
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


image_restoration = [dict(name=degradation, image_type=[degradation, "target"]) for degradation in degradation_list]
image_restoration_text = [[x['name']] for x in image_restoration]


def process_image_restoration_tasks(x):
    for task in image_restoration:
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
                    else:
                        deg_image, _ = add_degradation(np.array(Image.open(image["target"])), deg_type=t)
                        rets.append(deg_image)

            content_prompt = get_content_instruction() + images[-1]['prompt']

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)

            upsampling_noise = None
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs
