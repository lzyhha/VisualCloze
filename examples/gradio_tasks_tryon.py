import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image


task_instruction = "Each row shows a virtual try-on process that aims to put [IMAGE2] the clothing onto [IMAGE1] the person, producing [IMAGE3] the person wearing the new clothing."
content_instruction = ""
tryon = [
    dict(
        name='Virtual Try-On', 
        images=[
            os.path.join('examples/examples/tryon/00700_00.jpg'),
            os.path.join('examples/examples/tryon/03673_00.jpg'),
            os.path.join('examples/examples/tryon/00700_00_tryon_catvton_0.jpg'),
            os.path.join('examples/examples/tryon/00555_00.jpg'),
            os.path.join('examples/examples/tryon/12265_00.jpg'),
            os.path.join('examples/examples/tryon/00555_00_tryon_catvton_0.jpg'),
        ], 
        grid_h=2,
        grid_w=3,
        task_prompt=task_instruction, 
        content_prompt=content_instruction,
    ),
]
tryon_text = [[x['name']] for x in tryon]


def process_tryon_tasks(x):
    for task in tryon:
        if task['name'] == x[0]:
            task_prompt = task['task_prompt']
            content_prompt = task['content_prompt']

            images = task['images']
            rets = []
            for image in images:
                rets.append(Image.open(image))

            grid_h = task['grid_h']
            grid_w = task['grid_w']
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)

            upsampling_noise = None
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs
