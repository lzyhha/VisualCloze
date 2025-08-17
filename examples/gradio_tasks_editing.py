import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image


task_instruction = "In each row, a logical task is demonstrated to achieve [IMAGE2] a high-aesthetic image based on [IMAGE1] an aesthetically pleasing photograph. Each row shows a process to edit the image with the given editing instruction."
editing_instruction = "The editing instruction in the last row is: "
editing = [
    dict(
        name='add', 
        images=[
            os.path.join('examples/examples/omniedit/task_obj_add_273266.jpg'),
            os.path.join('examples/examples/omniedit/task_obj_add_273266_edit.jpg'),
            os.path.join('examples/examples/omniedit/task_obj_add_528329.jpg'),
            os.path.join('examples/examples/omniedit/task_obj_add_528329_edit.jpg'),
        ], 
        grid_h=2,
        grid_w=2,
        task_prompt=r"In each row, a logical task is demonstrated to achieve [IMAGE2] a high-aesthetic image based on [IMAGE1] an aesthetically pleasing photograph. Each row shows a process to edit the image with the given editing instruction. The editing instruction in the last row is: <editing instruction> Add a large hawk perched on a branch in the foreground. <\editing instruction>",
        content_prompt="",
    ), 
     dict(
        name='remove', 
        images=[
            os.path.join('examples/examples/omniedit/task_obj_add_528329_edit.jpg'),
            os.path.join('examples/examples/omniedit/task_obj_add_528329.jpg'),
            os.path.join('examples/examples/omniedit/task_obj_remove_855511_edit.jpg'),
            os.path.join('examples/examples/omniedit/task_obj_remove_855511.jpg'),
        ], 
        grid_h=2,
        grid_w=2,
        task_prompt=r"In each row, a logical task is demonstrated to achieve [IMAGE2] a high-aesthetic image based on [IMAGE1] an aesthetically pleasing photograph. Each row shows a process to edit the image with the given editing instruction. The editing instruction in the last row is: <editing instruction> Remove a small, orange and white monkey with black face sitting on a branch in the tree. <\editing instruction>",
        content_prompt="",
    ), 
]
editing_text = [[x['name']] for x in editing]


def process_editing_tasks(x):
    for task in editing:
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
