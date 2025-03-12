import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image


task_instruction ="Every row demonstrates how to transform [IMAGE1] an image with flawless clarity into [IMAGE2] an image with artistic doodle embellishments through a logical approach."
content_instruction = "The photo doodle effect in the last row is: "
photodoodle = [
    dict(
        name='sksmonstercalledlulu', 
        images=[
            os.path.join('examples/examples/photodoodle/sksmonstercalledlulu/5.jpg'),
            os.path.join('examples/examples/photodoodle/sksmonstercalledlulu/5_blend.jpg'),
            os.path.join('examples/examples/photodoodle/sksmonstercalledlulu/6.jpg'),
            os.path.join('examples/examples/photodoodle/sksmonstercalledlulu/6_blend.jpg'),
            os.path.join('examples/examples/photodoodle/sksmonstercalledlulu/9.jpg'),
            os.path.join('examples/examples/photodoodle/sksmonstercalledlulu/9_blend.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction, 
        content_prompt=content_instruction + "add a large pink furry monster called 'Lulu' behind the girl, hugging her. The monster should have a large single eye with a dark pupil, round pink ears, small white teeth, and fluffy texture. Position the monster so that its arms wrap gently around the girl from behind, with its head slightly leaning to the left above her. Make sure the monster's body is large and visible, overlapping the floor and partially obscuring the carpet pattern.",
    ),
     dict(
        name='skspaintingeffects', 
        images=[
            os.path.join('examples/examples/photodoodle/skspaintingeffects/12.jpg'),
            os.path.join('examples/examples/photodoodle/skspaintingeffects/12_blend.jpg'),
            os.path.join('examples/examples/photodoodle/skspaintingeffects/35.jpg'),
            os.path.join('examples/examples/photodoodle/skspaintingeffects/35_blend.jpg'),
            os.path.join('examples/examples/photodoodle/skspaintingeffects/37.jpg'),
            os.path.join('examples/examples/photodoodle/skspaintingeffects/37_blend.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction, 
        content_prompt=content_instruction + "add a cluster of colorful daisies to the left side of the kitten's face. Use alternating blue and orange petals with pink centers for each flower. Overlay a design of mixed purple and brown wavy shapes around the top and right sides of the image, creating an abstract artistic effect. Keep the rest of the background unchanged.",
    ),
     dict(
        name='sksmagiceffects', 
        images=[
            os.path.join('examples/examples/photodoodle/sksmagiceffects/29.jpg'),
            os.path.join('examples/examples/photodoodle/sksmagiceffects/29_blend.jpg'),
            os.path.join('examples/examples/photodoodle/sksmagiceffects/50.jpg'),
            os.path.join('examples/examples/photodoodle/sksmagiceffects/50_blend.jpg'),
            os.path.join('examples/examples/photodoodle/sksmagiceffects/24.jpg'),
            os.path.join('examples/examples/photodoodle/sksmagiceffects/24_blend.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction, 
        content_prompt=content_instruction + "add a large, yellow crescent moon to the top section of the circular structure. Place five large, yellow stars around the building.",
    ),
     dict(
        name='sksedgeeffect', 
        images=[
            os.path.join('examples/examples/photodoodle/sksedgeeffect/34.jpg'),
            os.path.join('examples/examples/photodoodle/sksedgeeffect/34_blend.jpg'),
            os.path.join('examples/examples/photodoodle/sksedgeeffect/1.jpg'),
            os.path.join('examples/examples/photodoodle/sksedgeeffect/1_blend.jpg'),
        ], 
        grid_h=2,
        grid_w=2,
        task_prompt=task_instruction, 
        content_prompt=content_instruction + "add artistic flower shapes around the edge of the elderly woman. Use a purple flower with a black outline on the top right, a pink flower with a red outline on the top left, and a white flower with a black outline on the bottom right. Surround the woman's silhouette with a blue outline, inside a pink outline, creating a layered edge effect.",
    )
]
photodoodle_text = [[x['name']] for x in photodoodle]


def process_photodoodle_tasks(x):
    for task in photodoodle:
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
