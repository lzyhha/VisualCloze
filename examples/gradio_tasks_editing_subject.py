import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image


task_instruction = "Every row demonstrates how to transform [IMAGE1] a reference image showcasing the dominant object, [IMAGE2] a high-quality image into [IMAGE3] a high-quality image through a logical approach."
content_instruction = "The last image of the final row displays: "
editing_with_subject = [
    dict(
        name='Editing with Subject', 
        examples=[
            dict(
                images=[
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-3633/data-00004-of-00022-3633_reference.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-3633/data-00004-of-00022-3633_qwen_subject_replacement_1737373818845_1.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-3633/data-00004-of-00022-3633_target.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-5419/data-00004-of-00022-5419_reference.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-5419/data-00004-of-00022-5419_qwen_subject_replacement_1737377830929_2.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-5419/data-00004-of-00022-5419_target.jpg'),
                ], 
                grid_h=2,
                grid_w=3,
                task_prompt=task_instruction, 
                content_prompt=content_instruction + "A sacred, serene marble religious sculpture. Perched on a rocky cliff overlooking the ocean, this item appears ethereal as the first light of dawn paints it in soft pink and gold hues, with waves crashing in the background.",
            ), 
            dict(
                images=[
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-3633/data-00004-of-00022-3633_reference.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-3633/data-00004-of-00022-3633_qwen_subject_replacement_1737373818845_1.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00004-of-00022-3633/data-00004-of-00022-3633_target.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00012-of-00022-8475/data-00012-of-00022-8475_reference.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00012-of-00022-8475/data-00012-of-00022-8475_qwen_subject_replacement_1737410088010_2.jpg'),
                    os.path.join('examples/examples/graph200k/editing/data-00012-of-00022-8475/data-00012-of-00022-8475_target.jpg'),
                ], 
                grid_h=2,
                grid_w=3,
                task_prompt=task_instruction, 
                content_prompt=content_instruction + "A crisp, golden lager in a glass. Nestled beside a flickering fireplace, it casts a cozy, amber glow on the rustic wooden floor of a mountain cabin, inviting sips after a day in the snow.",
            )
        ]
    ),
]
editing_with_subject_text = [[x['name']] for x in editing_with_subject]


def process_editing_with_subject_tasks(x):
    for task in editing_with_subject:
        if task['name'] == x[0]:
            example = random.choice(task['examples'])
            task_prompt = example['task_prompt']
            content_prompt = example['content_prompt']

            images = example['images']
            rets = []
            for image in images:
                rets.append(Image.open(image))

            grid_h = example['grid_h']
            grid_w = example['grid_w']
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)

            upsampling_noise = None
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs
