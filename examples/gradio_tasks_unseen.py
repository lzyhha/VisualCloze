import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image


def generate_paths_from_id(file_id: str, prompt: str) -> dict:
    """
    根据文件ID自动生成所有相关文件的路径
    
    Args:
        file_id: str - 文件的唯一标识符 (例如: '5c79f1ea582c3faa093d2e09b906321d')
        
    Returns:
        dict: 包含所有生成路径的字典
    """
    base_path = 'examples/examples'
    
    paths = {
        'target': f'{base_path}/{file_id}/{file_id}.jpg',
        'depth': f'{base_path}/{file_id}/{file_id}_depth-anything-v2_Large.jpg',
        'canny': f'{base_path}/{file_id}/{file_id}_canny_100_200_512.jpg',
        'hed': f'{base_path}/{file_id}/{file_id}_hed_512.jpg',
        'normal': f'{base_path}/{file_id}/{file_id}_dsine_normal_map.jpg',
        'openpose': f'{base_path}/{file_id}/{file_id}_openpose_fullres_nohand.jpg',
        'style_target': f'{base_path}/{file_id}/{file_id}_instantx-style_0.jpg',
        'style_source': f'{base_path}/{file_id}/{file_id}_instantx-style_0_style.jpg', 
        'foreground': f'{base_path}/{file_id}/{file_id}_ben2-background-removal.jpg', 
        'background': f'{base_path}/{file_id}/{file_id}_ben2-background-removal.jpg', 
        'mask': f'{base_path}/{file_id}/{file_id}_qwen2_5_mask.jpg',
        'sam2_mask': f'{base_path}/{file_id}/{file_id}_sam2_mask.jpg',
        'prompt': prompt
    }
    
    return paths


dense_prediction_data = [
    generate_paths_from_id('2b74476568f7562a6aa832d423132ed3', prompt="Group photo of five young adults enjoying a rooftop gathering at dusk. The group is positioned in the center, with three women and two men smiling and embracing. The woman on the far left wears a floral top and holds a drink, looking slightly to the right. Next to her, a woman in a denim jacket stands close to a woman in a white blouse, both smiling directly at the camera. The fourth woman, in an orange top, stands close to the man on the far right, who wears a red shirt and blue blazer, smiling broadly. The background features a cityscape with a tall building and string lights hanging overhead, creating a warm, festive atmosphere. Soft natural lighting, warm color palette, shallow depth of field, intimate and joyful mood, slightly blurred background, urban rooftop setting, evening ambiance."),
    generate_paths_from_id('de5a8b250bf407aa7e04913562dcba90', prompt="Close-up photo of a refreshing peach iced tea in a clear plastic cup, centrally positioned on a wooden surface. The drink is garnished with fresh mint leaves and ice cubes, with a yellow and white striped straw angled to the left. Surrounding the cup are whole and sliced peaches, scattered across the table, with their vibrant orange flesh and brown skin visible. The background is softly blurred, featuring bokeh effects from sunlight filtering through green foliage, creating a warm and inviting atmosphere. High contrast, natural lighting, shallow depth of field, vibrant color palette, photorealistic, glossy texture, summer vibe, visually balanced composition."),
    generate_paths_from_id('2c4e256fa512cb7e7f433f4c7f9101de', prompt="A digital illustration of a small orange tabby kitten sitting in the center of a sunlit meadow, surrounded by white daisies with yellow centers. The kitten has large, expressive eyes and a pink nose, positioned directly facing the viewer. The daisies are scattered around the kitten, with some in the foreground and others in the background, creating a sense of depth. The background is softly blurred, emphasizing the kitten and flowers, with warm, golden sunlight filtering through, casting a gentle glow. Digital art, photorealistic, shallow depth of field, soft natural lighting, warm color palette, high contrast, serene, whimsical, visually balanced, intimate, detailed textures."),
    generate_paths_from_id('78dc6506367d7aa43fe42a898abbfe4a', prompt="Ethereal digital illustration of a winged woman standing beside a majestic lion on a rocky outcrop. The woman, positioned slightly to the left, wears a flowing, cream-colored gown with intricate detailing and a red sash at the waist. Her long, dark hair cascades down her back, and she holds a golden, ornate vessel in her right hand. The lion stands to her right, its mane richly textured and its gaze directed forward. The background features a vibrant sky with fluffy clouds and a bright sun, casting a warm glow over the scene. The foreground includes delicate orange flowers and tall grasses, adding a touch of nature to the composition. Digital art, high contrast, vivid color palette, soft lighting, surreal and fantastical atmosphere, detailed textures, dynamic composition, harmonious balance, ethereal and majestic mood."),
    generate_paths_from_id('79f2ee632f1be3ad64210a641c4e201b', prompt="A serene portrait of a young woman with long dark hair, wearing a beige dress with intricate gold embroidery, standing in a softly lit room. She holds a large bouquet of pale pink roses in a black box, positioned in the center of the frame. The background features a tall green plant to the left and a framed artwork on the wall to the right. A window on the left allows natural light to gently illuminate the scene. The woman gazes down at the bouquet with a calm expression. Soft natural lighting, warm color palette, high contrast, photorealistic, intimate, elegant, visually balanced, serene atmosphere."),
    generate_paths_from_id('88d0ba30e2c0bc4401cf2633cac162d4', prompt="A serene cinematic still of a woman with long, platinum blonde hair sitting on a rocky shore, facing the ocean. She wears a long, dark green dress with intricate detailing on the sleeves. Her expression is joyful, looking upwards towards a black bird in mid-flight, positioned in the upper left of the frame. The ocean waves gently crash in the background, creating a soft, rhythmic pattern. The sky is overcast, casting a diffused, cool light over the scene. Cinematic still, medium depth of field, soft natural lighting, muted color palette, ethereal and tranquil atmosphere, visually balanced composition, gentle contrast, serene and contemplative mood."),
    generate_paths_from_id('0fdaecdb7906a1bf0d6e202363f15de3', prompt="A whimsical digital illustration of a retro-futuristic robot standing in a cozy, softly lit room. The robot, positioned centrally, has a metallic, spherical head with glowing red eyes and large headphones. It wears a brown leather vest and shorts, with articulated black arms and legs, and holds a vintage cassette tape in its right hand. The robot's feet are clad in brown and black boots. In the background, a blurred window with white frames is visible on the left, and a neon sign reading \"biogarty\" glows red on the right wall. Potted plants are placed on the floor and on a table in the background, adding a touch of greenery. The floor is wooden, and a laptop is partially visible in the left foreground. The scene is bathed in warm, natural light with a soft focus, creating a nostalgic and playful atmosphere. Digital illustration, medium depth of field, soft natural lighting, warm color palette, retro-futuristic, whimsical, visually balanced, glossy textures, cozy interior setting."), 
]


def generate_paths_from_id(file_id: str, prompt: str) -> dict:
    """
    根据文件ID自动生成所有相关文件的路径
    
    Args:
        file_id: str - 文件的唯一标识符 (例如: '5c79f1ea582c3faa093d2e09b906321d')
        
    Returns:
        dict: 包含所有生成路径的字典
    """
    base_path = 'examples/examples'
    
    paths = {
        'target': f'{base_path}/{file_id}/{file_id}.jpg',
        'depth': f'{base_path}/{file_id}/{file_id}_depth-anything-v2_Large.jpg',
        'canny': f'{base_path}/{file_id}/{file_id}_canny_100_200_512.jpg',
        'hed': f'{base_path}/{file_id}/{file_id}_hed_512.jpg',
        'normal': f'{base_path}/{file_id}/{file_id}_dsine_normal_map.jpg',
        'openpose': f'{base_path}/{file_id}/{file_id}_openpose_fullres_nohand.jpg',
        'style_target': f'{base_path}/{file_id}/{file_id}_instantx-style_0.jpg',
        'style_source': f'{base_path}/{file_id}/{file_id}_instantx-style_0_style.jpg', 
        'foreground': f'{base_path}/{file_id}/{file_id}_ben2-background-removal.jpg', 
        'background': f'{base_path}/{file_id}/{file_id}_ben2-background-removal.jpg', 
        'mask': f'{base_path}/{file_id}/{file_id}_qwen2_5_mask.jpg',
        'sam2_mask': f'{base_path}/{file_id}/{file_id}_sam2_mask.jpg',
        'prompt': prompt
    }
    
    return paths


unseen_tasks = [
    dict(
        name='Frontal Face Reconstruction', 
        images=[
            'examples/examples/face/34e1633a-369f-4324-86c3-3e6418ec00be/face_0.jpg',
            'examples/examples/face/34e1633a-369f-4324-86c3-3e6418ec00be/face_2.jpg',
            'examples/examples/face/34e1633a-369f-4324-86c3-3e6418ec00be/face_1.jpg',
            'examples/examples/face/cb5d403a-f1bb-4392-8302-24846893a797/face_0.jpg',
            'examples/examples/face/cb5d403a-f1bb-4392-8302-24846893a797/face_2.jpg',
            'examples/examples/face/cb5d403a-f1bb-4392-8302-24846893a797/face_1.jpg',
            'examples/examples/face/2ef6aa5a-e751-4bf2-a302-0237ab460627/face_8.jpg',
            'examples/examples/face/2ef6aa5a-e751-4bf2-a302-0237ab460627/face_6.jpg',
            'examples/examples/face/2ef6aa5a-e751-4bf2-a302-0237ab460627/face_1.jpg',
        ], 
        grid_h=3,
        grid_w=3,
        task_prompt="Each row presents multi-view of a face, given a frontal face reconstruction task that leverages [IMAGE1] a left side of the face and [IMAGE2] a right side of the face, to generate [IMAGE3] a frontal face that faces the center of the lens.", 
        content_prompt="The content of the last image in the final row is: the woman's frontal face that faces the center of the lens.", 
    ),
    dict(
        name='Image to Depth + Normal + Hed', 
        image_type=["target", "depth", "normal", "hed"], 
        mask=[0, 1, 1, 1],
    ),
    dict(
        name='Depth to Image + Relighting', 
        examples=[
            dict(
                images=[
                    'examples/examples/relighting/02dad6943d2033198a89c1d5f222db2eacb293c6_depth.jpg',
                    'examples/examples/relighting/02dad6943d2033198a89c1d5f222db2eacb293c6.jpg',
                    'examples/examples/relighting/02dad6943d2033198a89c1d5f222db2eacb293c6_Left.jpg',
                    'examples/examples/relighting/02af9fa52ff41e64de8e3212683c9ed43bd91010_depth.jpg',
                    'examples/examples/relighting/02af9fa52ff41e64de8e3212683c9ed43bd91010.jpg',
                    'examples/examples/relighting/02af9fa52ff41e64de8e3212683c9ed43bd91010_Left.jpg',
                ], 
                grid_h=2,
                grid_w=3,
                task_prompt="Each row outlines a logical process, starting from [IMAGE1] depth map highlighting gray-level depth variations, to achieve [IMAGE2] an image with flawless clarity and [IMAGE3] the image with manipulated illumination and changed background.",
                content_prompt="In the last row, the illumination comes from left side of the image, the light effects are " + "light and shadow.",
                mask=[0, 1, 1],
            ),
            dict(
                images=[
                    'examples/examples/relighting/02dd1c7c81e77e22ddba378a121fc371afcc9657_depth.jpg',
                    'examples/examples/relighting/02dd1c7c81e77e22ddba378a121fc371afcc9657.jpg',
                    'examples/examples/relighting/02dd1c7c81e77e22ddba378a121fc371afcc9657_Left.jpg',
                    #
                    'examples/examples/relighting/02dcc762ae13127e3975ec043f13342490f61cf8_depth.jpg',
                    'examples/examples/relighting/02dcc762ae13127e3975ec043f13342490f61cf8.jpg',
                    'examples/examples/relighting/02dcc762ae13127e3975ec043f13342490f61cf8_Left.jpg',
                    #
                    'examples/examples/relighting/02dd0f49dceaf611e0173319e26b4e6e1b7a6dd4_depth.jpg',
                    'examples/examples/relighting/02dd0f49dceaf611e0173319e26b4e6e1b7a6dd4.jpg',
                    'examples/examples/relighting/02dd0f49dceaf611e0173319e26b4e6e1b7a6dd4_Left.jpg',
                ], 
                grid_h=3,
                grid_w=3,
                task_prompt="Each row outlines a logical process, starting from [IMAGE1] depth map highlighting gray-level depth variations, to achieve [IMAGE2] an image with flawless clarity and [IMAGE3] the image with manipulated illumination and changed background.",
                content_prompt="In the last row, the illumination comes from left side of the image, the light effects are " + "shadow from window.",
                mask=[0, 1, 1],
            )
        ],
    ),
    dict(
        name='Pose + Edge to Image', 
        examples=[
            dict(
                images=[
                    'examples/examples/2b74476568f7562a6aa832d423132ed3/2b74476568f7562a6aa832d423132ed3_openpose_fullres_nohand.jpg',
                    'examples/examples/2b74476568f7562a6aa832d423132ed3/2b74476568f7562a6aa832d423132ed3_hed_512.jpg',
                    'examples/examples/2b74476568f7562a6aa832d423132ed3/2b74476568f7562a6aa832d423132ed3.jpg',
                    'examples/examples/78dc6506367d7aa43fe42a898abbfe4a/78dc6506367d7aa43fe42a898abbfe4a_openpose_fullres_nohand.jpg',
                    'examples/examples/78dc6506367d7aa43fe42a898abbfe4a/78dc6506367d7aa43fe42a898abbfe4a_edge.jpg',
                    'examples/examples/78dc6506367d7aa43fe42a898abbfe4a/78dc6506367d7aa43fe42a898abbfe4a.jpg',
                ], 
                grid_h=2,
                grid_w=3,
                task_prompt="Every row demonstrates how to transform [IMAGE1] human pose with colored lines for bone structure and [IMAGE2] canny map with sharp white edges and dark into [IMAGE3] a visually striking and clear picture through a logical approach.", 
                content_prompt="The content of the last image in the concluding row is: Ethereal digital illustration of a winged woman standing beside a majestic lion on a rocky outcrop. The woman, positioned slightly to the left, wears a flowing, cream-colored gown with intricate detailing and a red sash at the waist. Her long, dark hair cascades down her back, and she holds a golden, ornate vessel in her right hand. The lion stands to her right, its mane richly textured and its gaze directed forward. The background features a vibrant sky with fluffy clouds and a bright sun, casting a warm glow over the scene. The foreground includes delicate orange flowers and tall grasses, adding a touch of nature to the composition. Digital art, high contrast, vivid color palette, soft lighting, surreal and fantastical atmosphere, detailed textures, dynamic composition, harmonious balance, ethereal and majestic mood.", 
                mask=[0, 1],
            )
        ]
    ),
    dict(
        name='Attribute Transformation', 
        examples=[
            dict(
                images=[
                    'examples/examples/property/1_source.jpg',
                    'examples/examples/property/1_target.jpg',
                    'examples/examples/property/2_source.jpg',
                    'examples/examples/property/2_target.jpg',
                    'examples/examples/property/3_source.jpg',
                    'examples/examples/property/3_target.jpg',
                ], 
                grid_h=3,
                grid_w=2,
                task_prompt="In each row, a logical task is demonstrated to achieve [IMAGE2] a high-aesthetic image based on [IMAGE1] an aesthetically pleasing photograph. Each row shows a process to edit the image with the given editing instruction. The editing instruction in the last row is: <editing instruction> turn the color of the sunglasses to green. <\editing instruction>", 
                content_prompt="", 
                mask=[0, 1],
            )
        ]
    ),
    dict(
        name='Environment Modification', 
        examples=[
            dict(
                images=[
                'examples/examples/env/1_source.jpg',
                    'examples/examples/env/1_target.jpg',
                    'examples/examples/env/2_source.jpg',
                    'examples/examples/env/2_target.jpg',
                    'examples/examples/env/3_source.jpg',
                    'examples/examples/env/3_target.jpg',
                ], 
                grid_h=3,
                grid_w=2,
                task_prompt="In each row, a logical task is demonstrated to achieve [IMAGE2] a high-aesthetic image based on [IMAGE1] an aesthetically pleasing photograph. Each row shows a process to edit the image with the given editing instruction. The editing instruction in the last row is: <editing instruction> change the weather to a snowy scene in winter. <\editing instruction>", 
                content_prompt="", 
                mask=[0, 1],
            )
        ]
    )
]
unseen_tasks_text = [[x['name']] for x in unseen_tasks]


def process_unseen_tasks(x):
    for task in unseen_tasks:
        if 'Image to Depth + Normal + Hed' == x[0] == task['name']:
            image_type = task['image_type']
            image_prompt_list = [get_image_prompt(x)[0] for x in image_type]
            image_prompt_list = [f"[IMAGE{idx+1}] {image_prompt}" for idx, image_prompt in enumerate(image_prompt_list)]
            condition_prompt = image_prompt_list[0]
            target_prompt = ", ".join(image_prompt_list[1:])
            task_prompt = get_task_instruction(condition_prompt, target_prompt)
            # sample examples
            valid_data = [x for x in dense_prediction_data if all([x.get(t, None) is not None and os.path.exists(x[t]) for t in image_type])]
            n_samples = random.randint(3, min(len(valid_data), 3))
            images = random.sample(valid_data, k=n_samples)
            rets = []
            for image in images:
                for t in image_type:
                    rets.append(Image.open(image[t]))
            
            content_prompt = ""

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)
            
            upsampling_noise = 1.0
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break
        elif x[0] == task['name']:
            task = random.choice(task['examples'])
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
