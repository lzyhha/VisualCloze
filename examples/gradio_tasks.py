import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image
import numpy as np


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
    generate_paths_from_id('5bf755ed9dbb9b3e223e7ba35232b06e', prompt="A whimsical digital illustration of an astronaut emerging from a cracked eggshell on a barren, moon-like surface. The astronaut is centrally positioned, wearing a white space suit with a reflective visor helmet, holding a small yellow flag with the words 'HELLO WORLD' in black text. The eggshell is partially buried in the textured, rocky terrain, with scattered rocks and dust around it. The background is a blurred, dark blue gradient with circular bokeh effects, suggesting a distant, starry space environment. Soft, warm lighting from the top right creates a gentle glow on the astronaut's suit and the flag, adding a sense of discovery and playfulness. Digital illustration, shallow depth of field, soft focus, warm color palette, whimsical, surreal, high contrast, glossy textures, imaginative, visually balanced."), 
    generate_paths_from_id('9c565b1aad76b22f5bb836744a93561a', prompt="Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."),
    generate_paths_from_id('9d39f75f1f728e097efeaff39acb4710', prompt="Serene beach scene at sunrise with a clear blue sky and calm ocean waves. The foreground features sandy beach with footprints leading towards the water, and a large, twisted pine tree with lush green foliage arching over the scene from the right. The sun is rising on the horizon, casting a warm glow and long shadows on the sand. In the background, a rocky outcrop covered with greenery is visible to the left. The ocean stretches out to the right, with gentle waves lapping at the shore. Photorealistic, high contrast, vibrant colors, natural lighting, warm color palette, tranquil atmosphere, balanced composition, sharp details, inviting and peaceful."), 
    generate_paths_from_id('012cd3921e1f97d761eeff580f918ff9', prompt="Portrait of a young woman with long dark hair styled in an elegant updo, smiling directly at the camera. She is wearing a white, floral-embroidered strapless dress, positioned slightly to the right of the frame. Her makeup is subtle yet polished, with a focus on her eyes and lips. She wears a pair of dangling, ornate earrings. Surrounding her are vibrant red roses and lush green foliage, creating a natural and romantic backdrop. The lighting is soft and natural, highlighting her features and casting gentle shadows. The image has a shallow depth of field, with the background softly blurred, emphasizing the subject. Photorealistic, warm color palette, high contrast, intimate, serene, visually balanced."),
    generate_paths_from_id('53b3f413257bee9e499b823b44623b1a', prompt="A stunning photograph of a red fox standing in a snowy landscape, gazing intently at a small, icy stream in the foreground. The fox is positioned slightly to the left, its vibrant orange fur contrasting with the white snow. Surrounding the fox are delicate branches covered in frost, adding texture to the scene. Above, icicles hang from the branches, catching the light and creating a sense of cold. The reflection of the fox is visible in the still water, enhancing the symmetry of the composition. The background is softly blurred, with hints of blue and white suggesting a serene winter environment. High contrast, sharp focus on the fox, soft natural lighting, cool color palette with warm highlights, photorealistic, tranquil, visually balanced, ethereal winter atmosphere."),
    generate_paths_from_id('78dc6506367d7aa43fe42a898abbfe4a', prompt="Ethereal digital illustration of a winged woman standing beside a majestic lion on a rocky outcrop. The woman, positioned slightly to the left, wears a flowing, cream-colored gown with intricate detailing and a red sash at the waist. Her long, dark hair cascades down her back, and she holds a golden, ornate vessel in her right hand. The lion stands to her right, its mane richly textured and its gaze directed forward. The background features a vibrant sky with fluffy clouds and a bright sun, casting a warm glow over the scene. The foreground includes delicate orange flowers and tall grasses, adding a touch of nature to the composition. Digital art, high contrast, vivid color palette, soft lighting, surreal and fantastical atmosphere, detailed textures, dynamic composition, harmonious balance, ethereal and majestic mood."),
    generate_paths_from_id('79f2ee632f1be3ad64210a641c4e201b', prompt="A serene portrait of a young woman with long dark hair, wearing a beige dress with intricate gold embroidery, standing in a softly lit room. She holds a large bouquet of pale pink roses in a black box, positioned in the center of the frame. The background features a tall green plant to the left and a framed artwork on the wall to the right. A window on the left allows natural light to gently illuminate the scene. The woman gazes down at the bouquet with a calm expression. Soft natural lighting, warm color palette, high contrast, photorealistic, intimate, elegant, visually balanced, serene atmosphere."),
    generate_paths_from_id('88d0ba30e2c0bc4401cf2633cac162d4', prompt="A serene cinematic still of a woman with long, platinum blonde hair sitting on a rocky shore, facing the ocean. She wears a long, dark green dress with intricate detailing on the sleeves. Her expression is joyful, looking upwards towards a black bird in mid-flight, positioned in the upper left of the frame. The ocean waves gently crash in the background, creating a soft, rhythmic pattern. The sky is overcast, casting a diffused, cool light over the scene. Cinematic still, medium depth of field, soft natural lighting, muted color palette, ethereal and tranquil atmosphere, visually balanced composition, gentle contrast, serene and contemplative mood."),
    generate_paths_from_id('93bc1c43af2d6c91ac2fc966bf7725a2', prompt="Illustration of a young woman with long, wavy red hair sitting at a round wooden table in a sunlit café. She is positioned slightly to the right, holding a white cup in her right hand, looking directly at the viewer with a gentle smile. She wears a white long-sleeve top and blue jeans. On the table, there is a croissant, a bowl of jam, a cup of coffee, and an open magazine. The background features large windows with a view of a street lined with trees and parked cars, blurred to suggest motion. Potted plants are visible outside and inside the café. Warm, natural lighting, soft shadows, vibrant color palette, photorealistic textures, cozy and inviting atmosphere, digital illustration, high contrast, serene and relaxed mood."),
    generate_paths_from_id('10d7dcae5240b8cc8c9427e876b4f462', prompt="A stylish winter portrait of a young woman in a snowy landscape, wearing a brown fur coat, black turtleneck, and brown leather pants. She is positioned slightly to the right, looking down at her smartphone with a focused expression. A wide-brimmed brown cowboy hat sits atop her head. To her left, a Siberian Husky with striking blue eyes stands attentively, its fur a mix of black, white, and grey. The background features a blurred, desolate winter scene with bare trees and dry grasses, creating a serene and isolated atmosphere. The foreground includes snow-covered ground and sparse, dried plants. Photorealistic, medium depth of field, soft natural lighting, muted color palette, high contrast, fashion photography, sharp focus on the subject, tranquil, elegant, visually balanced."),
    generate_paths_from_id('0fdaecdb7906a1bf0d6e202363f15de3', prompt="A whimsical digital illustration of a retro-futuristic robot standing in a cozy, softly lit room. The robot, positioned centrally, has a metallic, spherical head with glowing red eyes and large headphones. It wears a brown leather vest and shorts, with articulated black arms and legs, and holds a vintage cassette tape in its right hand. The robot's feet are clad in brown and black boots. In the background, a blurred window with white frames is visible on the left, and a neon sign reading \"biogarty\" glows red on the right wall. Potted plants are placed on the floor and on a table in the background, adding a touch of greenery. The floor is wooden, and a laptop is partially visible in the left foreground. The scene is bathed in warm, natural light with a soft focus, creating a nostalgic and playful atmosphere. Digital illustration, medium depth of field, soft natural lighting, warm color palette, retro-futuristic, whimsical, visually balanced, glossy textures, cozy interior setting."), 
]


dense_prediction = [
    dict(
        name='Image to Depth', 
        image_type=["target", "depth"]),
    dict(
        name='Image to Canny', 
        image_type=["target", "canny"]),
    dict(
        name='Image to Hed', 
        image_type=["target", "hed"]),
    dict(
        name='Image to Normal', 
        image_type=["target", "normal"]),
    dict(
        name='Image to Pose', 
        image_type=["target", "openpose"]),
]
dense_prediction_text = [[x['name']] for x in dense_prediction]

conditional_generation = [
    dict(
        name='Depth to Image', 
        image_type=["depth", "target"]),
    dict(
        name='Foreground to Image', 
        image_type=["foreground", "target"]),
    dict(
        name='Background to Image', 
        image_type=["background", "target"]),
    dict(
        name='Canny to Image', 
        image_type=["canny", "target"]),
    dict(
        name='Hed to Image', 
        image_type=["hed", "target"]),
    dict(
        name='Normal to Image', 
        image_type=["normal", "target"]),
    dict(
        name='Pose to Image', 
        image_type=["openpose", "target"]),
    dict(
        name='Mask to Image', 
        image_type=["mask", "target"]),
    dict(
        name='SAM2 to Image', 
        image_type=["sam2_mask", "target"]),
]
conditional_generation_text = [[x['name']] for x in conditional_generation]


def process_dense_prediction_tasks(x):
    for task in dense_prediction:
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
            
            content_prompt = ""

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)
            
            upsampling_noise = 0.7
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs


def process_conditional_generation_tasks(x):
    for task in conditional_generation:
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
                    if t == "foreground":
                        mask = Image.open(image[t])
                        target_image = Image.open(image['target']).convert('RGB')
                        mask_np = np.array(mask).astype(np.float32) / 255.0
                        result = Image.fromarray((np.array(target_image) * mask_np).astype(np.uint8))
                        rets.append(result)
                    elif t == "background":
                        mask = Image.open(image[t])
                        target_image = Image.open(image['target']).convert('RGB')
                        mask_np = np.array(mask).astype(np.float32) / 255.0
                        mask_np = 1 - mask_np
                        result = Image.fromarray((np.array(target_image) * mask_np).astype(np.uint8))
                        rets.append(result)
                    else:
                        rets.append(Image.open(image[t]))

            content_prompt = get_content_instruction() + images[-1]['prompt']

            grid_h = n_samples
            grid_w = len(image_type)
            mask = task.get('mask', [0 for _ in range(grid_w - 1)] + [1])
            layout_prompt = get_layout_instruction(grid_w, grid_h)
            
            if 'Pose to Image' == task['name']:
                upsampling_noise = 0.3 
            else: 
                upsampling_noise = None
            steps = None
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs