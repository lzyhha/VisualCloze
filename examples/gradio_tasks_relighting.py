import os
from data.prefix_instruction import get_image_prompt, get_task_instruction, get_layout_instruction, get_content_instruction
import random
from PIL import Image


task_instruction = "Each row shows a process to manipulate the illumination of images and changes the background following the instruction."
content_instruction = "Beautiful woman, the illumination comes from left side of the image, "
relighting = [
    dict(
        name='sunset over sea', 
        images=[
            os.path.join('examples/examples/relighting/02daa50ac59bb9eabcbe0d5304af880d941bffc3.jpg'),
            os.path.join('examples/examples/relighting/02daa50ac59bb9eabcbe0d5304af880d941bffc3_Left.jpg'),
            os.path.join('examples/examples/relighting/02db8a5f38464943d496bd3b475c36a3d65e7095.jpg'),
            os.path.join('examples/examples/relighting/02db8a5f38464943d496bd3b475c36a3d65e7095_Left.jpg'),
            os.path.join('examples/examples/relighting/02db96d3ce2531dc4d51dda52492b78cf3577c56.jpg'),
            os.path.join('examples/examples/relighting/02db96d3ce2531dc4d51dda52492b78cf3577c56_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "sunset over sea.",
    ), 
    dict(
        name='light and shadow', 
        images=[
            os.path.join('examples/examples/relighting/02dad6943d2033198a89c1d5f222db2eacb293c6.jpg'),
            os.path.join('examples/examples/relighting/02dad6943d2033198a89c1d5f222db2eacb293c6_Left.jpg'),
            os.path.join('examples/examples/relighting/02db31cb32e74620523955b70807b3e11815451c.jpg'),
            os.path.join('examples/examples/relighting/02db31cb32e74620523955b70807b3e11815451c_Left.jpg'),
            os.path.join('examples/examples/relighting/02dcd82122ffe344c8d7c289dc770febb5121153.jpg'),
            os.path.join('examples/examples/relighting/02dcd82122ffe344c8d7c289dc770febb5121153_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "light and shadow.",
    ), 
    dict(
        name='sci-fi RGB glowing, cyberpunkw', 
        images=[
            os.path.join('examples/examples/relighting/02db5a81c222483058fecd76d62c5f7246b06ee4.jpg'),
            os.path.join('examples/examples/relighting/02db5a81c222483058fecd76d62c5f7246b06ee4_Left.jpg'),
            os.path.join('examples/examples/relighting/02db80670789cc6722f78747cf6ab8c292a898ab.jpg'),
            os.path.join('examples/examples/relighting/02db80670789cc6722f78747cf6ab8c292a898ab_Left.jpg'),
            os.path.join('examples/examples/relighting/02dc3e2cf9541a7d7ebff79cbf1fb0d95b4911e8.jpg'),
            os.path.join('examples/examples/relighting/02dc3e2cf9541a7d7ebff79cbf1fb0d95b4911e8_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "sci-fi RGB glowing, cyberpunk.",
    ), 
    dict(
        name='golden time', 
        images=[
            os.path.join('examples/examples/relighting/02dc6ca122863a582306a4f146b7bccb721a49e0.jpg'),
            os.path.join('examples/examples/relighting/02dc6ca122863a582306a4f146b7bccb721a49e0_Left.jpg'),
            os.path.join('examples/examples/relighting/02dc4ebfd90dc80dbc0f4174679ff3828605ec9c.jpg'),
            os.path.join('examples/examples/relighting/02dc4ebfd90dc80dbc0f4174679ff3828605ec9c_Left.jpg'),
            os.path.join('examples/examples/relighting/02dca7ccfad757fd596d33563d06b3ab7836d5af.jpg'),
            os.path.join('examples/examples/relighting/02dca7ccfad757fd596d33563d06b3ab7836d5af_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "golden time.",
    ), 
    dict(
        name='shadow from window', 
        images=[
            os.path.join('examples/examples/relighting/02dd1c7c81e77e22ddba378a121fc371afcc9657.jpg'),
            os.path.join('examples/examples/relighting/02dd1c7c81e77e22ddba378a121fc371afcc9657_Left.jpg'),
            os.path.join('examples/examples/relighting/02dcc762ae13127e3975ec043f13342490f61cf8.jpg'),
            os.path.join('examples/examples/relighting/02dcc762ae13127e3975ec043f13342490f61cf8_Left.jpg'),
            os.path.join('examples/examples/relighting/02dd0f49dceaf611e0173319e26b4e6e1b7a6dd4.jpg'),
            os.path.join('examples/examples/relighting/02dd0f49dceaf611e0173319e26b4e6e1b7a6dd4_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "shadow from window.",
    ), 
    dict(
        name='soft studio lighting', 
        images=[
            os.path.join('examples/examples/relighting/02dd6f77ccab6d63e7f2d7795f5d03180b46621c.jpg'),
            os.path.join('examples/examples/relighting/02dd6f77ccab6d63e7f2d7795f5d03180b46621c_Left.jpg'),
            os.path.join('examples/examples/relighting/02dd6a91d0d1d17a9f06e999654b541b555da242.jpg'),
            os.path.join('examples/examples/relighting/02dd6a91d0d1d17a9f06e999654b541b555da242_Left.jpg'),
        ], 
        grid_h=2,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "soft studio lighting.",
    ), 
    dict(
        name='evil, gothic, Yharnam', 
        images=[
            os.path.join('examples/examples/relighting/02aee2a8df8f6e6f16ca4ec278203543656cecf1.jpg'),
            os.path.join('examples/examples/relighting/02aee2a8df8f6e6f16ca4ec278203543656cecf1_Left.jpg'),
            os.path.join('examples/examples/relighting/02af9925c86c22b379e4e6d4f2762d66966ee281.jpg'),
            os.path.join('examples/examples/relighting/02af9925c86c22b379e4e6d4f2762d66966ee281_Left.jpg'),
            os.path.join('examples/examples/relighting/02dd79a669a4522f1d5631d75c14243f927848b8.jpg'),
            os.path.join('examples/examples/relighting/02dd79a669a4522f1d5631d75c14243f927848b8_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "evil, gothic, Yharnam.",
    ),
    dict(
        name='neon, Wong Kar-wai, warm', 
        images=[
            os.path.join('examples/examples/relighting/02af99b6765a77a8f2ac87aa42d2f2453dcd590f.jpg'),
            os.path.join('examples/examples/relighting/02af99b6765a77a8f2ac87aa42d2f2453dcd590f_Left.jpg'),
            os.path.join('examples/examples/relighting/02b02e2916bf2eb3608f5a806dc3b7ecbed3b649.jpg'),
            os.path.join('examples/examples/relighting/02b02e2916bf2eb3608f5a806dc3b7ecbed3b649_Left.jpg'),
            os.path.join('examples/examples/relighting/02af9fa52ff41e64de8e3212683c9ed43bd91010.jpg'),
            os.path.join('examples/examples/relighting/02af9fa52ff41e64de8e3212683c9ed43bd91010_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "neon, Wong Kar-wai, warm.",
    ),
    dict(
        name='home atmosphere, cozy bedroom illumination', 
        images=[
            os.path.join('examples/examples/relighting/02db22466eb3bc19d6a10195e1b48fff696c1582.jpg'),
            os.path.join('examples/examples/relighting/02db22466eb3bc19d6a10195e1b48fff696c1582_Left.jpg'),
            os.path.join('examples/examples/relighting/02c3760bf08f00d9e2163248e2864f5e1a70d709.jpg'),
            os.path.join('examples/examples/relighting/02c3760bf08f00d9e2163248e2864f5e1a70d709_Left.jpg'),
            os.path.join('examples/examples/relighting/02af06c41208b31248e94da13166a675c862b003.jpg'),
            os.path.join('examples/examples/relighting/02af06c41208b31248e94da13166a675c862b003_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction, 
        content_prompt=content_instruction + "home atmosphere, cozy bedroom illumination.",
    ), 
    dict(
        name='warm atmosphere, at home, bedroom', 
        images=[
            os.path.join('examples/examples/relighting/02c39e8e82f4be91d24252c8bfbfdef033ec8a32.jpg'),
            os.path.join('examples/examples/relighting/02c39e8e82f4be91d24252c8bfbfdef033ec8a32_Left.jpg'),
            os.path.join('examples/examples/relighting/02c5200cac1d0f19256232a09708ac47f6ddfab3.jpg'),
            os.path.join('examples/examples/relighting/02c5200cac1d0f19256232a09708ac47f6ddfab3_Left.jpg'),
            os.path.join('examples/examples/relighting/02dd6f77ccab6d63e7f2d7795f5d03180b46621c.jpg'),
            os.path.join('examples/examples/relighting/02dd6f77ccab6d63e7f2d7795f5d03180b46621c_Left_2.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "warm atmosphere, at home, bedroom.",
    ),
    dict(
        name='natural lighting', 
        images=[
            os.path.join('examples/examples/relighting/02dafead46f6d59172d8df216c1f5ad11f9899b5.jpg'),
            os.path.join('examples/examples/relighting/02dafead46f6d59172d8df216c1f5ad11f9899b5_Left.jpg'),
            os.path.join('examples/examples/relighting/02dc42496c4ffdb2a8e101ed82943b26fc2d9d24.jpg'),
            os.path.join('examples/examples/relighting/02dc42496c4ffdb2a8e101ed82943b26fc2d9d24_Left.jpg'),
        ], 
        grid_h=2,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "natural lighting.",
    ),
    dict(
        name='magic lit', 
        images=[
            os.path.join('examples/examples/relighting/02dd9913f85a62d9c1587b00f610cc753ebad649.jpg'),
            os.path.join('examples/examples/relighting/02dd9913f85a62d9c1587b00f610cc753ebad649_Left.jpg'),
            os.path.join('examples/examples/relighting/02afbcf084a1e35bda34c26d2271d56b6a1c621e.jpg'),
            os.path.join('examples/examples/relighting/02afbcf084a1e35bda34c26d2271d56b6a1c621e_Left.jpg'),
        ], 
        grid_h=2,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "magic lit.",
    ),
    dict(
        name='sunshine from window', 
        images=[
            os.path.join('examples/examples/relighting/02c53f12ec3d4a9a16d9b0ca3f7773ad2222100c.jpg'),
            os.path.join('examples/examples/relighting/02c53f12ec3d4a9a16d9b0ca3f7773ad2222100c_Left.jpg'),
            os.path.join('examples/examples/relighting/02c6c0f92a672110ff86bd12f4aa0d0083c9cf6b.jpg'),
            os.path.join('examples/examples/relighting/02c6c0f92a672110ff86bd12f4aa0d0083c9cf6b_Left.jpg'),
            os.path.join('examples/examples/relighting/02c5cc03d46ce15494caaf3d65a2b2c7e09089f2.jpg'),
            os.path.join('examples/examples/relighting/02c5cc03d46ce15494caaf3d65a2b2c7e09089f2_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "sunshine from window.",
    ),
    dict(
        name='neon light, city', 
        images=[
            os.path.join('examples/examples/relighting/02c7df6c0decd3d542e25089a0af6afe1e070b6a.jpg'),
            os.path.join('examples/examples/relighting/02c7df6c0decd3d542e25089a0af6afe1e070b6a_Left.jpg'),
            os.path.join('examples/examples/relighting/02c77b643fbdaec82912634655426553f3d7a537.jpg'),
            os.path.join('examples/examples/relighting/02c77b643fbdaec82912634655426553f3d7a537_Left.jpg'),
            os.path.join('examples/examples/relighting/02c73157a981e0ee669ca8125018efbdda1e1483.jpg'),
            os.path.join('examples/examples/relighting/02c73157a981e0ee669ca8125018efbdda1e1483_Left.jpg'),
        ], 
        grid_h=3,
        grid_w=2,
        task_prompt=task_instruction,
        content_prompt=content_instruction + "neon light, city.",
    ),
]
relighting_text = [[x['name']] for x in relighting]


def process_relighting_tasks(x):
    for task in relighting:
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

            upsampling_noise = 0.6
            steps = 50
            outputs = [mask, grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + rets
            break

    return outputs
