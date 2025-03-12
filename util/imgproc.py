import random

from PIL import Image
import PIL.Image
import numpy as np


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def center_crop(pil_image, crop_size):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def pad(pil_image, pad_size):
    while pil_image.size[0] >= 2 * pad_size[0] and pil_image.size[1] >= 2 * pad_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = min(pad_size[0] / pil_image.size[0], pad_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    new_image = Image.new('RGB', pad_size, (255, 255, 255))
    new_image.paste(pil_image, (0, 0))
    return new_image


def var_center_crop(pil_image, crop_size_list, random_top_k=4):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def var_pad(pil_image, pad_size_list, random_top_k=4):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in pad_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, pad_size_list)), reverse=True)[:random_top_k]
    )[1]
    return pad(pil_image, crop_size)


def match_size(w, h, crop_size_list, random_top_k=4):
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return crop_size


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0, step_size=1):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, step_size
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + step_size) * wp <= num_patches:
            hp += step_size
        else:
            wp -= step_size
    return crop_size_list

def to_rgb_if_rgba(img: Image.Image):
    if img.mode.upper() == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        return rgb_img
    else:
        return img
