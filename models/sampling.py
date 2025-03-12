import math
import random
from typing import Callable

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor
import torch.nn.functional as F

from .model import Flux
from .modules.autoencoder import AutoEncoder
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import CannyImageEncoder, DepthImageEncoder, ReduxImageEncoder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

def prepare_modified(t5: HFEmbedder, clip: HFEmbedder, img: list[list[torch.Tensor]], prompt: str | list[str], proportion_empty_prompts: float = 0.1, is_train: bool = True, text_emb: list[dict[str, Tensor]] = None) -> dict[str, Tensor]:
    assert isinstance(img, list) and all([isinstance(img[i], list) for i in range(len(img))])
    bs = len(img)
    if isinstance(img[0], torch.Tensor):
        max_len = max([i.shape[-2] * i.shape[-1] for i in img]) // 4
        img_mask = torch.zeros(bs, max_len, device=img[0].device, dtype=torch.int32)
    else:
        max_len = max([sum([i.shape[-2] * i.shape[-1] for i in sub_image]) for sub_image in img]) // 4
        img_mask = torch.zeros(bs, max_len, device=img[0][0].device, dtype=torch.int32)
    # pad img to same length for batch processing
    padded_img = []
    padded_img_ids = []
    for i in range(bs):
        img_i = img[i]
        flat_img_list = []
        flat_img_ids_list = []
        for j in range(len(img_i)):
            img_i_j = img_i[j].squeeze(0)
            c, h, w = img_i_j.shape
            img_ids = torch.zeros(h // 2, w // 2, 3)
            img_ids[..., 0] = j + 1
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            
            flat_img_ids = rearrange(img_ids, "h w c -> (h w) c")
            flat_img = rearrange(img_i_j, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            flat_img_list.append(flat_img)
            flat_img_ids_list.append(flat_img_ids)

        flat_img = torch.cat(flat_img_list, dim=0)
        flat_img_ids = torch.cat(flat_img_ids_list, dim=0)
        padded_img.append(F.pad(flat_img, (0, 0, 0, max_len - flat_img.shape[0])))
        padded_img_ids.append(F.pad(flat_img_ids, (0, 0, 0, max_len - flat_img_ids.shape[0])))
        img_mask[i, :flat_img.shape[0]] = 1

    img = torch.stack(padded_img, dim=0)
    img_ids = torch.stack(padded_img_ids, dim=0)
        
    if isinstance(prompt, str):
        prompt = [prompt]
        
    bs = len(prompt)
    drop_mask = []
    for idx in range(bs):
        if random.random() < proportion_empty_prompts:
            prompt[idx] = ""
        elif isinstance(prompt[idx], (list)):
            prompt[idx] = random.choice(prompt[idx]) if is_train else prompt[idx][0]
        if prompt[idx] == "":
            drop_mask.append(0)
        else:
            drop_mask.append(1)
    drop_mask = torch.tensor(drop_mask, device=img_mask.device, dtype=img_mask.dtype)
    
    if t5 is None:
        txt = torch.stack([item["txt"] for item in text_emb], dim=0).to(img.device)
    else:
        txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)
    txt_mask = torch.ones(bs, txt.shape[1], device=txt.device, dtype=torch.int32)

    if clip is None:
        vec = torch.stack([item["vec"] for item in text_emb], dim=0).to(img.device)
    else:
        vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    out_dict = {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
        "img_mask": img_mask.to(img.device),
        "txt_mask": txt_mask.to(txt.device),
        "drop_mask": drop_mask.to(img.device),
    }

    return out_dict


# ############################# Original Prepare Function #############################

def prepare(t5: HFEmbedder, 
                clip: HFEmbedder, 
                img: Tensor, 
                prompt: str | list[str]
                ) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def prepare_control(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    encoder: DepthImageEncoder | CannyImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")

    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    with torch.no_grad():
        img_cond = encoder(img_cond)
        img_cond = ae.encode(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond
    return return_dict


def prepare_fill(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    mask_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image and the mask
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float() / 255.0
    mask = rearrange(mask, "h w -> 1 1 h w")

    with torch.no_grad():
        img_cond = img_cond.to(img.device)
        mask = mask.to(img.device)
        img_cond = img_cond * (1 - mask)
        img_cond = ae.encode(img_cond)
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(
            mask,
            "b (h ph) (w pw) -> b (ph pw) h w",
            ph=8,
            pw=8,
        )
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if mask.shape[0] == 1 and bs > 1:
            mask = repeat(mask, "1 ... -> bs ...", bs=bs)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img_cond = torch.cat((img_cond, mask), dim=-1)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond.to(img.device)
    return return_dict


def prepare_redux(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    encoder: ReduxImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    with torch.no_grad():
        img_cond = encoder(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    txt = torch.cat((txt, img_cond.to(txt)), dim=-2)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens
    img_cond: Tensor | None = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
