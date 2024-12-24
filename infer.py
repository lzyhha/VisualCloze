import argparse
import functools
import json
import math
import os
import random
import socket
import time
import h5py
from collections import defaultdict

from einops import rearrange, repeat
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from safetensors.torch import load_file as load_sft
from tqdm import tqdm

from flux.model import Flux, FluxParams
from flux.sampling import prepare
from flux.util import configs, load_clip, load_t5, load_flow_model
from transport import Sampler, create_transport
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop
from train import T2IItemProcessor, MyDataset, dataloader_collate_fn


def none_or_str(value):
    if value == "None":
        return None
    return value


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device_str = f"cuda:{rank}"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    print("Init model")
    model = load_flow_model("flux-dev", device=device_str, dtype=dtype)

    print("Init vae")
    ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device_str)
    ae.requires_grad_(False)
    
    print("Init text encoder")
    t5 = load_t5(device_str, max_length=args.max_length)
    clip = load_clip(device_str)
        
    model.eval().to(device_str, dtype=dtype)

    # if args.debug == False:
    #     # assert train_args.model_parallel_size == args.num_gpus
    #     if (args.ckpt).endswith(".safetensors"):
    #         ckpt = load_sft(args.ckpt, device=device_str)
    #         missing, unexpected = model.load_state_dict(ckpt, strict=False, assign=True)
    #     else:
    #         if args.ema:
    #             print("Loading ema model.")
    #         ckpt = torch.load(
    #             os.path.join(
    #                 args.ckpt,
    #                 f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.pth",
    #             )
    #         )
    #         model.load_state_dict(ckpt, strict=True)
    #     del ckpt
        
    transport = create_transport(
        "Linear",
        "velocity",
        do_shift=args.do_shift,
    ) 

    # Setup data:
    print(f"Creating data")
    data_collection = {}
    micro_bsz = args.batch_size
    patch_size = 16
    print(f"patch size: {patch_size}")
    for train_res in args.resolution:
        train_res, _ = train_res.split(":")
        train_res = int(train_res)
        max_num_patches = round((train_res / patch_size) ** 2)
        # crop_size_list = generate_crop_size_list(max_num_patches, patch_size, step_size=1)
        crop_size_list = [(1024, 1024)]
        print("List of crop sizes:")
        for i in range(0, len(crop_size_list), 6):
            print(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i : i + 6]]))
        image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
                transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        dataset = MyDataset(
            args.data_path,
            train_res=None,
            item_processor=T2IItemProcessor(image_transform),
            cache_on_disk=args.cache_data_on_disk,
        )
        print(f"Dataset for {train_res} contains {len(dataset):,} images ({args.data_path})")
        loader = DataLoader(
            dataset,
            batch_size=micro_bsz,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataloader_collate_fn,
        )

        data_collection[train_res] = {
            "loader": loader,
            "loader_iter": iter(loader),
            "transport": transport,
        }
        
    sample_folder_dir = args.image_save_path
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, "data"), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, "data.json")
    info = []

    with open(args.caption_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    for step in range(args.max_steps):
        for high_res in args.resolution:
            res_cat, resolution = high_res.split(":")
            res_cat = int(res_cat)
                
            data_pack = data_collection[res_cat]
            x, caps, text_emb = next(data_pack["loader_iter"])
            x = [img.to(device_str, non_blocking=True) for img in x]
            bsz = len(x)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                if res_cat > 3072:
                    x = [(ae.tiled_encode(img[None].to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in x]
                else:
                    x = [(ae.encode(img[None].to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in x]
                    
            with torch.no_grad():
                inp = prepare(t5=t5, clip=clip, img=x, prompt=caps, proportion_empty_prompts=0, text_emb=text_emb)
            
            # random sample timesteps
            timesteps = torch.rand(len(x), device=inp["img"].device).to(dtype)
            x0 = [torch.randn_like(img) for img in inp["img"]]
            xt = [(1 - timesteps[i]) * x0[i] + timesteps[i] * inp["img"][i] for i in range(len(x))]
            xt = torch.stack(xt, dim=0).to(dtype)
            
            for guidance in args.txt_cfg_list:
                model_kwargs = dict(
                    txt=inp["txt"], 
                    txt_ids=inp["txt_ids"], 
                    txt_mask=inp["txt_mask"],
                    y=inp["vec"], 
                    img_ids=inp["img_ids"], 
                    img_mask=inp["img_mask"], 
                    guidance=torch.full((inp["img"].shape[0],), guidance, device=inp["img"].device, dtype=inp["img"].dtype), 
                    timesteps=timesteps,
                    xt=xt,
                )
                
                data_dict = data_pack["transport"].get_velocity(model, inp["img"], model_kwargs)
                samples = data_dict["velocity"]
                timesteps = data_dict["timesteps"]

                # Save samples to disk as individual .png files
                for i, (sample, timestep, cap) in enumerate(zip(samples, timesteps, caps)):
                    timestep = timestep.item()
                    save_path = f"{args.image_save_path}/data/{step}_{guidance:.2f}_{timestep:.2f}_{resolution}.h5"
                    with h5py.File(save_path, "w") as f:
                        f.create_dataset("samples", data=sample.float().cpu().numpy())
                    info.append(
                        { 
                            "idx": step,
                            "caption": cap,
                            "h5_url": f"{args.image_save_path}/data/{step}_{guidance:.2f}_{timestep:.2f}_{resolution}.h5",
                            "resolution": res_cat,
                            "timestep": timestep,
                            "guidance": guidance,
                        }
                    )
                with open(info_path, "w") as f:
                    f.write(json.dumps(info))

                dist.barrier()

    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="flux-dev")
    parser.add_argument("--text_encoder", type=str, nargs='+', default=['t5', 'clip'], help="List of text encoders to use (e.g., t5, clip, gemma)")
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    # parser.set_defaults(ema=True)
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
        "(sample{_ema}.png in the model checkpoint directory).",
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="prompts.txt",
    )
    parser.add_argument(
        "--low_res_list",
        type=str,
        default="256,512,1024",
        help="Comma-separated list of low resolution for sampling."
    )
    parser.add_argument(
        "--high_res_list",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated list of high resolution for sampling."
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="",
        nargs="+",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
    )
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="Time-aware",
    )
    parser.add_argument(
        "--scaling_watershed",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    parser.add_argument("--do_shift", default=True)
    parser.add_argument("--attn_token_select", action="store_true")
    parser.add_argument("--mlp_token_select", action="store_true")
    parser.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    parser.add_argument("--use_flash_attn", type=bool, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512, help="Max length for T5.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--txt_cfg_list", type=str, default="4.0")
    args = parser.parse_known_args()[0]
    
    args.txt_cfg_list = [float(cfg) for cfg in args.txt_cfg_list.split(",")]
    args.low_res_list = [int(res) for res in args.low_res_list.split(",")]
    args.high_res_list = [int(res) for res in args.high_res_list.split(",")]
    
    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)