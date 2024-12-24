import argparse
import functools
import json
import math
import os
import random
import socket
import time

from einops import rearrange, repeat
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from safetensors.torch import load_file as load_sft
from tqdm import tqdm

from flux.model import Flux, FluxParams
from flux.sampling import prepare_modified
from flux.util import configs, load_clip, load_t5, load_flow_model
from transport import Sampler, create_transport
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop


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

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    print("Init model")
    model = load_flow_model(args.model_name, device=device_str, lora_rank=args.lora_rank)
    # params = configs[args.model_name].params
    # with torch.device(device_str):
    #     if "lora" in args.model_name:
    #         model = FluxLoraWrapper(params).to(dtype)
    #     else:
    #         model = Flux(params).to(dtype)
    # for name, param in model.named_parameters():
    #     print(name)

    print("Init vae")
    ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device_str)
    ae.requires_grad_(False)
    
    print("Init text encoder")
    t5 = load_t5(device_str, max_length=args.max_length)
    clip = load_clip(device_str)
        
    model.eval().to(device_str, dtype=dtype)

    if not args.debug:
        # assert train_args.model_parallel_size == args.num_gpus
        if (args.ckpt).endswith(".safetensors"):
            ckpt = load_sft(args.ckpt, device=device_str)
            missing, unexpected = model.load_state_dict(ckpt, strict=False, assign=True)
        else:
            if args.ema:
                print("Loading ema model.")
            ckpt = torch.load(
                os.path.join(
                    args.ckpt,
                    f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.pth",
                )
            )
            model.load_state_dict(ckpt, strict=True)
        del ckpt
        
    # begin sampler
    transport = create_transport(
        "Linear",
        "velocity",
        do_shift=args.do_shift,
    ) 
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=args.solver,
        num_steps=args.num_sampling_steps,
        atol=args.atol,
        rtol=args.rtol,
        reverse=args.reverse,
        do_shift=args.do_shift,
        time_shifting_factor=args.time_shifting_factor,
    )
    # end sampler

    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, "images"), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, "data.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.loads(f.read())
        collected_id = []
        for i in info:
            collected_id.append(f'{i["idx"]}_{i["high_res"]}')
    else:
        info = []
        collected_id = []

    with open(args.caption_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    total = len(info)
    with torch.autocast("cuda", dtype):
        for idx, item in tqdm(enumerate(data)):
            caps_list = [item["prompt"]]
            
            for high_res in args.resolution:
                                
                if int(args.seed) != 0:
                    torch.random.manual_seed(int(args.seed))

                sample_id = f'{idx}_{high_res}'
                if sample_id in collected_id:
                    continue

                res_cat, resolution = high_res.split(":")
                res_cat = int(res_cat)
                do_extrapolation = res_cat > 1024

                n = len(caps_list)
                w, h = resolution.split("x")
                h, w = int(h), int(w)
                latent_w, latent_h = w // 8, h // 8
                x = torch.randn([1, 16, latent_h, latent_w], device=device_str).to(dtype)
                with torch.no_grad():
                    if args.do_classifier_free_guidance:
                        x = x.repeat(n * 2, 1, 1, 1)
                        inp = prepare_modified(t5=t5, clip=clip, img=x, prompt=[caps_list] + [""] * n, proportion_empty_prompts=0.0)
                    else:
                        inp = prepare_modified(t5=t5, clip=clip, img=x, prompt=caps_list, proportion_empty_prompts=0.0)

                if args.do_classifier_free_guidance:
                    model_kwargs = dict(
                        txt=inp["txt"], 
                        txt_ids=inp["txt_ids"], 
                        txt_mask=inp["txt_mask"],
                        y=inp["vec"], 
                        img_ids=inp["img_ids"], 
                        img_mask=inp["img_mask"], 
                        guidance=torch.full((x.shape[0],), 1, device=x.device, dtype=x.dtype), 
                        cfg_scale=args.guidance_scale,
                    )
                    samples = sample_fn(
                        inp["img"], model.forward_with_cfg, model_kwargs
                    )[-1]
                else:
                    model_kwargs = dict(
                        txt=inp["txt"], 
                        txt_ids=inp["txt_ids"], 
                        txt_mask=inp["txt_mask"],
                        y=inp["vec"], 
                        img_ids=inp["img_ids"], 
                        img_mask=inp["img_mask"], 
                        guidance=torch.full((x.shape[0],), args.guidance_scale, device=x.device, dtype=x.dtype), 
                    )
                    samples = sample_fn(
                        inp["img"], model.forward, model_kwargs
                    )[-1]
                samples = samples[:n]
                samples = rearrange(samples, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=latent_h//2, w=latent_w//2)
                samples = ae.decode(samples / ae.config.scaling_factor + ae.config.shift_factor)[0]
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)

                # Save samples to disk as individual .png files
                for i, (sample, cap) in enumerate(zip(samples, caps_list)):
                    
                    img = to_pil_image(sample.float())
                    save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.jpg"
                    img.save(save_path, format='JPEG', quality=95)
                    
                    info.append(
                        {
                            "idx": idx,
                            "caption": cap,
                            "image_url": f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png",
                            "high_res": high_res,
                            "solver": args.solver,
                            "num_sampling_steps": args.num_sampling_steps,
                        }
                    )

                with open(info_path, "w") as f:
                    f.write(json.dumps(info))

                total += len(samples)
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
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="flux-dev")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--do_classifier_free_guidance", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    args = parser.parse_known_args()[0]
    
    args.low_res_list = [int(res) for res in args.low_res_list.split(",")]
    args.high_res_list = [int(res) for res in args.high_res_list.split(",")]
    
    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)