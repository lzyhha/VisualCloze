import argparse
import json
import os
import random
import socket

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image

from data.prefix_instruction import test_task_dicts
from data.data_reader import T2IItemProcessor
from data.data_utils import check_item_graph200k
from visualcloze import VisualClozeModel


def concat_images_grid(images):
    row_images = []
    for row in images:
        row_widths, row_heights = zip(*(img.size for img in row))
        total_width = sum(row_widths)
        max_height = max(row_heights)
        
        new_row_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        x_offset = 0
        for img in row:
            new_row_img.paste(img, (x_offset, 0))
            x_offset += img.width
            
        row_images.append(new_row_img)

    col_widths, col_heights = zip(*(img.size for img in row_images))
    total_height = sum(col_heights)
    max_width = max(col_widths)
    
    final_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    
    y_offset = 0
    for img in row_images:
        final_img.paste(img, (0, y_offset))
        y_offset += img.height
    
    return final_img


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_process_group("nccl")
    
    print("Init model")
    visualcloze = VisualClozeModel(
        model_path=args.model_path, 
        resolution=args.resolution, 
        lora_rank=args.lora_rank
    )

    item_processor = T2IItemProcessor(None, resolution=args.resolution)

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
            collected_id.append(f'{i["idx"]}')
    else:
        info = []
        collected_id = []

    data = []
    # Loading data for testing
    with open(args.data_path, "r", encoding="utf-8") as file:
        if args.data_path.endswith('.jsonl'):
            for line in file:
                data.append(json.loads(line))
        else:
            data = json.load(file)

    for idx, item in tqdm(enumerate(data[::1])):

        for context_num in [1, 2, 3]:   
            for task_type in test_task_dicts:
                for image_type_list in task_type["image_list"]:
                    if not check_item_graph200k(item, image_type_list):
                        continue
                    task_name = "_".join(image_type_list) 

                    context_ids = []
                    context_items = []
                    
                    while len(context_items) < context_num - 1:
                        next_idx = random.randint(0, len(data) - 1)
                        if next_idx == idx or next_idx in context_ids:
                            continue
                        if not check_item_graph200k(data[next_idx], image_type_list):
                            continue
                        context_ids.append(next_idx)
                        context_items.append(data[next_idx])
                            
                    all_items = context_items + [item]
                    _, images, prompts, _, grid_shape = item_processor.graph200k_process_item(all_items, image_type_list, context_num, training_mode=False)
                    grid_w, grid_h = grid_shape
                    if f'{idx}' in collected_id:
                        continue
                    
                    images[-1][-1] = None
                    visualcloze.set_grid_size(grid_h, grid_w)
                    ret = visualcloze.process_images(
                        images, prompts, 
                        seed=args.seed, 
                        cfg=args.guidance_scale, 
                        steps=args.num_sampling_steps, 
                        upsampling_noise=None, 
                        upsampling_steps=None, 
                        is_upsampling=False
                    )[-1]
                    images[-1][-1] = ret
                    
                    image = concat_images_grid(images)
                    save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{idx}_{context_num}_{task_name}.jpg"
                    image.save(save_path, format='JPEG', quality=95)
                                                
                    info.append(
                        {
                            "idx": idx,
                            "caption": prompts,
                            "image_url": f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{idx}_{context_num}_{task_name}.jpg",
                            "seed": args.seed, 
                            "guidance_scale": args.guidance_scale, 
                            "steps": args.num_sampling_steps, 
                            "solver": args.solver,
                            "num_sampling_steps": args.num_sampling_steps,
                        }
                    )

                    with open(info_path, "w") as f:
                        f.write(json.dumps(info))

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
    parser.add_argument("--model_path", type=str, required=True)
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
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=384,
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
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--do_classifier_free_guidance", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    args = parser.parse_known_args()[0]
    
    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)