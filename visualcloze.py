
import random
from einops import rearrange
from diffusers.models import AutoencoderKL
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from models.sampling import prepare_modified
from models.util import load_clip, load_t5, load_flow_model
from transport import Sampler, create_transport
from util.imgproc import to_rgb_if_rgba


def center_crop(image, target_size):
    width, height = image.size
    new_width, new_height = target_size

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))


def resize_with_aspect_ratio(img, resolution, divisible=16, aspect_ratio=None):
    """Resize image while maintaining aspect ratio, ensuring area is close to resolution**2 and dimensions are divisible by 16
    
    Args:
        img: PIL Image or torch.Tensor (C,H,W)/(B,C,H,W)
        resolution: target resolution
        divisible: ensure output dimensions are divisible by this number
    
    Returns:
        Resized image of the same type as input
    """
    # Check input type and get dimensions
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        if img.dim() == 3:
            c, h, w = img.shape
            batch_dim = False
        else:
            b, c, h, w = img.shape
            batch_dim = True
    else:
        w, h = img.size
        
    # Calculate new dimensions
    if aspect_ratio is None:
        aspect_ratio = w / h
    target_area = resolution * resolution
    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)
    
    # Ensure divisible by divisible
    new_w = max(new_w // divisible, 1) * divisible
    new_h = max(new_h // divisible, 1) * divisible
    
    # Adjust size based on input type
    if is_tensor:
        # Use torch interpolation method
        mode = 'bilinear'
        align_corners = False
        if batch_dim:
            return F.interpolate(img, size=(new_h, new_w), 
                               mode=mode, align_corners=align_corners)
        else:
            return F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                               mode=mode, align_corners=align_corners).squeeze(0)
    else:
        # Use PIL LANCZOS resampling
        return img.resize((new_w, new_h), Image.LANCZOS)


class VisualClozeModel:
    def __init__(
        self, model_path, model_name="flux-dev-fill-lora", max_length=512, lora_rank=256, 
        atol=1e-6, rtol=1e-3, solver='euler', time_shifting_factor=1, 
        resolution=384, precision='bf16'):
        self.atol = atol
        self.rtol = rtol
        self.solver = solver
        self.time_shifting_factor = time_shifting_factor
        self.resolution = resolution
        self.precision = precision
        self.max_length = max_length
        self.lora_rank = lora_rank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]
        
        # Initialize model
        print("Initializing model...")
        self.model = load_flow_model(model_name, device=self.device, lora_rank=self.lora_rank)
        
        # Initialize VAE
        print("Initializing VAE...")
        self.ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=self.dtype).to(self.device)
        self.ae.requires_grad_(False)
        
        # Initialize text encoders
        print("Initializing text encoders...")
        self.t5 = load_t5(self.device, max_length=self.max_length)
        self.clip = load_clip(self.device)
        
        self.model.eval().to(self.device, dtype=self.dtype)
        
        # Load model weights
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt, strict=False)
        del ckpt
        
        # Initialize sampler
        transport = create_transport(
            "Linear",
            "velocity",
            do_shift=True,
        ) 
        self.sampler = Sampler(transport)
        self.sample_fn = self.sampler.sample_ode(
            sampling_method=self.solver,
            num_steps=30,
            atol=self.atol,
            rtol=self.rtol,
            reverse=False,
            do_shift=True,
            time_shifting_factor=self.time_shifting_factor,
        )
        
        # Image transformation
        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
        self.grid_h = None
        self.grid_w = None
        
    def set_grid_size(self, h, w):
        """Set grid size"""
        self.grid_h = h
        self.grid_w = w
    
    def upsampling(self, image, target_size, cfg, upsampling_steps, upsampling_noise, generator, content_prompt):
        content_instruction = [
            "The content of the last image in the final row is: ",
            "The last image of the last row depicts: ",
            "In the final row, the last image shows: ",
            "The last image in the bottom row illustrates: ",
            "The content of the bottom-right image is: ",
            "The final image in the last row portrays: ",
            "The last image of the final row displays: ",
            "In the last row, the final image captures: ",
            "The bottom-right corner image presents: ",
            "The content of the last image in the concluding row is: ",
            "In the last row, ",
            "The editing instruction in the last row is: ", 
        ]
        for c in content_instruction:
            if content_prompt.startswith(c):
                content_prompt = content_prompt.replace(c, '')
        
        if target_size is None:
            aspect_ratio = 1
            target_area = 1024 * 1024
            new_h = int((target_area / aspect_ratio) ** 0.5)
            new_w = int(new_h * aspect_ratio)
            target_size = (new_w, new_h)

        if target_size[0] * target_size[1] > 1024 * 1024:
            aspect_ratio = target_size[0] / target_size[1]
            target_area = 1024 * 1024
            new_h = int((target_area / aspect_ratio) ** 0.5)
            new_w = int(new_h * aspect_ratio)
            target_size = (new_w, new_h)
        
        image = image.resize(((target_size[0] // 16) * 16, (target_size[1] // 16) * 16))
        if upsampling_noise >= 1.0:
            return image

        self.sample_fn = self.sampler.sample_ode(
            sampling_method=self.solver,
            num_steps=upsampling_steps,
            atol=self.atol,
            rtol=self.rtol,
            reverse=False,
            do_shift=False,
            time_shifting_factor=1.0, 
            strength=upsampling_noise
        )

        processed_image = self.image_transform(image)
        processed_image = processed_image.to(self.device, non_blocking=True)
        blank = torch.zeros_like(processed_image, device=self.device, dtype=self.dtype)
        mask = torch.full((1, 1, processed_image.shape[1], processed_image.shape[2]), fill_value=1, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            latent = self.ae.encode(processed_image[None].to(self.ae.dtype)).latent_dist.sample()
            blank = self.ae.encode(blank[None].to(self.ae.dtype)).latent_dist.sample()
            latent = (latent - self.ae.config.shift_factor) * self.ae.config.scaling_factor
            blank = (blank - self.ae.config.shift_factor) * self.ae.config.scaling_factor
            latent_h, latent_w = latent.shape[2:]

            mask = rearrange(mask, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=8, pw=8) 
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            
            latent = latent.to(self.dtype)
            blank = blank.to(self.dtype)
            latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            blank = rearrange(blank, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            
            img_cond = torch.cat((blank, mask), dim=-1)
    
            # Generate noise
            noise = torch.randn([1, 16, latent_h, latent_w], device=self.device, generator=generator).to(self.dtype)
            x = [[noise]]
            
            inp = prepare_modified(t5=self.t5, clip=self.clip, img=x, prompt=[content_prompt], proportion_empty_prompts=0.0)
            inp["img"] = inp["img"] * (1 - upsampling_noise) + latent * upsampling_noise
            model_kwargs = dict(
                txt=inp["txt"], 
                txt_ids=inp["txt_ids"], 
                txt_mask=inp["txt_mask"],
                y=inp["vec"], 
                img_ids=inp["img_ids"], 
                img_mask=inp["img_mask"], 
                cond=img_cond,
                guidance=torch.full((1,), cfg, device=self.device, dtype=self.dtype),
            )
            sample = self.sample_fn(
                inp["img"], self.model.forward, model_kwargs
            )[-1]
            
            sample = sample[:1]
            sample = rearrange(sample, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=latent_h // 2, w=latent_w // 2)
            sample = self.ae.decode(sample / self.ae.config.scaling_factor + self.ae.config.shift_factor)[0]
            sample = (sample + 1.0) / 2.0
            sample.clamp_(0.0, 1.0)
            sample = sample[0]
            
            output_image = to_pil_image(sample.float())
            
            return output_image
    
    def process_images(
        self, images: list[list[Image.Image]], 
        prompts: list[str], 
        seed: int = 0, 
        cfg: int = 30, 
        steps: int = 30, 
        upsampling_steps: int = 10, 
        upsampling_noise: float = 0.4, 
        is_upsampling: bool =True):
        """
        Processes a list of images based on the provided text prompts and settings, with optional upsampling to enhance image resolution or detail.

        Parameters:
            images (list[list[Image.Image]]): A collection of images arranged in a grid layout, where each row represents an in-context example or the current query. 
            The current query should be placed in the last row. The target image may be None in the input, while all other images should be of the PIL Image type (Image.Image).
            
            prompts (list[str]): A list containing three prompts: the layout prompt, task prompt, and content prompt, respectively.
            
            seed (int): A fixed integer seed to ensure reproducibility of random elements during processing.
            
            cfg (int): The strength of Classifier-Free Diffusion Guidance, which controls the degree of influence over the generated results.
            
            steps (int): The number of sampling steps to be performed during processing.
            
            upsampling_steps (int): The number of denoising steps to apply when performing upsampling.
            
            upsampling_noise (float): The noise level used as a starting point when upsampling with SDEdit. A higher value reduces noise, and setting it to 1 disables SDEdit, causing the PIL resize function to be used instead.
            
            is_upsampling (bool, optional): A flag indicating whether upsampling should be applied using SDEdit.

        Returns:
            Processed images resulting from the algorithm, with optional upsampling applied based on the `is_upsampling` flag.
        """
        
        if seed == 0:
            seed = random.randint(0, 2 ** 32 - 1)
        
        self.sample_fn = self.sampler.sample_ode(
            sampling_method=self.solver,
            num_steps=steps,
            atol=self.atol,
            rtol=self.rtol,
            reverse=False,
            do_shift=True,
            time_shifting_factor=self.time_shifting_factor,
        )

        # Use class grid size
        grid_h, grid_w = self.grid_h, self.grid_w
        
        # Ensure all images are RGB mode or None
        for i in range(0, grid_h):
            images[i] = [img.convert("RGB") if img is not None else None for img in images[i]]
        
        # Adjust all image sizes
        resolution = self.resolution
        processed_images = []
        mask_position = []
        target_size = None
        upsampling_size = None
        
        for i in range(grid_h):
            # Find the size of the first non-empty image in this row
            reference_size = None
            for j in range(0, grid_w):
                if images[i][j] is not None:
                    if i == grid_h - 1 and upsampling_size is None:
                        upsampling_size = images[i][j].size

                    resized = resize_with_aspect_ratio(images[i][j], resolution, aspect_ratio=None)
                    reference_size = resized.size
                    if i == grid_h - 1 and target_size is None:
                        target_size = reference_size
                    break
            
            # Process all images in this row
            for j in range(0, grid_w):
                if images[i][j] is not None:
                    target = resize_with_aspect_ratio(images[i][j], resolution, aspect_ratio=None)
                    if target.width <= target.height:
                        target = target.resize((reference_size[0], int(reference_size[0] / target.width * target.height)))
                        target = center_crop(target, reference_size)
                    elif target.width > target.height:
                        target = target.resize((int(reference_size[1] / target.height * target.width), reference_size[1]))
                        target = center_crop(target, reference_size)
                    
                    processed_images.append(target)
                    if i == grid_h - 1:
                        mask_position.append(0)
                else:
                    # If this row has a reference size, use it; otherwise use default size
                    if reference_size:
                        blank = Image.new('RGB', reference_size, (0, 0, 0))
                    else:
                        blank = Image.new('RGB', (resolution, resolution), (0, 0, 0))
                    processed_images.append(blank)
                    if i == grid_h - 1:
                        mask_position.append(1)
                    else:
                        raise ValueError('Please provide each image in the in-context example.')
            
        # return processed_images
        
        if len(mask_position) > 1 and sum(mask_position) > 1:
            if target_size is None:
                new_w = 384
            else:
                new_w = target_size[0]
            for i in range(len(processed_images)):
                if processed_images[i] is not None:
                    new_h = int(processed_images[i].height * (new_w / processed_images[i].width))
                    new_w = int(new_w / 16) * 16
                    new_h = int(new_h / 16) * 16
                    processed_images[i] = processed_images[i].resize((new_w, new_h))
                
        # Build grid image and mask
        with torch.autocast("cuda", self.dtype):
            grid_image = []
            fill_mask = []
            for i in range(grid_h):
                row_images = [self.image_transform(img) for img in processed_images[i * grid_w: (i + 1) * grid_w]]
                if i == grid_h - 1:
                    row_masks = [torch.full((1, 1, row_images[0].shape[1], row_images[0].shape[2]), fill_value=m, device=self.device) for m in mask_position]
                else:
                    row_masks = [torch.full((1, 1, row_images[0].shape[1], row_images[0].shape[2]), fill_value=0, device=self.device) for m in mask_position]

                grid_image.append(torch.cat(row_images, dim=2).to(self.device, non_blocking=True))
                fill_mask.append(torch.cat(row_masks, dim=3))
            # Encode condition image
            with torch.no_grad():
                fill_cond = [self.ae.encode(img[None].to(self.ae.dtype)).latent_dist.sample()[0] for img in grid_image]
                fill_cond = [(img - self.ae.config.shift_factor) * self.ae.config.scaling_factor for img in fill_cond]
                
                # Rearrange mask
                fill_mask = [rearrange(mask, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=8, pw=8) for mask in fill_mask]
                fill_mask = [rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for mask in fill_mask]
            
            fill_cond = [img.to(self.dtype) for img in fill_cond]
            fill_cond = [rearrange(img.unsqueeze(0), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for img in fill_cond]
            
            fill_cond =  torch.cat(fill_cond, dim=1)
            fill_mask =  torch.cat(fill_mask, dim=1)
            img_cond = torch.cat((fill_cond, fill_mask), dim=-1)
        
            # Generate sample
            noise = []
            sliced_subimage = []
            rng = torch.Generator(device=self.device).manual_seed(int(seed))
            for sub_img in grid_image:
                h, w = sub_img.shape[-2:]
                sliced_subimage.append((h, w))
                latent_w, latent_h = w // 8, h // 8
                noise.append(torch.randn([1, 16, latent_h, latent_w], device=self.device, generator=rng).to(self.dtype))
            x = [noise]
            
            with torch.no_grad():
                inp = prepare_modified(t5=self.t5, clip=self.clip, img=x, prompt=[' '.join(prompts)], proportion_empty_prompts=0.0)
                
                model_kwargs = dict(
                    txt=inp["txt"], 
                    txt_ids=inp["txt_ids"], 
                    txt_mask=inp["txt_mask"],
                    y=inp["vec"], 
                    img_ids=inp["img_ids"], 
                    img_mask=inp["img_mask"], 
                    cond=img_cond,
                    guidance=torch.full((1,), cfg, device=self.device, dtype=self.dtype),
                )
                samples = self.sample_fn(
                    inp["img"], self.model.forward, model_kwargs
                )[-1]

            # Get query row
            samples = samples[:1]
            row_samples = []
            start = 0
            with torch.no_grad():
                for size in sliced_subimage:
                    end = start + (size[0] * size[1] // 256)
                    latent_h = size[0] // 8
                    latent_w = size[1] // 8
                    row_sample = samples[:, start:end, :]
                    row_sample = rearrange(row_sample, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=latent_h//2, w=latent_w//2)
                    row_sample = self.ae.decode(row_sample / self.ae.config.scaling_factor + self.ae.config.shift_factor)[0]
                    row_sample = (row_sample + 1.0) / 2.0
                    row_sample.clamp_(0.0, 1.0)
                    row_samples.append(row_sample[0])
                    start = end

            # Convert all samples to PIL images
            output_images = []
            for row_sample in row_samples:
                output_image = to_pil_image(row_sample.float())
                output_images.append(output_image)
            
            torch.cuda.empty_cache()
            
            ret = []
            ret_w = output_images[-1].width
            ret_h = output_images[-1].height
            
            row_start = (grid_h - 1) * grid_w
            row_end = grid_h * grid_w
            for i in range(row_start, row_end):
                # when the image is masked, then output it
                if mask_position[i - row_start] and is_upsampling:
                    cropped = output_images[-1].crop(((i - row_start) * ret_w // self.grid_w, 0, ((i - row_start) + 1) * ret_w // self.grid_w, ret_h))
                    upsampled = self.upsampling(
                        cropped, 
                        upsampling_size, 
                        cfg, 
                        upsampling_steps=upsampling_steps, 
                        upsampling_noise=upsampling_noise, 
                        generator=rng, 
                        content_prompt=prompts[2])
                    ret.append(upsampled)
                elif mask_position[i - row_start]:
                    cropped = output_images[-1].crop(((i - row_start) * ret_w // self.grid_w, 0, ((i - row_start) + 1) * ret_w // self.grid_w, ret_h))
                    ret.append(cropped)
            
            return ret