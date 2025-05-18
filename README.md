<p align="center">
  <img src="https://github.com/lzyhha/VisualCloze/blob/main/figures/visualcloze.png" height=100>

</p>

<div align="center">
<h1> VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning </h1>

</div>

<div align="center">

[[Paper](https://arxiv.org/abs/2504.07960)] &emsp; [[Online Demo](https://huggingface.co/spaces/VisualCloze/VisualCloze)] &emsp; [[Project Page](https://visualcloze.github.io/)] 
<br>[[🤗 Full Model Card (<strong><span style="color:hotpink">Diffusers</span></strong>)](https://huggingface.co/VisualCloze/VisualClozePipeline-384)] &emsp; [[🤗 LoRA Model Card (<strong><span style="color:hotpink">Diffusers</span></strong>)](https://huggingface.co/VisualCloze/VisualClozePipeline-LoRA-384)] <br> [[🤗 Dataset Card](https://huggingface.co/datasets/VisualCloze/Graph200K)] <br>


</div>

## 📰 News
- [2025-5-15] 🤗🤗🤗 VisualCloze has been merged into the [**official pipelines of diffusers**](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/visualcloze). For usage guidance, please refer to the [Model Card](https://huggingface.co/VisualCloze/VisualClozePipeline-384).
- [2025-5-18] 🥳🥳🥳 We have released the LoRA weights supporting diffusers at [LoRA Model Card 384](https://huggingface.co/VisualCloze/VisualClozePipeline-LoRA-384) and [LoRA Model Card 512](https://huggingface.co/VisualCloze/VisualClozePipeline-LoRA-512).


## 🌠 Key Features

An in-context learning based universal image generation framework.
1. Support various in-domain tasks. 🔥 [Examples](#supporting-various-in-domain-tasks) 
2. Generalize to <strong><span style="color:hotpink"> unseen tasks</span></strong> through in-context learning.  🔥 [Examples](#generalization-to-unseen-tasks)  
3. Unify multiple tasks into one step and generate both target image and intermediate results.  🔥 [Examples](#consolidate-multi-tasks) 
4. Support reverse-engineering a set of conditions from a target image. 🔥 [Examples](#reverse-generation) 

## 🔥 **Examples**

[![Huggingface VisualCloze](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/VisualCloze/VisualCloze)

### Supporting various in-domain tasks
![Supporting various in-domain tasks](https://github.com/lzyhha/VisualCloze/blob/main/figures/seen.jpg)

### Generalization to unseen tasks

Using in-context examples as task demonstrations to enable the model generalize to unseen tasks.


![Generalization to unseen tasks](https://github.com/lzyhha/VisualCloze/blob/main/figures/face.jpg)

![Generalization to unseen tasks](https://github.com/lzyhha/VisualCloze/blob/main/figures/unseen.jpg)

### Consolidate multi-tasks

Our method can unify multiple tasks into one step and generate not only the target image but also the intermediate results.

![Consolidate multi-tasks](https://github.com/lzyhha/VisualCloze/blob/main/figures/consolidate.jpg)

### Reverse generation

Our method supports reverse generation, 
i.e.,  reverse-engineering a set of conditions from a target. 

![Reverse generation](https://github.com/lzyhha/VisualCloze/blob/main/figures/reverse.jpg)

## 🔧 Dependencies and Installation

See [installation structions](https://github.com/lzyhha/VisualCloze/blob/main/docs/INSTALL.md) for details.

## ⏬ Dataset

We have released the Graph200K dataset in [huggingface](https://huggingface.co/datasets/VisualCloze/Graph200K). 
To use it in VisualCloze, we preprocess it using the [script](https://github.com/lzyhha/VisualCloze/blob/main/processing.py). 

Please refer to [dataset](https://github.com/lzyhha/VisualCloze/blob/main/docs/DATASET.md) for more details.

## 🚀 Training

After preprocessing the Graph200K dataset as shown in [dataset](https://github.com/lzyhha/VisualCloze/blob/main/docs/DATASET.md), 
please setting the `path` item in [visualcloze.yaml](https://github.com/lzyhha/VisualCloze/blob/main/configs/data/visualcloze.yaml) as the generated json file.

```yaml
META:
  -
    path: "the json file of the training set after preprocessing"
    type: 'image_grid_graph200k'
```

Then, you can train the model using a script like [exps/train.sh](https://github.com/lzyhha/VisualCloze/blob/main/exps/train.sh). 
You should personalize `gpu_num`, `batch_size`, and `micro_batch_size` according to your device. 

```bash
bash exps/train.sh
```

For training, we use 8 A100 GPUs with a batch size of 2, requiring 50GB of memory with Fully Sharded Data Parallelism. And gradient accumulation can be employed to support a larger batch size. Additionally, 40GB GPUs can also be used when the batch size is set to 1.

## 💻 Inference

### 1. Download Models

In [huggingface](https://huggingface.co/VisualCloze/VisualCloze), 
we release [visualcloze-384-lora](https://huggingface.co/VisualCloze/VisualCloze/blob/main/visualcloze-384-lora.pth) and [visualcloze-512-lora](https://huggingface.co/VisualCloze/VisualCloze/blob/main/visualcloze-512-lora.pth), 
trained with the grid resolution of 384 and 512, respectively. 
**<span style="color:hotpink">The grid resolution means that each image is resized to the area of the square of it before concatenating images into a grid layout.</span>**


**Note: Apart from the Graph200K, the released models are trained with a part of internal multi-task
datasets, to cover more diverse tasks and improve the generalization ability.**

⭐⭐ Using `huggingface-cli` downloading our model:

**Note**: The weights here are provided for the training, testing, and gradio demo in this repository. For usage with **Diffusers**, please refer to the [Custom Sampling with Diffusers](https://github.com/lzyhha/VisualCloze?tab=readme-ov-file#3-custom-sampling-with-diffusers).

```bash
huggingface-cli download --resume-download VisualCloze/VisualCloze --local-dir /path/to/ckpt
```

or using git for cloning the model you want to use:

```bash
git clone https://huggingface.co/VisualCloze/VisualCloze
```

### 2. Web Demo (Gradio)


To host a local gradio demo for interactive inference, run the following command:

```bash
# By default, we use the model trained under the grid resolution of 384. 
python app.py --model_path "path to downloaded visualcloze-384-lora.pth" --resolution 384

# To use the model with the grid resolution of 512, you should set the resolution parameter to 512.
python app.py --model_path "path to downloaded visualcloze-512-lora.pth" --resolution 512
```

#### Usage Tips
- [SDEdit](https://arxiv.org/abs/2108.01073) is used to upsampling the generated image that has the initial resoluation of 384/512 when using grid resolution of 384/512. You can set `upsampling noise` in the advanced options to adjust the noise levels added to the image. 
For tasks that have strict requirements on the spatial alignment of inputs and outputs, 
you can increase `upsampling noise` or even set it to 1 to disable SDEdit.
- ❗❗❗ Before clicking the generate button, **please wait until all images, prompts, and other components are fully loaded**, especially when using task examples. Otherwise, the inputs from the previous and current sessions may get mixed.


### 3. Custom Sampling with Diffusers

⭐⭐⭐ VisualCloze has been merged into the [**official pipelines of diffusers**](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/visualcloze). 
For usage guidance, please refer to the [Model Card](https://huggingface.co/VisualCloze/VisualClozePipeline-384).
 
First, please install diffusers.

```shell
pip install git+https://github.com/huggingface/diffusers.git
```

Note that chinese users can use the command below to download the model:

```bash
git lfs install
git clone https://www.wisemodel.cn/VisualCloze/VisualClozePipeline-384.git
git clone https://www.wisemodel.cn/VisualCloze/VisualClozePipeline-512.git
```

Then you can use VisualClozePipeline to run the model. 

```python
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image


# Load in-context images (make sure the paths are correct and accessible)
# The images are from the VITON-HD dataset at https://github.com/shadow2496/VITON-HD
image_paths = [
    # in-context examples
    [
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/00700_00.jpg'),
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/03673_00.jpg'),
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/00700_00_tryon_catvton_0.jpg'),
    ],
    # query with the target image
    [
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/00555_00.jpg'),
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/tryon/12265_00.jpg'),
        None
    ],
]

# Task and content prompt
task_prompt = "Each row shows a virtual try-on process that aims to put [IMAGE2] the clothing onto [IMAGE1] the person, producing [IMAGE3] the person wearing the new clothing."
content_prompt = None

# Load the VisualClozePipeline
pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Loading the VisualClozePipeline via LoRA
# pipe = VisualClozePipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", resolution=384, torch_dtype=torch.bfloat16)
# pipe.load_lora_weights('VisualCloze/VisualClozePipeline-LoRA-384', weight_name='visualcloze-lora-384.safetensors')
# pipe.to("cuda")

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_height=1632,
    upsampling_width=1232,
    upsampling_strength=0.3,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

### 4. Custom Sampling without Diffusers

We also implement a pipeline of the visualcloze in [visualcloze.py of this repository](https://github.com/lzyhha/VisualCloze/blob/main/visualcloze.py). 
This can be easily used for custom reasoning. 
In [inference.py](https://github.com/lzyhha/VisualCloze/blob/main/inference.py), we show an example of usage on virtual try-on.

```python
from visualcloze import VisualClozeModel

model = VisualClozeModel(
  model_path="the path of model weigts", 
  resolution=384 or 512, 
  lora_rank=256
)
'''
grid_h: 
The number of in-context examples + 1. 
When without in-context example, it should be set to 1. 

grid_w: 
The number of images involved in a task. 
In the Depth-to-Image task, it is 2. 
In the Virtual Try-On, it is 3.
'''
model.set_grid_size(grid_h, grid_w)
'''
images: 
List[List[PIL.Image.Image]]. A grid-layout image collection, 
each row represents an in-context example or the current query, 
where the current query should be placed in the last row. 
The target image can be None in the input. 
The other images should be the PIL Image class (Image.Image).

prompts: 
List[str]. Three prompts, representing the layout prompt, task prompt, 
and content prompt respectively.
'''
result = model.process_images(
  images, 
  prompts, 
)[-1] # return PIL.Image.Image
```

Execute the usage example and see the output in example.jpg.
```bash
python inference.py --model_path "path to downloaded visualcloze-384-lora.pth" --resolution 384

python inference.py --model_path "path to downloaded visualcloze-512-lora.pth" --resolution 512
```

### 5. Inference on Graph200K test set

To generate images on the test set of the Graph200K, run the following command:
```bash
# Set data_path to the json file of the test set, which is generated when preprocessing.
# Set model_path to the path of model weights.
bash exp/sample.sh
```

You can modify `test_task_dicts` in [prefix_instruction.py](https://github.com/lzyhha/VisualCloze/blob/main/data/prefix_instruction.py) to customize your required tasks.


# 📚 Citation

If you find VisualCloze useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{li2025visualcloze,
  title={VisualCloze : A Universal Image Generation Framework via Visual In-Context Learning},
  author={Li, Zhong-Yu and Du, Ruoyi and Yan, Juncheng and Zhuo, Le and Li, Zhen and Gao, Peng and Ma, Zhanyu and Cheng, Ming-Ming},
  journal={arXiv preprint arxiv:},
  year={2025}
}
```
