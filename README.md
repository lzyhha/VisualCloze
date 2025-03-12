<p align="center">
  <img src="https://github.com/lzyhha/VisualCloze/blob/main/figures/visualcloze.png" height=100>

</p>

<div align="center">
<h1> VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning </h1>

</div>

<div align="center">

[[Paper]()] &emsp; [[Online Demo]()] &emsp; [[Project Page]()] &emsp; <br>[[🤗 Model Card]()] &emsp; [[🤗 Dataset Card]()] <br>


</div>

## 🌠 Key Features:

An in-context learning based universal image generation framework.
1. Support various in-domain tasks. 🔥 [Examples](#supporting-various-in-domain-tasks) 
2. Generalize to <strong><span style="color:hotpink"> unseen tasks</span></strong> through in-context learning.  🔥 [Examples](#generalization-to-unseen-tasks)  
3. Unify multiple tasks into one step and generate both target image and intermediate results.  🔥 [Examples](#consolidate-multi-tasks) 
4. Support reverse-engineering a set of conditions from a target image. 🔥 [Examples](#reverse-generation) 

## 🔥 **Examples**

[![Huggingface VisualCloze](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](xx)

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

See [installation structions](docs/INSTALL.md) for details.

## ⏬ Dataset

We have released the Graph200K dataset in [huggingface](). 
To use it in VisualCloze, we preprocess it using the [script](). 

Please refer to [dataset](docs/DATASET.md) for more details.

## 🚀 Training

After preprocessing the Graph200K dataset as shown in [dataset](docs/DATASET.md), 
please setting the `path` item in [visualcloze.yaml](configs/data/visualcloze.yaml) as the generated json file. (todo)

```yaml
META:
  -
    path: "the json file of the training set after preprocessing"
    type: 'image_grid_graph200k'
```

Then, you can train the model using a script like [exps/train.sh](exps/train.sh). 
You should personalize `gpu_num`, `batch_size`, and `micro_batch_size` according to your device. 

```bash
bash exps/train.sh
```

## 💻 Inference

### 1. Download Models

In [huggingface](xx), 
we release [visualcloze-384-lora]() and [visualcloze-512-lora](), 
trained with the grid resolution of 384 and 512, respectively. 
**<span style="color:hotpink">The grid resolution means that each image is resized to the area of the square of it before concatenating images into a grid layout.</span>**


**Note: Apart from the Graph200K, the released models are trained with a part of internal multi-task
datasets, to cover more diverse tasks and improve the generalization ability.**

⭐⭐ Using `huggingface-cli` downloading our model:

```bash
huggingface-cli download --resume-download VisualCloze/VisualCloze --local-dir /path/to/ckpt
```

or using git for cloning the model you want to use:
(todo)
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

#### Usage Tips:
- [SDEdit](https://arxiv.org/abs/2108.01073) is used to upsampling the generated image that has the initial resoluation of 384/512 when using grid resolution of 384/512. You can set `upsampling noise` in the advanced options to adjust the noise levels added to the image. 
For tasks that have strict requirements on the spatial alignment of inputs and outputs, 
you can increase `upsampling noise` or even set it to 1 to disable SDEdit.
- ❗❗❗ Before clicking the generate button, **please wait until all images, prompts, and other components are fully loaded**, especially when using task examples. Otherwise, the inputs from the previous and current sessions may get mixed.


### 3. Custom Sampling

We have implement a pipeline of the visualcloze in [visualcloze.py](visualcloze.py). 
This can be easily used for custom reasoning. 
In [inference.py](inference.py), we show an example of usage on virtual try-on.

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

### 4. Inference on Graph200K test set

To generate images on the test set of the Graph200K, run the following command:
```bash
# Set data_path to the json file of the test set, which is generated when preprocessing.
# Set model_path to the path of model weights.
bash exp/sample.sh
```

You can modify `test_task_dicts` in [prefix_instruction.py](data/prefix_instruction.py) to customize your required tasks.


# 📚 Citation

If you find VisualCloze useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{li2025visualcloze,
  title={VisualCloze : A Universal Image Generation Framework via Visual In-Context Learning},
  author={Li, Zhong-Yu and Du, ruoyi and Yan, Juncheng and Zhuo, Le and Li, Zhen and Gao, Peng and Ma, Zhanyu and Cheng, Ming-Ming},
  journal={arXiv preprint arxiv:},
  year={2025}
}
```