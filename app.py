import argparse
from visualcloze import VisualClozeModel
import gradio as gr
import examples
import torch
from functools import partial
from data.prefix_instruction import get_layout_instruction


max_grid_h = 5
max_grid_w = 5
default_grid_h = 2
default_grid_w = 3
default_upsampling_noise = 0.4
default_steps = 30


GUIDANCE = """

## 📋 Quick Start Guide:
1. Adjust **Number of In-context Examples**, 0 disables in-context learning.
2. Set **Task Columns**, the number of images involved in a task.
3. Upload Images. For in-context examples, upload all images. For the current query, upload images exclude the target.
4. Click **Generate** to create the images.
5. Parameters can be fine-tuned under **Advanced Options**.

## 🔥 Task Examples:
Click the task button in the right bottom to acquire **examples** of various tasks. 
Each click on a task may result in different examples. 
**Make sure all images and prompts are loaded before clicking the generate button.**

"""

CITATION = r"""
If you find VisualCloze is helpful, please consider to star ⭐ the <a href='https://github.com/lzyhha/VisualCloze' target='_blank'>Github Repo</a>. Thanks! 
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@article{li2025visualcloze,
  title={VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning},
  author={Li, Zhong-Yu and Du, ruoyi and Yan, Juncheng and Zhuo, Le and Li, Zhen and Gao, Peng and Ma, Zhanyu and Cheng, Ming-Ming},
  journal={arXiv preprint arxiv:},
  year={2025}
}
```
📋 **License**
<br>
This project is licensed under apache-2.0.

📧 **Contact**
<br>
Need help or have questions? Contact us at: lizhongyu [AT] mail.nankai.edu.cn.
"""

NOTE = r"""
❗❗❗ Before clicking the generate button, **please wait until all images, prompts, and other components are fully loaded**, especially when using task examples. Otherwise, the inputs from the previous and current sessions may get mixed.
"""

def create_demo(model):
    with gr.Blocks(title="VisualCloze Demo") as demo:
        gr.Markdown("# VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning")

        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="xxx">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="xxx">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
            <a href="xxx">
                <img src='https://img.shields.io/badge/VisualCloze%20checkpoint-HF%20Model-green?logoColor=violet&label=%F0%9F%A4%97%20Checkpoint'>
            </a>
            <a href="xxx">
                <img src='https://img.shields.io/badge/VisualCloze%20datasets-HF%20Dataset-6B88E3?logoColor=violet&label=%F0%9F%A4%97%20Graph200k%20Dataset'>
            </a>
        </div>
        """)
        
        gr.Markdown(GUIDANCE)

        # Pre-create all possible image components
        all_image_inputs = []
        rows = []
        row_texts = []
        with gr.Row():

            with gr.Column(scale=2):
                # Image grid
                for i in range(max_grid_h):
                    # Add row label before each row
                    row_texts.append(gr.Markdown(
                        "## Query" if i == default_grid_h - 1 else f"## In-context Example {i + 1}",  
                        elem_id=f"row_text_{i}", 
                        visible=i < default_grid_h
                    ))
                    with gr.Row(visible=i < default_grid_h, elem_id=f"row_{i}") as row:
                        rows.append(row)
                        for j in range(max_grid_w):
                            img_input = gr.Image(
                                label=f"In-context Example {i + 1}/{j + 1}" if i != default_grid_h - 1 else f"Query {j + 1}",
                                type="pil",
                                visible= i < default_grid_h and j < default_grid_w,
                                interactive=True,
                                elem_id=f"img_{i}_{j}"
                            )
                            all_image_inputs.append(img_input)

                # Prompts
                layout_prompt = gr.Textbox(
                    label="Layout Description (Auto-filled, Read-only)",
                    placeholder="Layout description will be automatically filled based on grid size...",
                    value=get_layout_instruction(default_grid_w, default_grid_h),
                    elem_id="layout_prompt",
                    interactive=False
                )
                
                task_prompt = gr.Textbox(
                    label="Task Description (Can be modified by referring to examples to perform custom tasks, but may lead to unstable results)",
                    placeholder="Describe what task should be performed...",
                    value="",
                    elem_id="task_prompt"
                )
                
                content_prompt = gr.Textbox(
                    label="(Optional) Content Description (Image caption, Editing instructions, etc.)",
                    placeholder="Describe the content requirements...",
                    value="",
                    elem_id="content_prompt"
                )
                
                generate_btn = gr.Button("Generate", elem_id="generate_btn")
                gr.Markdown(NOTE)

                grid_h = gr.Slider(minimum=0, maximum=max_grid_h-1, value=default_grid_h-1, step=1, label="Number of In-context Examples", elem_id="grid_h")
                grid_w = gr.Slider(minimum=1, maximum=max_grid_w, value=default_grid_w, step=1, label="Task Columns", elem_id="grid_w")
                
                with gr.Accordion("Advanced options", open=False):
                    seed = gr.Number(label="Seed (0 for random)", value=0, precision=0)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=default_steps, step=1)
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=50.0, value=30, step=1)
                    upsampling_steps = gr.Slider(label="Upsampling steps (SDEdit)", minimum=1, maximum=100.0, value=10, step=1)
                    upsampling_noise = gr.Slider(label="Upsampling noise (SDEdit)", minimum=0, maximum=1.0, value=default_upsampling_noise, step=0.05)

                gr.Markdown(CITATION)

            # Output
            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Generated Results",
                    show_label=True,
                    elem_id="output_gallery",
                    columns=None,
                    rows=None,
                    height="auto",
                    allow_preview=True,
                    object_fit="contain"
                )
            
                gr.Markdown("# Task Examples")
                gr.Markdown("Each click on a task may result in different examples.")
                text_dense_prediction_tasks = gr.Textbox(label="Task", visible=False)
                dense_prediction_tasks = gr.Dataset(
                    samples=examples.dense_prediction_text, 
                    label='Dense Prediction', 
                    samples_per_page=1000, 
                    components=[text_dense_prediction_tasks])
                
                text_conditional_generation_tasks = gr.Textbox(label="Task", visible=False)
                conditional_generation_tasks = gr.Dataset(
                    samples=examples.conditional_generation_text, 
                    label='Conditional Generation',
                    samples_per_page=1000, 
                    components=[text_conditional_generation_tasks])
                
                text_image_restoration_tasks = gr.Textbox(label="Task", visible=False)
                image_restoration_tasks = gr.Dataset(
                    samples=examples.image_restoration_text, 
                    label='Image Restoration',
                    samples_per_page=1000, 
                    components=[text_image_restoration_tasks])
                
                text_style_transfer_tasks = gr.Textbox(label="Task", visible=False)
                style_transfer_tasks = gr.Dataset(
                    samples=examples.style_transfer_text, 
                    label='Style Transfer', 
                    samples_per_page=1000, 
                    components=[text_style_transfer_tasks])
                
                text_style_condition_fusion_tasks = gr.Textbox(label="Task", visible=False)
                style_condition_fusion_tasks = gr.Dataset(
                    samples=examples.style_condition_fusion_text, 
                    label='Style Condition Fusion', 
                    samples_per_page=1000, 
                    components=[text_style_condition_fusion_tasks]) 
                
                text_tryon_tasks = gr.Textbox(label="Task", visible=False)
                tryon_tasks = gr.Dataset(
                    samples=examples.tryon_text, 
                    label='Virtual Try-On', 
                    samples_per_page=1000, 
                    components=[text_tryon_tasks])

                text_relighting_tasks = gr.Textbox(label="Task", visible=False)
                relighting_tasks = gr.Dataset(
                    samples=examples.relighting_text, 
                    label='Relighting', 
                    samples_per_page=1000, 
                    components=[text_relighting_tasks])
                
                text_photodoodle_tasks = gr.Textbox(label="Task", visible=False)
                photodoodle_tasks = gr.Dataset(
                    samples=examples.photodoodle_text, 
                    label='Photodoodle', 
                    samples_per_page=1000, 
                    components=[text_photodoodle_tasks])   

                text_editing_tasks = gr.Textbox(label="Task", visible=False)
                editing_tasks = gr.Dataset(
                    samples=examples.editing_text, 
                    label='Editing', 
                    samples_per_page=1000, 
                    components=[text_editing_tasks])

                text_unseen_tasks = gr.Textbox(label="Task", visible=False)
                unseen_tasks = gr.Dataset(
                    samples=examples.unseen_tasks_text, 
                    label='Unseen Tasks (May produce unstable effects)', 
                    samples_per_page=1000, 
                    components=[text_unseen_tasks]) 
                
                gr.Markdown("# Subject-driven Tasks Examples")
                text_subject_driven_tasks = gr.Textbox(label="Task", visible=False)
                subject_driven_tasks = gr.Dataset(
                    samples=examples.subject_driven_text, 
                    label='Subject-driven Generation', 
                    samples_per_page=1000, 
                    components=[text_subject_driven_tasks])
                
                text_condition_subject_fusion_tasks = gr.Textbox(label="Task", visible=False)
                condition_subject_fusion_tasks = gr.Dataset(
                    samples=examples.condition_subject_fusion_text, 
                    label='Condition+Subject Fusion', 
                    samples_per_page=1000, 
                    components=[text_condition_subject_fusion_tasks])
                
                text_style_transfer_with_subject_tasks = gr.Textbox(label="Task", visible=False)
                style_transfer_with_subject_tasks = gr.Dataset(
                    samples=examples.style_transfer_with_subject_text, 
                    label='Style Transfer with Subject', 
                    samples_per_page=1000, 
                    components=[text_style_transfer_with_subject_tasks])

                text_condition_subject_style_fusion_tasks = gr.Textbox(label="Task", visible=False)
                condition_subject_style_fusion_tasks = gr.Dataset(
                    samples=examples.condition_subject_style_fusion_text, 
                    label='Condition+Subject+Style Fusion', 
                    samples_per_page=1000, 
                    components=[text_condition_subject_style_fusion_tasks])

                text_editing_with_subject_tasks = gr.Textbox(label="Task", visible=False)
                editing_with_subject_tasks = gr.Dataset(
                    samples=examples.editing_with_subject_text, 
                    label='Editing with Subject', 
                    samples_per_page=1000, 
                    components=[text_editing_with_subject_tasks])

                text_image_restoration_with_subject_tasks = gr.Textbox(label="Task", visible=False)
                image_restoration_with_subject_tasks = gr.Dataset(
                    samples=examples.image_restoration_with_subject_text, 
                    label='Image Restoration with Subject', 
                    samples_per_page=1000, 
                    components=[text_image_restoration_with_subject_tasks])
                
        def update_grid(h, w):
            actual_h = h + 1
            model.set_grid_size(actual_h, w)
            
            updates = []
            
            # Update image component visibility
            for i in range(max_grid_h * max_grid_w):
                curr_row = i // max_grid_w
                curr_col = i % max_grid_w
                updates.append(
                    gr.update(
                        label=f"In-context Example {curr_row + 1}/{curr_col + 1}" if curr_row != actual_h - 1 else f"Query {curr_col + 1}",
                        elem_id=f"img_{curr_row}_{curr_col}", 
                        visible=(curr_row < actual_h and curr_col < w)))
            
            # Update row visibility and labels
            updates_row = []
            updates_row_text = []
            for i in range(max_grid_h):
                updates_row.append(gr.update(f"row_{i}", visible=(i < actual_h)))
                updates_row_text.append(
                    gr.update(
                        elem_id=f"row_text_{i}", 
                        visible=i < actual_h, 
                        value="## Query" if i == actual_h - 1 else f"## In-context Example {i + 1}", 
                    )
                )
            
            updates.extend(updates_row)
            updates.extend(updates_row_text)
            updates.append(gr.update(elem_id="layout_prompt", value=get_layout_instruction(w, actual_h)))
            return updates

        def generate_image(*inputs):
            images = []
            if grid_h.value + 1 != model.grid_h or grid_w.value != model.grid_w:
                raise gr.Error('Please wait for the loading to complete.')
            for i in range(model.grid_h):
                images.append([])
                for j in range(model.grid_w):
                    images[i].append(inputs[i * max_grid_w + j])
                    if i != model.grid_h - 1:
                        if inputs[i * max_grid_w + j] is None:
                            raise gr.Error('Please upload in-context examples. Possible that the task examples have not finished loading yet.')
            seed, cfg, steps, upsampling_steps, upsampling_noise, layout_text, task_text, content_text = inputs[-8:]
            
            try:
                results = generate(
                    images, 
                    [layout_text, task_text, content_text], 
                    seed=seed, cfg=cfg, steps=steps, 
                    upsampling_steps=upsampling_steps, upsampling_noise=upsampling_noise
                )
            except Exception as e:
                raise gr.Error('Process error. Possible that the task examples have not finished loading yet. Error: ' + str(e))

            output = gr.update(
                elem_id='output_gallery', 
                value=results, 
                columns=min(len(results), 2),
                rows=int(len(results) / 2 + 0.5))
            
            return output

        def process_tasks(task, func):
            outputs = func(task)
            mask = outputs[0]
            state = outputs[1:8]
            if state[5] is None:
                state[5] = default_upsampling_noise
            if state[6] is None:
                state[6] = default_steps
            images = outputs[8:-len(mask)]
            output = outputs[-len(mask):]
            for i in range(len(mask)):
                if mask[i] == 1:
                    images.append(None)
                else:
                    images.append(output[-len(mask) + i])
            
            state[0] = state[0] - 1
            cur_hrid_h = state[0]
            cur_hrid_w = state[1]

            current_example = [None] * 25
            for i, image in enumerate(images):
                pos = (i // cur_hrid_w) * 5 + (i % cur_hrid_w)
                if image is not None:
                    current_example[pos] = image
            update_grid(cur_hrid_h, cur_hrid_w)
            output = gr.update(
                elem_id='output_gallery', 
                value=[o for o, m in zip(output, mask) if m == 1], 
                columns=min(sum(mask), 2),
                rows=int(sum(mask) / 2 + 0.5))
            return [output] + current_example + state
        
        dense_prediction_tasks.click(
            partial(process_tasks, func=examples.process_dense_prediction_tasks), 
            inputs=[dense_prediction_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full", 
            show_progress_on=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps] + [generate_btn])

        conditional_generation_tasks.click(
            partial(process_tasks, func=examples.process_conditional_generation_tasks), 
            inputs=[conditional_generation_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        image_restoration_tasks.click(
            partial(process_tasks, func=examples.process_image_restoration_tasks), 
            inputs=[image_restoration_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        style_transfer_tasks.click(
            partial(process_tasks, func=examples.process_style_transfer_tasks), 
            inputs=[style_transfer_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        style_condition_fusion_tasks.click(
            partial(process_tasks, func=examples.process_style_condition_fusion_tasks), 
            inputs=[style_condition_fusion_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")
            
        relighting_tasks.click(
            partial(process_tasks, func=examples.process_relighting_tasks), 
            inputs=[relighting_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        tryon_tasks.click(
            partial(process_tasks, func=examples.process_tryon_tasks), 
            inputs=[tryon_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")
        
        photodoodle_tasks.click(
            partial(process_tasks, func=examples.process_photodoodle_tasks), 
            inputs=[photodoodle_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        editing_tasks.click(
            partial(process_tasks, func=examples.process_editing_tasks), 
            inputs=[editing_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")
        
        unseen_tasks.click(
            partial(process_tasks, func=examples.process_unseen_tasks), 
            inputs=[unseen_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        subject_driven_tasks.click(
            partial(process_tasks, func=examples.process_subject_driven_tasks), 
            inputs=[subject_driven_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        style_transfer_with_subject_tasks.click(
            partial(process_tasks, func=examples.process_style_transfer_with_subject_tasks), 
            inputs=[style_transfer_with_subject_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        condition_subject_fusion_tasks.click(
            partial(process_tasks, func=examples.process_condition_subject_fusion_tasks), 
            inputs=[condition_subject_fusion_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        condition_subject_style_fusion_tasks.click(
            partial(process_tasks, func=examples.process_condition_subject_style_fusion_tasks), 
            inputs=[condition_subject_style_fusion_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        editing_with_subject_tasks.click(
            partial(process_tasks, func=examples.process_editing_with_subject_tasks), 
            inputs=[editing_with_subject_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")

        image_restoration_with_subject_tasks.click(
            partial(process_tasks, func=examples.process_image_restoration_with_subject_tasks), 
            inputs=[image_restoration_with_subject_tasks], 
            outputs=[output_gallery] + all_image_inputs + [grid_h, grid_w, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps], 
            show_progress="full")
        # Initialize grid
        model.set_grid_size(default_grid_h, default_grid_w)
        
        # Connect event processing function to all components that need updating
        output_components = all_image_inputs + rows + row_texts + [layout_prompt]

        grid_h.change(fn=update_grid, inputs=[grid_h, grid_w], outputs=output_components)
        grid_w.change(fn=update_grid, inputs=[grid_h, grid_w], outputs=output_components)
        
        # Modify generate button click event
        generate_btn.click(
            fn=generate_image,
            inputs=all_image_inputs + [seed, cfg, steps, upsampling_steps, upsampling_noise] + [layout_prompt, task_prompt, content_prompt],
            outputs=output_gallery
        )
        
    return demo


def generate(
    images, 
    prompts, 
    seed, cfg, steps, 
    upsampling_steps, upsampling_noise):
    with torch.no_grad():
        return model.process_images(
            images=images, 
            prompts=prompts, 
            seed=seed, 
            cfg=cfg, 
            steps=steps, 
            upsampling_steps=upsampling_steps, 
            upsampling_noise=upsampling_noise)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--resolution", type=int, default=384)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize model
    model = VisualClozeModel(resolution=args.resolution, model_path=args.model_path, precision=args.precision)
    
    # Create Gradio demo
    demo = create_demo(model)
    
    # Start Gradio server
    demo.launch()