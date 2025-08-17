import argparse
from visualcloze import VisualClozeModel
import gradio as gr
import examples
import torch
from functools import partial
from data.prefix_instruction import get_layout_instruction
from PIL import Image


def create_demo(model):
    task_prompt = "Each row shows a virtual try-on process that aims to put [IMAGE2] the clothing onto [IMAGE1] the person, producing [IMAGE3] the person wearing the new clothing."

    with gr.Blocks(title="Virtual Try-On Demo") as demo:
        gr.Markdown("# Virtual Try-On")

        with gr.Row():
            with gr.Column():
                person_image = gr.Image(label="Person", type="pil")
                clothing_image = gr.Image(label="Clothing", type="pil")
                generate_btn = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(label="Output", type="pil")

        def generate_tryon(person, clothing):
            if person is None or clothing is None:
                raise gr.Error("Please upload both a person and a clothing image.")

            # The try-on task uses a 2x3 grid, but for a single query, we only need one row.
            # We can use a dummy in-context example to satisfy the model's input requirements.
            dummy_image = Image.new("RGB", (384, 384))
            images = [
                [dummy_image, dummy_image, dummy_image], # Dummy in-context example
                [person, clothing, None] # Query
            ]

            prompts = [
                get_layout_instruction(3, 2),
                task_prompt,
                ""
            ]

            results = model.process_images(
                images=images,
                prompts=prompts,
                seed=0,
                cfg=30,
                steps=30,
                upsampling_steps=10,
                upsampling_noise=0.4
            )

            return results[0]

        generate_btn.click(
            fn=generate_tryon,
            inputs=[person_image, clothing_image],
            outputs=output_image
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--low_vram", action="store_true", help="Enable low VRAM mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize model
    model = VisualClozeModel(
        resolution=args.resolution,
        model_path=args.model_path,
        precision=args.precision,
        low_vram_mode=args.low_vram
    )

    # Set grid size for the try-on task
    model.set_grid_size(2, 3)

    # Create Gradio demo
    demo = create_demo(model)

    # Start Gradio server
    demo.launch()
